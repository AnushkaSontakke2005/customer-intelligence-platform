"""Churn modeling with classical and gradient-boosted algorithms."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_intelligence.config import MODEL_CONFIG
from customer_intelligence.features.build_features import MODEL_FEATURES


CATEGORICAL_FEATURES = ["region", "acquisition_channel", "plan_type", "nps_bucket"]
TELCO_CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]
CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + TELCO_CATEGORICAL_FEATURES


def _optional_classifier(package: str, class_name: str):
    try:
        module = __import__(package, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:
        return None


def build_preprocessor() -> ColumnTransformer:
    """Build model preprocessing for numeric and categorical features."""

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), MODEL_FEATURES),
            ("categorical", encoder, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict[str, Any]:
    """Return requested model family candidates with optional dependency fallbacks."""

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=MODEL_CONFIG.random_state,
            n_jobs=-1,
        ),
    }

    XGBClassifier = _optional_classifier("xgboost", "XGBClassifier")
    models["xgboost"] = (
        XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=MODEL_CONFIG.random_state,
        )
        if XGBClassifier
        else RandomForestClassifier(n_estimators=180, random_state=MODEL_CONFIG.random_state, n_jobs=-1)
    )

    LGBMClassifier = _optional_classifier("lightgbm", "LGBMClassifier")
    models["lightgbm"] = (
        LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            random_state=MODEL_CONFIG.random_state,
        )
        if LGBMClassifier
        else RandomForestClassifier(n_estimators=180, random_state=MODEL_CONFIG.random_state, n_jobs=-1)
    )

    CatBoostClassifier = _optional_classifier("catboost", "CatBoostClassifier")
    models["catboost"] = (
        CatBoostClassifier(
            iterations=250,
            learning_rate=0.05,
            depth=5,
            verbose=False,
            random_seed=MODEL_CONFIG.random_state,
        )
        if CatBoostClassifier
        else RandomForestClassifier(n_estimators=180, random_state=MODEL_CONFIG.random_state, n_jobs=-1)
    )
    return models


def train_churn_models(df: pd.DataFrame) -> tuple[dict[str, Pipeline], pd.DataFrame, str]:
    """Train all churn candidates and select the highest ROC-AUC model."""

    X = df[MODEL_FEATURES + CATEGORICAL_FEATURES]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=MODEL_CONFIG.test_size, random_state=MODEL_CONFIG.random_state, stratify=y
    )
    trained: dict[str, Pipeline] = {}
    rows = []
    for name, estimator in candidate_models().items():
        model = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test, proba),
                "average_precision": average_precision_score(y_test, proba),
                "f1": f1_score(y_test, pred),
            }
        )
        trained[name] = model
    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    return trained, metrics, str(metrics.iloc[0]["model"])


def score_churn(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Append churn probability and risk tier."""

    scored = df.copy()
    scored["churn_probability"] = model.predict_proba(scored[MODEL_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    scored["risk_tier"] = pd.cut(
        scored["churn_probability"],
        bins=[0, 0.35, 0.65, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return scored
