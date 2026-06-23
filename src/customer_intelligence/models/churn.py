"""Churn modeling with classical and gradient-boosted algorithms."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_intelligence.config import BUSINESS_CONFIG
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


def calculate_business_threshold(
    y_true: pd.Series,
    y_prob,
    fp_cost: float = BUSINESS_CONFIG.retention_offer_cost,
    fn_cost: float = 150.0,
) -> dict[str, float]:
    """Find the threshold that minimizes retention campaign business cost.

    False positives waste outreach budget. False negatives miss likely churners
    and lose expected revenue. This turns model selection from a pure ML metric
    exercise into a business decision.
    """

    y_true_array = pd.Series(y_true).to_numpy()
    precisions, recalls, thresholds = precision_recall_curve(y_true_array, y_prob)
    if len(thresholds) == 0:
        thresholds = [0.5]

    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        fp = int(((y_pred == 1) & (y_true_array == 0)).sum())
        fn = int(((y_pred == 0) & (y_true_array == 1)).sum())
        cost = fp * fp_cost + fn * fn_cost
        rows.append((float(threshold), fp, fn, float(cost), y_pred))

    optimal_threshold, fp, fn, min_cost, optimal_pred = min(rows, key=lambda item: item[3])
    default_pred = (y_prob >= 0.5).astype(int)
    default_fp = int(((default_pred == 1) & (y_true_array == 0)).sum())
    default_fn = int(((default_pred == 0) & (y_true_array == 1)).sum())
    default_cost = float(default_fp * fp_cost + default_fn * fn_cost)

    return {
        "optimal_threshold": optimal_threshold,
        "business_cost_at_optimal_threshold": min_cost,
        "business_cost_at_default_threshold": default_cost,
        "business_cost_reduction": default_cost - min_cost,
        "false_positives_at_optimal_threshold": fp,
        "false_negatives_at_optimal_threshold": fn,
        "precision_at_optimal_threshold": float(precision_score(y_true_array, optimal_pred, zero_division=0)),
        "recall_at_optimal_threshold": float(recall_score(y_true_array, optimal_pred, zero_division=0)),
        "f1_at_optimal_threshold": float(f1_score(y_true_array, optimal_pred, zero_division=0)),
        "fp_cost": float(fp_cost),
        "fn_cost": float(fn_cost),
    }


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
        threshold_metrics = calculate_business_threshold(y_test, proba)
        rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test, proba),
                "average_precision": average_precision_score(y_test, proba),
                "f1": f1_score(y_test, pred),
                **threshold_metrics,
            }
        )
        trained[name] = model
    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    return trained, metrics, str(metrics.iloc[0]["model"])


def score_churn(df: pd.DataFrame, model: Pipeline, decision_threshold: float = 0.5) -> pd.DataFrame:
    """Append churn probability and risk tier."""

    scored = df.copy()
    scored["churn_probability"] = model.predict_proba(scored[MODEL_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    scored["decision_threshold"] = decision_threshold
    scored["retention_priority"] = scored["churn_probability"] >= decision_threshold
    scored["risk_tier"] = pd.cut(
        scored["churn_probability"],
        bins=[0, 0.35, 0.65, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return scored
