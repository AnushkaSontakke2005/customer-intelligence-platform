"""Customer lifetime value prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_intelligence.config import BUSINESS_CONFIG
from customer_intelligence.config import MODEL_CONFIG
from customer_intelligence.features.build_features import MODEL_FEATURES
from customer_intelligence.models.churn import CATEGORICAL_FEATURES


LTV_NUMERIC_FEATURES = [
    "tenure_months",
    "senior_citizen",
    "service_count",
    "support_load",
]

LTV_CATEGORICAL_FEATURES = [
    "partner",
    "dependents",
    "internet_service",
    "online_security",
    "tech_support",
    "contract",
    "payment_method",
]

LTV_FEATURES = LTV_NUMERIC_FEATURES + LTV_CATEGORICAL_FEATURES

REVENUE_LEAKAGE_COLUMNS = [
    "monthly_spend",
    "total_revenue_12m",
    "monetary",
    "rfm_score",
    "revenue_per_purchase",
    "avg_revenue_per_tenure_month",
    "avg_session_minutes",
]


def build_ltv_target(df: pd.DataFrame) -> pd.Series:
    """Build expected margin LTV from revenue and survival uncertainty.

    ``monthly_spend`` is used to construct the target because LTV is an
    economic quantity. It is intentionally excluded from the LTV features so
    the model must infer value from contract, tenure, service adoption, and
    support/stickiness signals instead of recovering an algebraic formula.
    """

    churn_risk = df.get("churn_probability", df.get("churned", 0.5)).clip(lower=1e-6)
    expected_remaining_months = 1 / (churn_risk + 1e-6)
    monthly_margin = df["monthly_spend"] * BUSINESS_CONFIG.average_gross_margin
    return (monthly_margin * expected_remaining_months).clip(lower=0)


def build_leaky_ltv_target(df: pd.DataFrame) -> pd.Series:
    """Previous target style kept only for side-by-side audit reporting."""

    churn_risk = df.get("churn_probability", df.get("churned", 0.5))
    return df["total_revenue_12m"] * (1 - churn_risk) * 1.18


def _build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", encoder, categorical_features),
        ]
    )


def _build_regressor(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(random_state=MODEL_CONFIG.random_state)),
        ]
    )


def _fit_regressor(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    fallback = Pipeline(
        [
            ("preprocessor", model.named_steps["preprocessor"]),
            ("model", RandomForestRegressor(n_estimators=200, random_state=MODEL_CONFIG.random_state)),
        ]
    )
    try:
        return model.fit(X_train, y_train)
    except Exception:
        return fallback.fit(X_train, y_train)


def _evaluate_predictions(y_true: pd.Series, predictions) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def _top_permutation_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, float]]:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=MODEL_CONFIG.random_state,
        scoring="r2",
    )
    rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(X_test.columns, result.importances_mean)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:5]


def _target_correlations(data: pd.DataFrame, target: pd.Series) -> list[dict[str, object]]:
    encoded = pd.get_dummies(data[LTV_FEATURES], drop_first=False)
    correlations = encoded.corrwith(target).fillna(0).abs().sort_values(ascending=False)
    return [
        {
            "feature": feature,
            "pearson_abs_corr_with_ltv": float(correlation),
            "leakage_flag": bool(correlation > 0.95),
        }
        for feature, correlation in correlations.items()
    ]


def train_ltv_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    """Train a robust CLV regressor and return holdout metrics."""

    data = df.copy()
    data["ltv_target"] = build_ltv_target(data)
    X = data[LTV_FEATURES]
    y = data["ltv_target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=MODEL_CONFIG.test_size, random_state=MODEL_CONFIG.random_state
    )

    model = _build_regressor(_build_preprocessor(LTV_NUMERIC_FEATURES, LTV_CATEGORICAL_FEATURES))
    model = _fit_regressor(model, X_train, y_train)
    predictions = model.predict(X_test)

    leaky_X = data[MODEL_FEATURES + CATEGORICAL_FEATURES]
    leaky_y = build_leaky_ltv_target(data)
    leaky_X_train, leaky_X_test, leaky_y_train, leaky_y_test = train_test_split(
        leaky_X, leaky_y, test_size=MODEL_CONFIG.test_size, random_state=MODEL_CONFIG.random_state
    )
    leaky_model = _build_regressor(_build_preprocessor(MODEL_FEATURES, CATEGORICAL_FEATURES))
    leaky_model = _fit_regressor(leaky_model, leaky_X_train, leaky_y_train)
    leaky_predictions = leaky_model.predict(leaky_X_test)

    metrics = _evaluate_predictions(y_test, predictions)
    metrics.update(
        {
            "ltv_version": "v2_decoupled",
            "target_definition": "monthly_margin / churn_probability",
            "excluded_revenue_features": REVENUE_LEAKAGE_COLUMNS,
            "top_features": _top_permutation_importance(model, X_test, y_test),
            "target_correlations": _target_correlations(data, y),
            "leakage_flags": [
                row for row in _target_correlations(data, y) if row["leakage_flag"]
            ],
            "comparison": {
                "old_leaky_target": {
                    "target": "total_revenue_12m * (1 - churn_probability) * 1.18",
                    "feature_set": "MODEL_FEATURES, including revenue-derived columns",
                    **_evaluate_predictions(leaky_y_test, leaky_predictions),
                },
                "new_decoupled_target": {
                    "target": "monthly_spend * 0.62 / (churn_probability + 1e-6)",
                    "feature_set": "stickiness, contract, support, service adoption only",
                    **metrics,
                },
            },
        }
    )
    return model, metrics


def score_ltv(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Append predicted lifetime value."""

    scored = df.copy()
    scored["predicted_ltv"] = model.predict(scored[LTV_FEATURES]).round(2)
    return scored
