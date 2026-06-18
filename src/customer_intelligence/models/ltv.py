"""Customer lifetime value prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_intelligence.config import MODEL_CONFIG
from customer_intelligence.features.build_features import MODEL_FEATURES
from customer_intelligence.models.churn import CATEGORICAL_FEATURES


def build_ltv_target(df: pd.DataFrame) -> pd.Series:
    """Proxy next-year CLV as expected gross retained revenue."""

    churn_risk = df.get("churn_probability", df.get("churned", 0))
    return df["total_revenue_12m"] * (1 - churn_risk) * 1.18


def train_ltv_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    """Train a robust CLV regressor and return holdout metrics."""

    data = df.copy()
    data["ltv_target"] = build_ltv_target(data)
    X = data[MODEL_FEATURES + CATEGORICAL_FEATURES]
    y = data["ltv_target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=MODEL_CONFIG.test_size, random_state=MODEL_CONFIG.random_state
    )
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), MODEL_FEATURES),
            ("categorical", encoder, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(random_state=MODEL_CONFIG.random_state)),
        ]
    )
    fallback = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=MODEL_CONFIG.random_state)),
        ]
    )
    try:
        model.fit(X_train, y_train)
    except Exception:
        model = fallback.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, {"mae": float(mean_absolute_error(y_test, predictions)), "r2": float(r2_score(y_test, predictions))}


def score_ltv(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Append predicted lifetime value."""

    scored = df.copy()
    scored["predicted_ltv"] = model.predict(scored[MODEL_FEATURES + CATEGORICAL_FEATURES]).round(2)
    return scored
