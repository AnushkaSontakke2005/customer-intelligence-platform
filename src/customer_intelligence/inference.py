"""Batch and API inference helpers."""

from __future__ import annotations

import joblib
import pandas as pd

from customer_intelligence.business.recommendations import build_recommendations
from customer_intelligence.config import MODEL_DIR
from customer_intelligence.features.build_features import build_customer_features
from customer_intelligence.models.churn import score_churn
from customer_intelligence.models.ltv import score_ltv


def load_models() -> dict[str, object]:
    """Load production model artifacts."""

    return {
        "churn": joblib.load(MODEL_DIR / "best_churn_model.joblib"),
        "ltv": joblib.load(MODEL_DIR / "ltv_model.joblib"),
    }


def score_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering, churn scoring, LTV scoring, and recommendations."""

    models = load_models()
    features = build_customer_features(df)
    scored = score_churn(features, models["churn"])
    scored = score_ltv(scored, models["ltv"])
    return build_recommendations(scored)
