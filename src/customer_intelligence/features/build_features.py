"""Feature engineering pipeline for segmentation, churn, and LTV models."""

from __future__ import annotations

import pandas as pd

TELCO_NUMERIC_DEFAULTS = {
    "tenure_months": 0,
    "senior_citizen": 0,
    "service_count": 0,
    "has_fiber": 0,
    "month_to_month_contract": 0,
    "electronic_check": 0,
    "automatic_payment": 0,
    "paperless_billing_flag": 0,
    "tech_support_flag": 0,
    "online_security_flag": 0,
}

TELCO_CATEGORICAL_DEFAULTS = {
    "gender": "Unknown",
    "partner": "Unknown",
    "dependents": "Unknown",
    "phone_service": "Unknown",
    "multiple_lines": "Unknown",
    "internet_service": "Unknown",
    "online_security": "Unknown",
    "online_backup": "Unknown",
    "device_protection": "Unknown",
    "tech_support": "Unknown",
    "streaming_tv": "Unknown",
    "streaming_movies": "Unknown",
    "contract": "Unknown",
    "paperless_billing": "Unknown",
    "payment_method": "Unknown",
}


def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute recency, frequency, monetary, and quantile score features."""

    out = df.copy()
    out["recency"] = out["days_since_last_purchase"]
    out["frequency"] = out["purchases_12m"]
    out["monetary"] = out["total_revenue_12m"]
    out["r_score"] = pd.qcut(out["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["f_score"] = pd.qcut(out["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["m_score"] = pd.qcut(out["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["rfm_score"] = out["r_score"] * 100 + out["f_score"] * 10 + out["m_score"]
    return out


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready behavioral, value, lifecycle, and support features."""

    out = add_rfm_features(df)
    for column, default in TELCO_NUMERIC_DEFAULTS.items():
        if column not in out:
            out[column] = default
    for column, default in TELCO_CATEGORICAL_DEFAULTS.items():
        if column not in out:
            out[column] = default
    out["revenue_per_purchase"] = out["total_revenue_12m"] / out["purchases_12m"].clip(lower=1)
    out["engagement_intensity"] = out["sessions_30d"] / out["customer_age_days"].clip(lower=30) * 30
    out["support_load"] = out["support_tickets_90d"] / out["sessions_30d"].clip(lower=1)
    out["avg_revenue_per_tenure_month"] = out["total_revenue_12m"] / out["tenure_months"].clip(lower=1)
    out["service_density"] = out["service_count"] / out["tenure_months"].clip(lower=1)
    out["contract_risk_score"] = (
        out["month_to_month_contract"] * 2
        + out["electronic_check"]
        + out["has_fiber"]
        - out["automatic_payment"]
        - out["tech_support_flag"]
    )
    out["is_high_value"] = (out["monetary"] >= out["monetary"].quantile(0.8)).astype(int)
    out["is_inactive"] = (out["recency"] >= 60).astype(int)
    out["nps_bucket"] = pd.cut(
        out["nps"],
        bins=[-101, 0, 30, 70, 101],
        labels=["detractor", "passive_low", "passive_high", "promoter"],
    ).astype(str)
    return out


MODEL_FEATURES = [
    "customer_age_days",
    "monthly_spend",
    "purchases_12m",
    "sessions_30d",
    "avg_session_minutes",
    "support_tickets_90d",
    "days_since_last_purchase",
    "discount_usage_rate",
    "nps",
    "total_revenue_12m",
    "recency",
    "frequency",
    "monetary",
    "rfm_score",
    "revenue_per_purchase",
    "engagement_intensity",
    "support_load",
    "tenure_months",
    "senior_citizen",
    "service_count",
    "has_fiber",
    "month_to_month_contract",
    "electronic_check",
    "automatic_payment",
    "paperless_billing_flag",
    "tech_support_flag",
    "online_security_flag",
    "avg_revenue_per_tenure_month",
    "service_density",
    "contract_risk_score",
    "is_high_value",
    "is_inactive",
]
