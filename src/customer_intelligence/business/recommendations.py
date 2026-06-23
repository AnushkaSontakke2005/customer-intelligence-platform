"""Retention recommendation engine."""

from __future__ import annotations

import pandas as pd

from customer_intelligence.config import BUSINESS_CONFIG


def assign_retention_action(row: pd.Series) -> str:
    """Map customer risk/value drivers to a retention playbook."""

    risk = float(row.get("churn_probability", row.get("churned", 0)))
    high_ltv = bool(row.get("is_high_ltv", row.get("is_high_value", 0)))
    if risk < BUSINESS_CONFIG.medium_risk_threshold:
        return "nurture_with_personalized_content"
    if row.get("support_tickets_90d", 0) >= 2:
        return "priority_support_recovery"
    if high_ltv:
        return "concierge_success_outreach"
    if row.get("discount_usage_rate", 0) > 0.35:
        return "targeted_value_offer"
    if row.get("sessions_30d", 0) < 3:
        return "activation_journey"
    return "winback_bundle"


def build_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Attach recommended action, expected saved revenue, and ROI."""

    out = df.copy()
    value_basis = out["predicted_ltv"] if "predicted_ltv" in out else out["total_revenue_12m"]
    out["is_high_ltv"] = (value_basis >= value_basis.quantile(0.8)).astype(int)
    out["retention_action"] = out.apply(assign_retention_action, axis=1)
    risk = out.get("churn_probability", out.get("churned", 0))
    out["expected_saved_revenue"] = (
        risk
        * value_basis
        * BUSINESS_CONFIG.proactive_success_rate
    ).round(2)
    out["intervention_cost"] = BUSINESS_CONFIG.retention_offer_cost
    out["retention_roi"] = ((out["expected_saved_revenue"] - out["intervention_cost"]) / out["intervention_cost"]).round(2)
    return out
