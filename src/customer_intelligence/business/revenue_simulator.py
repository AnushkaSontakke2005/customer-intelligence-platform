"""Revenue impact simulator for retention scenarios."""

from __future__ import annotations

import pandas as pd

from customer_intelligence.config import BUSINESS_CONFIG


def simulate_retention_impact(
    df: pd.DataFrame,
    target_top_pct: float = 0.2,
    success_rate: float = BUSINESS_CONFIG.proactive_success_rate,
) -> dict[str, float]:
    """Estimate incremental gross profit from prioritizing highest-risk customers."""

    scored = df.copy()
    if "churn_probability" not in scored:
        scored["churn_probability"] = scored.get("churned", 0)
    scored["risk_value"] = scored["churn_probability"] * scored["total_revenue_12m"]
    n_targeted = max(1, int(len(scored) * target_top_pct))
    targeted = scored.nlargest(n_targeted, "risk_value")
    saved_revenue = float((targeted["risk_value"] * success_rate).sum())
    gross_profit = saved_revenue * BUSINESS_CONFIG.average_gross_margin
    campaign_cost = n_targeted * BUSINESS_CONFIG.retention_offer_cost
    return {
        "targeted_customers": n_targeted,
        "saved_revenue": round(saved_revenue, 2),
        "incremental_gross_profit": round(gross_profit, 2),
        "campaign_cost": round(campaign_cost, 2),
        "net_impact": round(gross_profit - campaign_cost, 2),
        "roi": round((gross_profit - campaign_cost) / campaign_cost, 2) if campaign_cost else 0.0,
    }
