"""Executive KPI calculations."""

from __future__ import annotations

import pandas as pd


def calculate_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Return board-level customer health metrics."""

    customers = len(df)
    revenue = float(df["total_revenue_12m"].sum())
    churn_rate = float(df["churned"].mean()) if "churned" in df else 0.0
    arpu = revenue / customers if customers else 0.0
    high_value_at_risk = float(
        df.loc[(df.get("is_high_value", 0) == 1) & (df.get("churned", 0) == 1), "total_revenue_12m"].sum()
    )
    return {
        "customers": customers,
        "annual_revenue": revenue,
        "churn_rate": churn_rate,
        "arpu": arpu,
        "high_value_revenue_at_risk": high_value_at_risk,
    }
