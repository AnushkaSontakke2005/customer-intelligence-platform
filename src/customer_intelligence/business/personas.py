"""Customer persona generation from behavioral segments."""

from __future__ import annotations

import pandas as pd


def generate_personas(df: pd.DataFrame, segment_col: str = "kmeans_segment") -> pd.DataFrame:
    """Summarize each segment as an interview-ready customer persona."""

    rows = []
    for segment, group in df.groupby(segment_col):
        value = group["monetary"].mean()
        recency = group["recency"].mean()
        engagement = group["sessions_30d"].mean()
        support = group["support_tickets_90d"].mean()
        if value >= df["monetary"].quantile(0.75) and engagement >= df["sessions_30d"].median():
            persona = "Premium Loyalists"
            strategy = "Protect with concierge support, loyalty perks, and early-access benefits."
        elif recency >= 60 and support >= 1:
            persona = "Frustrated At-Risk Customers"
            strategy = "Route to service recovery, root-cause fixes, and manager-level outreach."
        elif engagement < df["sessions_30d"].median():
            persona = "Dormant Trialists"
            strategy = "Trigger activation journeys, onboarding nudges, and habit-building content."
        else:
            persona = "Value Seekers"
            strategy = "Use bundled offers, pricing education, and personalized product discovery."
        rows.append(
            {
                "segment": int(segment) if str(segment).lstrip("-").isdigit() else segment,
                "persona": persona,
                "customers": len(group),
                "avg_annual_value": round(value, 2),
                "avg_recency_days": round(recency, 1),
                "avg_sessions_30d": round(engagement, 1),
                "recommended_strategy": strategy,
            }
        )
    return pd.DataFrame(rows).sort_values("customers", ascending=False)
