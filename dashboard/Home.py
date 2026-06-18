"""Streamlit executive dashboard home page."""

import pandas as pd
import streamlit as st

from app_utils import SCORES_PATH, ensure_dashboard_outputs
from customer_intelligence.business.kpis import calculate_kpis
from customer_intelligence.business.revenue_simulator import simulate_retention_impact


st.set_page_config(page_title="Customer Intelligence Platform", layout="wide")
st.title("Customer Intelligence Platform")
st.caption("Executive view of customer health, churn risk, lifetime value, and retention ROI.")

ensure_dashboard_outputs()

df = pd.read_csv(SCORES_PATH)
kpis = calculate_kpis(df)
simulation = simulate_retention_impact(df)

cols = st.columns(5)
cols[0].metric("Customers", f"{kpis['customers']:,}")
cols[1].metric("Annual Revenue", f"${kpis['annual_revenue']:,.0f}")
cols[2].metric("Churn Rate", f"{kpis['churn_rate']:.1%}")
cols[3].metric("ARPU", f"${kpis['arpu']:,.0f}")
cols[4].metric("Retention ROI", f"{simulation['roi']:.1f}x")

left, right = st.columns(2)
with left:
    st.subheader("Risk Tier Distribution")
    st.bar_chart(df["risk_tier"].value_counts())
with right:
    st.subheader("Revenue by Segment")
    st.bar_chart(df.groupby("kmeans_segment")["total_revenue_12m"].sum())

st.subheader("Top Retention Opportunities")
st.dataframe(
    df.sort_values("expected_saved_revenue", ascending=False)[
        ["customer_id", "risk_tier", "churn_probability", "predicted_ltv", "retention_action", "expected_saved_revenue"]
    ].head(25),
    use_container_width=True,
)
