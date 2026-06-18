"""Retention simulator dashboard page."""

import pandas as pd
import streamlit as st

from dashboard.app_utils import SCORES_PATH, ensure_dashboard_outputs
from customer_intelligence.business.revenue_simulator import simulate_retention_impact


st.title("Revenue Impact Simulator")
ensure_dashboard_outputs()

df = pd.read_csv(SCORES_PATH)
target_top_pct = st.slider("Target top risk-value percentile", 0.05, 0.50, 0.20, 0.05)
success_rate = st.slider("Expected retention success rate", 0.05, 0.40, 0.18, 0.01)
result = simulate_retention_impact(df, target_top_pct=target_top_pct, success_rate=success_rate)

cols = st.columns(5)
cols[0].metric("Targeted Customers", f"{result['targeted_customers']:,}")
cols[1].metric("Saved Revenue", f"${result['saved_revenue']:,.0f}")
cols[2].metric("Gross Profit", f"${result['incremental_gross_profit']:,.0f}")
cols[3].metric("Campaign Cost", f"${result['campaign_cost']:,.0f}")
cols[4].metric("Net Impact", f"${result['net_impact']:,.0f}")
st.json(result)
