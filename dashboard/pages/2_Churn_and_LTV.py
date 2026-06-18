"""Churn and LTV dashboard page."""

import pandas as pd
import streamlit as st

from app_utils import SCORES_PATH, ensure_dashboard_outputs

st.title("Churn and Lifetime Value")
ensure_dashboard_outputs()

df = pd.read_csv(SCORES_PATH)
st.subheader("Churn Probability by Plan")
st.bar_chart(df.groupby("plan_type")["churn_probability"].mean())
st.subheader("Predicted LTV by Risk Tier")
st.bar_chart(df.groupby("risk_tier")["predicted_ltv"].mean())
st.subheader("High-Value Customers at Risk")
st.dataframe(
    df[(df["is_high_value"] == 1) & (df["risk_tier"] == "high")].sort_values("predicted_ltv", ascending=False).head(50),
    use_container_width=True,
)
