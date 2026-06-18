"""Segmentation dashboard page."""

import pandas as pd
import streamlit as st

from dashboard.app_utils import PERSONAS_PATH, SCORES_PATH, ensure_dashboard_outputs

st.title("Segmentation and Personas")
ensure_dashboard_outputs()

df = pd.read_csv(SCORES_PATH)
personas = pd.read_csv(PERSONAS_PATH)
st.subheader("Persona Playbooks")
st.dataframe(personas, use_container_width=True)

st.subheader("Behavioral Segment Profile")
profile = df.groupby("kmeans_segment").agg(
    customers=("customer_id", "count"),
    revenue=("total_revenue_12m", "sum"),
    avg_ltv=("predicted_ltv", "mean"),
    churn_probability=("churn_probability", "mean"),
    avg_sessions=("sessions_30d", "mean"),
)
st.dataframe(profile.round(2), use_container_width=True)
