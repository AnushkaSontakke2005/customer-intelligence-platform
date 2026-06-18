"""Shared Streamlit dashboard helpers."""

from pathlib import Path

import streamlit as st

from customer_intelligence.pipeline import run_training_pipeline


SCORES_PATH = Path("data/processed/customer_scores.csv")
PERSONAS_PATH = Path("data/processed/personas.csv")


@st.cache_resource(show_spinner="Training IBM Telco models and preparing dashboard data...")
def ensure_dashboard_outputs() -> None:
    """Create processed dashboard files when Streamlit starts from a clean deploy."""

    if not SCORES_PATH.exists() or not PERSONAS_PATH.exists():
        run_training_pipeline()
