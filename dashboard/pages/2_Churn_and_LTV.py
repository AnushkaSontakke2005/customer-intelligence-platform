"""Churn and LTV dashboard page."""

import pandas as pd
import streamlit as st

from app_utils import SCORES_PATH, ensure_dashboard_outputs


DEFAULT_LTV_QUANTILE = 0.80
DEFAULT_RISK_TIER = "high"


def _first_non_empty_threshold(df: pd.DataFrame):
    """Find the closest relaxed LTV/risk threshold pair that returns customers."""

    for ltv_quantile in [0.80, 0.70, 0.60, 0.50]:
        ltv_cutoff = df["predicted_ltv"].quantile(ltv_quantile)
        for risk_threshold in [0.65, 0.50, 0.35, 0.25, 0.20, 0.15, 0.132, 0.10]:
            count = int(((df["predicted_ltv"] >= ltv_cutoff) & (df["churn_probability"] >= risk_threshold)).sum())
            if count > 0:
                return {
                    "ltv_quantile": ltv_quantile,
                    "ltv_cutoff": float(ltv_cutoff),
                    "risk_threshold": risk_threshold,
                    "matching_customers": count,
                }
    return None


st.title("Churn and Lifetime Value")
ensure_dashboard_outputs()

df = pd.read_csv(SCORES_PATH)
df["risk_tier_normalized"] = df["risk_tier"].astype(str).str.strip().str.lower()

st.subheader("Churn Probability by Plan")
st.bar_chart(df.groupby("plan_type")["churn_probability"].mean())

st.subheader("Predicted LTV by Risk Tier")
st.bar_chart(df.groupby("risk_tier")["predicted_ltv"].mean())

st.subheader("High-Value Customers at Risk")
ltv_cutoff = df["predicted_ltv"].quantile(DEFAULT_LTV_QUANTILE)
high_ltv_condition = df["predicted_ltv"] >= ltv_cutoff
high_risk_condition = df["risk_tier_normalized"] == DEFAULT_RISK_TIER
filtered_df = df[high_ltv_condition & high_risk_condition].sort_values("predicted_ltv", ascending=False).head(50)

debug_summary = {
    "shape_before_filtering": df.shape,
    "shape_after_filtering": filtered_df.shape,
    "risk_tier_distribution": df["risk_tier_normalized"].value_counts(dropna=False).to_dict(),
    "churn_probability_min": float(df["churn_probability"].min()),
    "churn_probability_mean": float(df["churn_probability"].mean()),
    "churn_probability_max": float(df["churn_probability"].max()),
    "predicted_ltv_min": float(df["predicted_ltv"].min()),
    "predicted_ltv_mean": float(df["predicted_ltv"].mean()),
    "predicted_ltv_max": float(df["predicted_ltv"].max()),
    "customers_meeting_ltv_filter": int(high_ltv_condition.sum()),
    "customers_meeting_risk_filter": int(high_risk_condition.sum()),
    "customers_meeting_both_filters": int((high_ltv_condition & high_risk_condition).sum()),
    "missing_churn_probability": int(df["churn_probability"].isna().sum()),
    "missing_predicted_ltv": int(df["predicted_ltv"].isna().sum()),
    "risk_tier_dtype": str(df["risk_tier"].dtype),
    "predicted_ltv_dtype": str(df["predicted_ltv"].dtype),
    "churn_probability_dtype": str(df["churn_probability"].dtype),
}
print(debug_summary)

with st.expander("Debug filter diagnostics"):
    st.write(debug_summary)
    st.write("Current thresholds")
    st.json(
        {
            "ltv_quantile": DEFAULT_LTV_QUANTILE,
            "ltv_cutoff": float(ltv_cutoff),
            "risk_tier_required": DEFAULT_RISK_TIER,
        }
    )

if filtered_df.empty:
    st.info("No high-value at-risk customers found under current thresholds.")
    st.write(
        f"Current filter: predicted LTV >= top {int((1 - DEFAULT_LTV_QUANTILE) * 100)}% cutoff "
        f"(${ltv_cutoff:,.2f}) and risk tier = '{DEFAULT_RISK_TIER}'."
    )

    suggestion = _first_non_empty_threshold(df)
    if suggestion:
        st.warning(
            "Suggested thresholds that produce results: "
            f"predicted LTV >= top {int((1 - suggestion['ltv_quantile']) * 100)}% cutoff "
            f"(${suggestion['ltv_cutoff']:,.2f}) and churn probability >= "
            f"{suggestion['risk_threshold']:.3f}. "
            f"This returns {suggestion['matching_customers']} customers."
        )
        suggested_df = df[
            (df["predicted_ltv"] >= suggestion["ltv_cutoff"])
            & (df["churn_probability"] >= suggestion["risk_threshold"])
        ].sort_values(["churn_probability", "predicted_ltv"], ascending=False)
        st.dataframe(suggested_df.head(50), use_container_width=True)
else:
    st.dataframe(filtered_df, use_container_width=True)
