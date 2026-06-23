import numpy as np

from customer_intelligence.data.ingestion import generate_synthetic_customer_data
from customer_intelligence.features.build_features import build_customer_features
from customer_intelligence.models.churn import calculate_business_threshold
from customer_intelligence.models.ltv import LTV_CATEGORICAL_FEATURES, LTV_NUMERIC_FEATURES, build_ltv_target


def test_ltv_features_exclude_direct_revenue_leakage():
    forbidden_features = {
        "monthly_spend",
        "total_revenue_12m",
        "monetary",
        "rfm_score",
        "revenue_per_purchase",
        "avg_revenue_per_tenure_month",
        "avg_session_minutes",
    }
    assert forbidden_features.isdisjoint(LTV_NUMERIC_FEATURES)
    assert forbidden_features.isdisjoint(LTV_CATEGORICAL_FEATURES)


def test_ltv_target_uses_churn_adjusted_monthly_margin():
    df = build_customer_features(generate_synthetic_customer_data(n_customers=10, seed=13))
    df["churn_probability"] = 0.25
    target = build_ltv_target(df)
    expected = df["monthly_spend"] * 0.62 / (0.25 + 1e-6)
    assert np.allclose(target, expected)


def test_business_threshold_can_improve_default_cost():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.45, 0.40, 0.35, 0.34, 0.20, 0.10])
    result = calculate_business_threshold(y_true, y_prob, fp_cost=12, fn_cost=150)
    assert result["optimal_threshold"] < 0.5
    assert result["business_cost_at_optimal_threshold"] < result["business_cost_at_default_threshold"]
