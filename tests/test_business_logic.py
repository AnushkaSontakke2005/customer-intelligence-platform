from customer_intelligence.business.recommendations import build_recommendations
from customer_intelligence.business.revenue_simulator import simulate_retention_impact
from customer_intelligence.data.ingestion import generate_synthetic_customer_data
from customer_intelligence.features.build_features import build_customer_features


def test_recommendations_and_simulator_return_business_outputs():
    df = build_customer_features(generate_synthetic_customer_data(n_customers=100, seed=5))
    df["churn_probability"] = 0.7
    recs = build_recommendations(df)
    result = simulate_retention_impact(recs)
    assert "retention_action" in recs.columns
    assert result["targeted_customers"] == 20
    assert "roi" in result
