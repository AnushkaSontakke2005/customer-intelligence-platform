from customer_intelligence.data.ingestion import generate_synthetic_customer_data
from customer_intelligence.features.build_features import MODEL_FEATURES, build_customer_features


def test_feature_pipeline_adds_model_features():
    df = generate_synthetic_customer_data(n_customers=100, seed=11)
    features = build_customer_features(df)
    for column in MODEL_FEATURES:
        assert column in features.columns
    assert features["rfm_score"].between(111, 555).all()
