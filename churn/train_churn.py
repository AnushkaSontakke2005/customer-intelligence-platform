"""CLI entrypoint for churn model training."""

from customer_intelligence.data.ingestion import ingest_customers
from customer_intelligence.features.build_features import build_customer_features
from customer_intelligence.models.churn import train_churn_models


if __name__ == "__main__":
    df = build_customer_features(ingest_customers())
    _, metrics, best_model = train_churn_models(df)
    print(metrics)
    print(f"Best model: {best_model}")
