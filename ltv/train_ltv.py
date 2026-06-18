"""CLI entrypoint for LTV model training."""

from customer_intelligence.data.ingestion import ingest_customers
from customer_intelligence.features.build_features import build_customer_features
from customer_intelligence.models.ltv import train_ltv_model


if __name__ == "__main__":
    df = build_customer_features(ingest_customers())
    _, metrics = train_ltv_model(df)
    print(metrics)
