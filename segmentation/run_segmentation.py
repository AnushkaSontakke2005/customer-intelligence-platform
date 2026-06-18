"""CLI entrypoint for customer segmentation."""

from customer_intelligence.data.ingestion import ingest_customers
from customer_intelligence.features.build_features import build_customer_features
from customer_intelligence.models.segmentation import fit_segmentation_models


if __name__ == "__main__":
    df = build_customer_features(ingest_customers())
    segmented, _, metrics = fit_segmentation_models(df)
    segmented.to_csv("data/processed/segmented_customers.csv", index=False)
    print(metrics)
