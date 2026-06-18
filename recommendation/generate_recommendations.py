"""CLI entrypoint for retention recommendations."""

import pandas as pd

from customer_intelligence.business.recommendations import build_recommendations


if __name__ == "__main__":
    scores = pd.read_csv("data/processed/customer_scores.csv")
    build_recommendations(scores).to_csv("data/processed/retention_recommendations.csv", index=False)
    print("Wrote data/processed/retention_recommendations.csv")
