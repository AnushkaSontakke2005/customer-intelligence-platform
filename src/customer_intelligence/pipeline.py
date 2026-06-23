"""End-to-end training pipeline for the Customer Intelligence Platform."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from customer_intelligence.business.kpis import calculate_kpis
from customer_intelligence.business.personas import generate_personas
from customer_intelligence.business.recommendations import build_recommendations
from customer_intelligence.business.revenue_simulator import simulate_retention_impact
from customer_intelligence.config import MODEL_DIR, PROCESSED_DATA_DIR
from customer_intelligence.data.ingestion import ingest_customers
from customer_intelligence.data.validation import validate_customer_data
from customer_intelligence.features.build_features import build_customer_features
from customer_intelligence.models.churn import score_churn, train_churn_models
from customer_intelligence.models.ltv import score_ltv, train_ltv_model
from customer_intelligence.models.segmentation import fit_segmentation_models


def run_training_pipeline(input_path: str | None = None) -> dict[str, object]:
    """Run ingestion, validation, features, segmentation, churn, LTV, and business scoring."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = ingest_customers(input_path)
    validation = validate_customer_data(raw)
    if not validation.passed:
        raise ValueError(f"Data validation failed: {validation.errors}")

    features = build_customer_features(raw)
    segmented, segmentation_models, segmentation_metrics = fit_segmentation_models(features)
    churn_models, churn_metrics, best_churn_name = train_churn_models(segmented)
    best_churn_metrics = churn_metrics[churn_metrics["model"] == best_churn_name].iloc[0].to_dict()
    best_decision_threshold = float(best_churn_metrics["optimal_threshold"])
    scored = score_churn(segmented, churn_models[best_churn_name], decision_threshold=best_decision_threshold)
    ltv_model, ltv_metrics = train_ltv_model(scored)
    scored = score_ltv(scored, ltv_model)
    recommendations = build_recommendations(scored)
    personas = generate_personas(recommendations)

    recommendations.to_csv(PROCESSED_DATA_DIR / "customer_scores.csv", index=False)
    churn_metrics.to_csv(PROCESSED_DATA_DIR / "churn_model_metrics.csv", index=False)
    personas.to_csv(PROCESSED_DATA_DIR / "personas.csv", index=False)
    joblib.dump(churn_models[best_churn_name], MODEL_DIR / "best_churn_model.joblib")
    joblib.dump(ltv_model, MODEL_DIR / "ltv_model.joblib")
    joblib.dump(segmentation_models["kmeans"], MODEL_DIR / "kmeans_segmentation.joblib")
    Path(MODEL_DIR / "best_churn_threshold.json").write_text(
        json.dumps(
            {
                "model": best_churn_name,
                "optimal_threshold": best_decision_threshold,
                "fp_cost": best_churn_metrics["fp_cost"],
                "fn_cost": best_churn_metrics["fn_cost"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "validation_warnings": validation.warnings,
        "segmentation_metrics": segmentation_metrics,
        "best_churn_model": best_churn_name,
        "best_churn_threshold": best_decision_threshold,
        "churn_metrics": churn_metrics.to_dict(orient="records"),
        "best_churn_business_metrics": best_churn_metrics,
        "ltv_metrics": ltv_metrics,
        "kpis": calculate_kpis(recommendations),
        "revenue_simulation": simulate_retention_impact(recommendations),
    }
    Path(PROCESSED_DATA_DIR / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_training_pipeline(), indent=2))
