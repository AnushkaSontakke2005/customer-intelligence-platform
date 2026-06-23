"""MLflow experiment tracking wrapper."""

from __future__ import annotations

import mlflow

from customer_intelligence.pipeline import run_training_pipeline


def run_tracked_training(experiment_name: str = "customer-intelligence-platform") -> dict[str, object]:
    """Execute the training pipeline and log core metrics to MLflow."""

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="end_to_end_training"):
        summary = run_training_pipeline()
        mlflow.set_tag("ltv_version", "v2_decoupled")
        mlflow.set_tag(
            "mlflow.note.content",
            "LTV v2 decouples the expected-margin target from raw revenue features. "
            "The target uses monthly margin divided by churn probability, while the model "
            "uses only tenure, contract, service adoption, support, household, payment, "
            "and internet-service features.",
        )
        mlflow.log_param("best_churn_model", summary["best_churn_model"])
        mlflow.log_metric("ltv_mae", summary["ltv_metrics"]["mae"])
        mlflow.log_metric("ltv_rmse", summary["ltv_metrics"]["rmse"])
        mlflow.log_metric("ltv_r2", summary["ltv_metrics"]["r2"])
        mlflow.log_metric("retention_roi", summary["revenue_simulation"]["roi"])
        mlflow.log_metric("net_impact", summary["revenue_simulation"]["net_impact"])
        for row in summary["churn_metrics"]:
            if row["model"] == summary["best_churn_model"]:
                mlflow.log_metric("best_churn_roc_auc", row["roc_auc"])
                mlflow.log_metric("best_churn_average_precision", row["average_precision"])
        mlflow.log_artifacts("data/processed", artifact_path="processed_outputs")
    return summary


if __name__ == "__main__":
    print(run_tracked_training())
