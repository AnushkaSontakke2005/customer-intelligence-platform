"""Explainable AI helpers for SHAP and LIME."""

from __future__ import annotations

import pandas as pd


def generate_global_explanations(model, X: pd.DataFrame, max_rows: int = 500) -> dict[str, object]:
    """Return SHAP-style explanations when installed, otherwise model feature hints."""

    sample = X.head(max_rows)
    try:
        import shap

        transformed = model.named_steps["preprocessor"].transform(sample)
        estimator = model.named_steps["model"]
        explainer = shap.Explainer(estimator, transformed)
        values = explainer(transformed)
        return {"method": "shap", "values_shape": values.values.shape}
    except Exception as exc:
        return {
            "method": "fallback_feature_importance",
            "reason": str(exc),
            "note": "Install shap for full additive explanations.",
        }


def generate_local_lime_explanation(model, X: pd.DataFrame, row_index: int = 0) -> dict[str, object]:
    """Return a LIME explanation summary when lime is available."""

    try:
        import lime.lime_tabular

        transformed = model.named_steps["preprocessor"].transform(X)
        explainer = lime.lime_tabular.LimeTabularExplainer(
            transformed,
            mode="classification",
            feature_names=[f"feature_{i}" for i in range(transformed.shape[1])],
        )
        explanation = explainer.explain_instance(transformed[row_index], model.named_steps["model"].predict_proba)
        return {"method": "lime", "explanation": explanation.as_list()}
    except Exception as exc:
        return {"method": "fallback", "reason": str(exc), "note": "Install lime for local perturbation explanations."}
