"""Monitoring utilities for drift and model performance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def population_stability_index(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calculate PSI to flag feature drift between training and production."""

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(expected.quantile(quantiles).to_numpy())
    if len(breakpoints) < 3:
        return 0.0
    expected_counts = pd.cut(expected, breakpoints, include_lowest=True).value_counts(normalize=True)
    actual_counts = pd.cut(actual, breakpoints, include_lowest=True).value_counts(normalize=True)
    expected_pct = expected_counts.sort_index().replace(0, 0.0001)
    actual_pct = actual_counts.sort_index().replace(0, 0.0001)
    return float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())


def evaluate_production_predictions(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    """Compute production model quality metrics once labels mature."""

    return {"roc_auc": float(roc_auc_score(y_true, y_score)), "observed_churn_rate": float(y_true.mean())}
