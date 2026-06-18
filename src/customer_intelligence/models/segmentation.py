"""Customer segmentation with KMeans, DBSCAN, and hierarchical clustering."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from customer_intelligence.config import MODEL_CONFIG


SEGMENTATION_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "sessions_30d",
    "support_tickets_90d",
    "nps",
    "engagement_intensity",
]


def _segment_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), SEGMENTATION_FEATURES)],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("clusterer", estimator)])


def fit_segmentation_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, float]]:
    """Fit three clustering approaches and append segment labels."""

    scored = df.copy()
    models = {
        "kmeans": _segment_pipeline(
            KMeans(n_clusters=MODEL_CONFIG.n_clusters, random_state=MODEL_CONFIG.random_state, n_init=10)
        ),
        "dbscan": _segment_pipeline(
            DBSCAN(eps=MODEL_CONFIG.dbscan_eps, min_samples=MODEL_CONFIG.dbscan_min_samples)
        ),
        "hierarchical": _segment_pipeline(AgglomerativeClustering(n_clusters=MODEL_CONFIG.n_clusters)),
    }
    metrics: dict[str, float] = {}
    for name, model in models.items():
        labels = model.fit_predict(scored)
        scored[f"{name}_segment"] = labels
        if len(set(labels)) > 1 and len(set(labels)) < len(scored):
            features = model.named_steps["preprocessor"].transform(scored)
            metrics[f"{name}_silhouette"] = float(silhouette_score(features, labels))
        else:
            metrics[f"{name}_silhouette"] = 0.0
    return scored, models, metrics
