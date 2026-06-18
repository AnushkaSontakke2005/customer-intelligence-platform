"""Central configuration for the Customer Intelligence Platform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "mlflow" / "models"


@dataclass(frozen=True)
class BusinessConfig:
    """Business assumptions used by simulators and recommendation logic."""

    average_gross_margin: float = 0.62
    retention_offer_cost: float = 12.0
    proactive_success_rate: float = 0.18
    high_risk_threshold: float = 0.65
    medium_risk_threshold: float = 0.35


@dataclass(frozen=True)
class ModelConfig:
    """Modeling defaults that keep experiments reproducible."""

    random_state: int = 42
    test_size: float = 0.2
    n_clusters: int = 4
    dbscan_eps: float = 0.9
    dbscan_min_samples: int = 10


BUSINESS_CONFIG = BusinessConfig()
MODEL_CONFIG = ModelConfig()
