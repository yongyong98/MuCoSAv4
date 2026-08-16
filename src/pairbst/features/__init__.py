"""Streaming feature extraction and compatibility adapters."""

from .extract import (
    ExtractionConfig,
    ModelExtractionRun,
    extract_model_features,
    extract_models_sequentially,
    extract_roi_features,
)
from .h5store import (
    FEATURE_DATASETS,
    ResumableFeatureStore,
    partial_path_for,
    validate_feature_file,
)
from .import_legacy import RecoveryFeatureAdapter, RecoverySourceContract
from .pooling import OnlineMeanMax, pool_grid_features

__all__ = [
    "FEATURE_DATASETS",
    "ExtractionConfig",
    "ModelExtractionRun",
    "OnlineMeanMax",
    "RecoveryFeatureAdapter",
    "RecoverySourceContract",
    "ResumableFeatureStore",
    "extract_model_features",
    "extract_models_sequentially",
    "extract_roi_features",
    "partial_path_for",
    "pool_grid_features",
    "validate_feature_file",
]
