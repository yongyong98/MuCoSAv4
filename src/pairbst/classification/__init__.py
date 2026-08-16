"""Deterministic linear-probe evaluation for the PAIR-BST benchmark."""

from .linear_probe import (
    LinearProbeConfig,
    LinearProbeCVResult,
    run_outer_cv_linear_probe,
)
from .metrics import classification_metrics

__all__ = [
    "LinearProbeConfig",
    "LinearProbeCVResult",
    "classification_metrics",
    "run_outer_cv_linear_probe",
]
