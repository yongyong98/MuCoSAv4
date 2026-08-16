"""Predeclared paired comparisons and multiplicity correction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .bootstrap import (
    paired_cluster_bootstrap_classification,
    paired_cluster_bootstrap_mean,
)


def apply_holm_correction(
    comparisons: pd.DataFrame,
    *,
    p_column: str = "p_value",
    family_columns: Sequence[str] = ("family",),
    output_column: str = "p_value_holm",
) -> pd.DataFrame:
    """Apply Holm's step-down correction independently within each family."""

    if p_column not in comparisons:
        raise ValueError(f"Missing p-value column: {p_column}")
    frame = comparisons.copy()
    frame[output_column] = np.nan
    if family_columns:
        missing = [column for column in family_columns if column not in frame]
        if missing:
            raise ValueError(f"Missing family columns: {missing}")
        grouper: Any = list(family_columns)
        if len(family_columns) == 1:
            grouper = family_columns[0]
        groups = frame.groupby(grouper, dropna=False, sort=False).groups.values()
    else:
        groups = [frame.index]

    for indices in groups:
        index_array = np.asarray(list(indices))
        p_values = frame.loc[index_array, p_column].to_numpy(dtype=float)
        if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
            raise ValueError("All p-values must be finite and in [0, 1].")
        order = np.argsort(p_values, kind="mergesort")
        sorted_p = p_values[order]
        adjusted_sorted = np.maximum.accumulate(
            (len(sorted_p) - np.arange(len(sorted_p))) * sorted_p
        )
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = adjusted_sorted
        frame.loc[index_array, output_column] = adjusted
    return frame


def paired_model_comparison(
    y_true: Sequence[Any] | np.ndarray,
    y_pred_a: Sequence[Any] | np.ndarray,
    y_pred_b: Sequence[Any] | np.ndarray,
    patient_ids: Sequence[Any] | np.ndarray,
    *,
    model_a: str,
    model_b: str,
    task: str,
    family: str,
    labels: Sequence[Any] | np.ndarray | None = None,
    strata: Sequence[Any] | np.ndarray | None = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260814,
) -> pd.DataFrame:
    """Create audit-ready rows for a predeclared paired model comparison."""

    result = paired_cluster_bootstrap_classification(
        y_true,
        y_pred_a,
        y_pred_b,
        patient_ids,
        labels=labels,
        strata=strata,
        stratified=strata is not None,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    frame = result.summary_frame()
    frame.insert(0, "family", family)
    frame.insert(1, "task", task)
    frame.insert(2, "model_a", model_a)
    frame.insert(3, "model_b", model_b)
    frame["direction"] = "model_a_minus_model_b"
    frame["bootstrap_seed"] = seed
    return frame


def paired_query_metric_comparison(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    patient_ids: Sequence[Any] | np.ndarray,
    *,
    model_a: str,
    model_b: str,
    task: str,
    metric: str,
    family: str,
    strata: Sequence[Any] | np.ndarray | None = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260814,
) -> pd.DataFrame:
    """Create one paired cluster-bootstrap row for an aligned retrieval metric."""

    result = paired_cluster_bootstrap_mean(
        values_a,
        values_b,
        patient_ids,
        strata=strata,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    return pd.DataFrame(
        [
            {
                "family": family,
                "task": task,
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "difference_a_minus_b": result["difference_a_minus_b"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "p_value": result["p_value"],
                "p_value_method": result["p_value_method"],
                "n_randomization": result["n_randomization"],
                "probability_a_better": result["probability_a_better"],
                "direction": "model_a_minus_model_b",
                "confidence_level": confidence_level,
                "n_bootstrap": n_bootstrap,
                "n_patients": result["n_patients"],
                "bootstrap_seed": seed,
            }
        ]
    )
