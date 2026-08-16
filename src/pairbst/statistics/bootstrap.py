"""Efficient patient-cluster bootstrap and paired randomization procedures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


PRIMARY_METRICS = ("balanced_accuracy", "macro_f1")
ALL_METRICS = ("balanced_accuracy", "macro_f1", "accuracy", "weighted_f1")


@dataclass
class ClusterBootstrapResult:
    """Point estimates, percentile intervals, and bootstrap distributions."""

    point_estimates: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    distributions: dict[str, np.ndarray]
    n_bootstrap: int
    confidence_level: float
    seed: int
    n_patients: int
    stratified: bool

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "metric": metric,
                    "estimate": self.point_estimates[metric],
                    "ci_low": self.confidence_intervals[metric][0],
                    "ci_high": self.confidence_intervals[metric][1],
                    "confidence_level": self.confidence_level,
                    "n_bootstrap": self.n_bootstrap,
                    "n_patients": self.n_patients,
                    "stratified": self.stratified,
                }
                for metric in self.point_estimates
            ]
        )


@dataclass
class PairedBootstrapResult:
    """Paired A-minus-B effects calculated from identical cluster draws.

    Confidence intervals come from the paired patient-cluster bootstrap.  The
    P values are calculated separately under the null with patient-level
    random swaps of systems A and B; a bootstrap distribution is not a null
    distribution and therefore must not be used as one.
    """

    point_differences: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    p_values: dict[str, float]
    probability_a_better: dict[str, float]
    distributions: dict[str, np.ndarray]
    n_bootstrap: int
    confidence_level: float
    seed: int
    n_patients: int
    p_value_method: str = "patient_cluster_random_swap"

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "metric": metric,
                    "difference_a_minus_b": self.point_differences[metric],
                    "ci_low": self.confidence_intervals[metric][0],
                    "ci_high": self.confidence_intervals[metric][1],
                    "p_value": self.p_values[metric],
                    "probability_a_better": self.probability_a_better[metric],
                    "confidence_level": self.confidence_level,
                    "n_bootstrap": self.n_bootstrap,
                    "n_patients": self.n_patients,
                    "p_value_method": self.p_value_method,
                }
                for metric in self.point_differences
            ]
        )


def _validate_bootstrap_parameters(n_bootstrap: int, confidence_level: float) -> None:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")


def _encode_predictions(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    labels: Sequence[Any] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.shape != true.shape or true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty, equally shaped 1D arrays.")
    if labels is None:
        encoder = LabelEncoder().fit(np.concatenate([true, pred]))
        classes = encoder.classes_
        return encoder.transform(true), encoder.transform(pred), classes

    classes = np.asarray(labels)
    if classes.ndim != 1 or classes.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence.")
    class_keys = [value.item() if isinstance(value, np.generic) else value for value in classes]
    if len(set(class_keys)) != len(class_keys):
        raise ValueError("labels must not contain duplicates.")
    mapping = {value: index for index, value in enumerate(class_keys)}

    def encode(values: np.ndarray) -> np.ndarray:
        output = np.empty(values.size, dtype=np.int64)
        unknown: set[Any] = set()
        for index, value in enumerate(values):
            key = value.item() if isinstance(value, np.generic) else value
            if key not in mapping:
                unknown.add(key)
            else:
                output[index] = mapping[key]
        if unknown:
            raise ValueError(
                f"Predictions contain values outside labels: {sorted(unknown, key=str)}"
            )
        return output

    return encode(true), encode(pred), classes


def _cluster_layout(
    patient_ids: Sequence[Any] | np.ndarray,
    n_samples: int,
    *,
    strata: Sequence[Any] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    patients = np.asarray(patient_ids)
    if patients.shape != (n_samples,):
        raise ValueError(f"patient_ids must have shape ({n_samples},).")
    patient_classes, patient_inverse = np.unique(patients.astype(str), return_inverse=True)
    n_patients = patient_classes.size
    if n_patients < 2:
        raise ValueError("Patient-cluster bootstrap requires at least two patients.")

    if strata is None:
        groups = [np.arange(n_patients, dtype=np.int64)]
    else:
        stratum_array = np.asarray(strata)
        if stratum_array.shape != (n_samples,):
            raise ValueError(f"strata must have shape ({n_samples},).")
        patient_strata = np.empty(n_patients, dtype=object)
        for patient_index in range(n_patients):
            values = np.unique(stratum_array[patient_inverse == patient_index])
            if values.size != 1:
                raise ValueError(
                    f"Patient {patient_classes[patient_index]!r} spans multiple bootstrap strata."
                )
            patient_strata[patient_index] = values[0]
        groups = [
            np.flatnonzero(patient_strata == stratum)
            for stratum in sorted(np.unique(patient_strata).tolist(), key=str)
        ]
    return patient_classes, patient_inverse, groups


def _cluster_confusions(
    true: np.ndarray,
    pred: np.ndarray,
    patient_inverse: np.ndarray,
    n_patients: int,
    n_classes: int,
) -> np.ndarray:
    flat_code = (
        patient_inverse * (n_classes * n_classes) + true * n_classes + pred
    )
    counts = np.bincount(
        flat_code,
        minlength=n_patients * n_classes * n_classes,
    )
    return counts.reshape(n_patients, n_classes, n_classes).astype(np.float64)


def _metrics_from_confusions(confusions: np.ndarray) -> dict[str, np.ndarray]:
    matrices = np.asarray(confusions, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = matrices[None, :, :]
    diagonal = np.diagonal(matrices, axis1=1, axis2=2)
    actual = matrices.sum(axis=2)
    predicted = matrices.sum(axis=1)
    total = matrices.sum(axis=(1, 2))
    recalls = np.divide(diagonal, actual, out=np.zeros_like(diagonal), where=actual != 0)
    precisions = np.divide(
        diagonal, predicted, out=np.zeros_like(diagonal), where=predicted != 0
    )
    f1 = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(recalls),
        where=(precisions + recalls) != 0,
    )
    return {
        "balanced_accuracy": recalls.mean(axis=1),
        "macro_f1": f1.mean(axis=1),
        "accuracy": np.divide(
            diagonal.sum(axis=1), total, out=np.zeros_like(total), where=total != 0
        ),
        "weighted_f1": np.divide(
            (f1 * actual).sum(axis=1), total, out=np.zeros_like(total), where=total != 0
        ),
    }


def _bootstrap_patient_weights(
    rng: np.random.Generator,
    groups: list[np.ndarray],
    n_patients: int,
    n_bootstrap: int,
) -> np.ndarray:
    weights = np.zeros((n_bootstrap, n_patients), dtype=np.int32)
    for patient_group in groups:
        size = patient_group.size
        probabilities = np.full(size, 1.0 / size)
        draws = rng.multinomial(size, probabilities, size=n_bootstrap)
        weights[:, patient_group] = draws
    return weights


def _percentile_interval(
    distribution: np.ndarray, confidence_level: float
) -> tuple[float, float]:
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(distribution, [tail, 1.0 - tail])
    return float(low), float(high)


def cluster_bootstrap_classification(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    patient_ids: Sequence[Any] | np.ndarray,
    *,
    labels: Sequence[Any] | np.ndarray | None = None,
    strata: Sequence[Any] | np.ndarray | None = None,
    stratified: bool = False,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260814,
    chunk_size: int = 1_000,
) -> ClusterBootstrapResult:
    """Bootstrap classification outcomes by resampling whole patients.

    The safe default is an unstratified cluster bootstrap.  Pass explicit
    patient-stable ``strata`` to preserve a predeclared stratification
    variable.  For backward compatibility, ``stratified=True`` without
    explicit strata uses the true outcome, but that is valid only when every
    patient has one unique task label.
    """

    _validate_bootstrap_parameters(n_bootstrap, confidence_level)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    true, pred, classes = _encode_predictions(y_true, y_pred, labels)
    chosen_strata = (
        np.asarray(strata)
        if strata is not None
        else (np.asarray(y_true) if stratified else None)
    )
    patients, patient_inverse, groups = _cluster_layout(
        patient_ids, true.size, strata=chosen_strata
    )
    cluster_matrices = _cluster_confusions(
        true, pred, patient_inverse, patients.size, classes.size
    )
    point_arrays = _metrics_from_confusions(cluster_matrices.sum(axis=0))
    point = {metric: float(values[0]) for metric, values in point_arrays.items()}

    rng = np.random.default_rng(seed)
    distributions = {
        metric: np.empty(n_bootstrap, dtype=np.float64) for metric in ALL_METRICS
    }
    flattened = cluster_matrices.reshape(patients.size, -1)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        weights = _bootstrap_patient_weights(rng, groups, patients.size, stop - start)
        matrices = (weights @ flattened).reshape(stop - start, classes.size, classes.size)
        metrics = _metrics_from_confusions(matrices)
        for metric in ALL_METRICS:
            distributions[metric][start:stop] = metrics[metric]

    intervals = {
        metric: _percentile_interval(values, confidence_level)
        for metric, values in distributions.items()
    }
    return ClusterBootstrapResult(
        point_estimates=point,
        confidence_intervals=intervals,
        distributions=distributions,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
        n_patients=int(patients.size),
        stratified=chosen_strata is not None,
    )


def cluster_bootstrap_classification_by_seed(
    y_true: Sequence[Any] | np.ndarray,
    predictions_by_seed: Sequence[Sequence[Any]] | np.ndarray,
    patient_ids: Sequence[Any] | np.ndarray,
    seeds: Sequence[int] | np.ndarray,
    *,
    labels: Sequence[Any] | np.ndarray | None = None,
    strata: Sequence[Any] | np.ndarray | None = None,
    stratified: bool = False,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260814,
    chunk_size: int = 1_000,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run patient-cluster bootstrap independently for every model seed.

    The first axis of ``predictions_by_seed`` is the independent linear-probe
    seed.  Results are deliberately not collapsed across seeds.  This helper
    therefore cannot create an artificial cross-seed confidence interval and
    is suitable only for the seed-specific supplementary uncertainty output.
    """

    true = np.asarray(y_true)
    predictions = np.asarray(predictions_by_seed)
    seed_values = np.asarray(seeds, dtype=np.int64)
    if predictions.ndim != 2 or predictions.shape[1] != true.size:
        raise ValueError(
            "predictions_by_seed must have shape (n_seeds, n_samples) matching y_true."
        )
    if seed_values.ndim != 1 or seed_values.size != predictions.shape[0]:
        raise ValueError("seeds must contain exactly one identifier per prediction row.")
    if np.unique(seed_values).size != seed_values.size:
        raise ValueError("seeds must be unique.")

    summaries: list[pd.DataFrame] = []
    distributions: dict[str, np.ndarray] = {}
    for seed_index, seed_value in enumerate(seed_values.tolist()):
        derived_seed = int(
            np.random.SeedSequence([int(bootstrap_seed), int(seed_value)])
            .generate_state(1, dtype=np.uint32)[0]
        )
        result = cluster_bootstrap_classification(
            true,
            predictions[seed_index],
            patient_ids,
            labels=labels,
            strata=strata,
            stratified=stratified,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=derived_seed,
            chunk_size=chunk_size,
        )
        summaries.append(
            result.summary_frame().assign(
                seed=int(seed_value),
                bootstrap_seed=derived_seed,
            )
        )
        for metric, values in result.distributions.items():
            distributions[f"seed_{int(seed_value)}::{metric}"] = values
    return pd.concat(summaries, ignore_index=True), distributions


def cluster_bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    patient_ids: Sequence[Any] | np.ndarray,
    *,
    strata: Sequence[Any] | np.ndarray | None = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260814,
    chunk_size: int = 1_000,
) -> dict[str, Any]:
    """Patient-cluster percentile interval for a query-level mean metric."""

    _validate_bootstrap_parameters(n_bootstrap, confidence_level)
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or observations.size == 0 or not np.isfinite(observations).all():
        raise ValueError("values must be a finite, non-empty one-dimensional array.")
    patients, patient_inverse, groups = _cluster_layout(
        patient_ids, observations.size, strata=strata
    )
    cluster_sums = np.bincount(
        patient_inverse, weights=observations, minlength=patients.size
    )
    cluster_counts = np.bincount(patient_inverse, minlength=patients.size)
    rng = np.random.default_rng(seed)
    distribution = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        weights = _bootstrap_patient_weights(rng, groups, patients.size, stop - start)
        numerator = weights @ cluster_sums
        denominator = weights @ cluster_counts
        distribution[start:stop] = numerator / denominator
    return {
        "estimate": float(observations.mean()),
        "ci_low": _percentile_interval(distribution, confidence_level)[0],
        "ci_high": _percentile_interval(distribution, confidence_level)[1],
        "distribution": distribution,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "seed": seed,
        "n_patients": int(patients.size),
        "stratified": strata is not None,
    }


def paired_cluster_bootstrap_mean(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    patient_ids: Sequence[Any] | np.ndarray,
    *,
    strata: Sequence[Any] | np.ndarray | None = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260814,
    chunk_size: int = 1_000,
) -> dict[str, Any]:
    """Paired patient-cluster interval and random-swap P value for a mean effect.

    This is the paired procedure for query-level retrieval measures such as AP,
    Hit, or majority-vote correctness.  Arrays must already be aligned to the
    same query IDs; both models are evaluated on identical patient draws.
    """

    _validate_bootstrap_parameters(n_bootstrap, confidence_level)
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if (
        a.ndim != 1
        or b.shape != a.shape
        or a.size == 0
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("values_a and values_b must be finite, aligned, non-empty 1D arrays.")
    patients, patient_inverse, groups = _cluster_layout(patient_ids, a.size, strata=strata)
    cluster_difference_sums = np.bincount(
        patient_inverse, weights=a - b, minlength=patients.size
    )
    cluster_counts = np.bincount(patient_inverse, minlength=patients.size)
    rng = np.random.default_rng(seed)
    distribution = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        weights = _bootstrap_patient_weights(rng, groups, patients.size, stop - start)
        distribution[start:stop] = (
            (weights @ cluster_difference_sums) / (weights @ cluster_counts)
        )
    # Null randomization: swap A and B jointly for every observation belonging
    # to a patient.  This retains within-patient dependence while imposing the
    # exchangeability null.  It is intentionally distinct from the bootstrap
    # distribution used for the confidence interval.
    randomization_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    observed_difference = float((a - b).mean())
    extreme = 0
    null_scale = float(cluster_counts.sum())
    threshold = max(0.0, abs(observed_difference) - np.finfo(np.float64).eps * 8)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        signs = randomization_rng.integers(
            0, 2, size=(stop - start, patients.size), dtype=np.int8
        ).astype(np.float64)
        signs *= 2.0
        signs -= 1.0
        null_differences = (signs @ cluster_difference_sums) / null_scale
        extreme += int(np.count_nonzero(np.abs(null_differences) >= threshold))
    ci_low, ci_high = _percentile_interval(distribution, confidence_level)
    return {
        "difference_a_minus_b": observed_difference,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float((extreme + 1) / (n_bootstrap + 1)),
        "p_value_method": "patient_cluster_random_swap",
        "n_randomization": n_bootstrap,
        "probability_a_better": float(np.mean(distribution > 0)),
        "distribution": distribution,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "seed": seed,
        "n_patients": int(patients.size),
        "stratified": strata is not None,
    }


def paired_cluster_bootstrap_classification(
    y_true: Sequence[Any] | np.ndarray,
    y_pred_a: Sequence[Any] | np.ndarray,
    y_pred_b: Sequence[Any] | np.ndarray,
    patient_ids: Sequence[Any] | np.ndarray,
    *,
    labels: Sequence[Any] | np.ndarray | None = None,
    strata: Sequence[Any] | np.ndarray | None = None,
    stratified: bool = False,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260814,
    chunk_size: int = 1_000,
) -> PairedBootstrapResult:
    """Paired classification interval plus patient-level random-swap P value."""

    _validate_bootstrap_parameters(n_bootstrap, confidence_level)
    true_a, pred_a, classes = _encode_predictions(y_true, y_pred_a, labels)
    true_b, pred_b, classes_b = _encode_predictions(y_true, y_pred_b, classes)
    if not np.array_equal(true_a, true_b) or not np.array_equal(classes, classes_b):
        raise RuntimeError("Internal label encoding mismatch.")
    chosen_strata = (
        np.asarray(strata)
        if strata is not None
        else (np.asarray(y_true) if stratified else None)
    )
    patients, patient_inverse, groups = _cluster_layout(
        patient_ids, true_a.size, strata=chosen_strata
    )
    matrices_a = _cluster_confusions(
        true_a, pred_a, patient_inverse, patients.size, classes.size
    )
    matrices_b = _cluster_confusions(
        true_b, pred_b, patient_inverse, patients.size, classes.size
    )
    point_a = _metrics_from_confusions(matrices_a.sum(axis=0))
    point_b = _metrics_from_confusions(matrices_b.sum(axis=0))
    point_differences = {
        metric: float(point_a[metric][0] - point_b[metric][0])
        for metric in ALL_METRICS
    }

    rng = np.random.default_rng(seed)
    distributions = {
        metric: np.empty(n_bootstrap, dtype=np.float64) for metric in ALL_METRICS
    }
    flat_a = matrices_a.reshape(patients.size, -1)
    flat_b = matrices_b.reshape(patients.size, -1)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        weights = _bootstrap_patient_weights(rng, groups, patients.size, stop - start)
        metrics_a = _metrics_from_confusions(
            (weights @ flat_a).reshape(stop - start, classes.size, classes.size)
        )
        metrics_b = _metrics_from_confusions(
            (weights @ flat_b).reshape(stop - start, classes.size, classes.size)
        )
        for metric in ALL_METRICS:
            distributions[metric][start:stop] = metrics_a[metric] - metrics_b[metric]

    intervals = {
        metric: _percentile_interval(values, confidence_level)
        for metric, values in distributions.items()
    }
    # Generate a separate null distribution by swapping the complete A/B
    # prediction contribution for each patient.  Using the ordinary bootstrap
    # tails as a hypothesis test would be statistically invalid because that
    # distribution is centered on the observed effect, not on the null.
    randomization_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    extreme_counts = {metric: 0 for metric in ALL_METRICS}
    flat_delta = flat_a - flat_b
    flat_total = flat_a.sum(axis=0) + flat_b.sum(axis=0)
    thresholds = {
        metric: max(0.0, abs(value) - np.finfo(np.float64).eps * 8)
        for metric, value in point_differences.items()
    }
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        swaps = randomization_rng.integers(
            0, 2, size=(stop - start, patients.size), dtype=np.int8
        ).astype(np.float64)
        null_a_flat = swaps @ flat_delta + flat_b.sum(axis=0)
        null_b_flat = flat_total[None, :] - null_a_flat
        null_a = _metrics_from_confusions(
            null_a_flat.reshape(stop - start, classes.size, classes.size)
        )
        null_b = _metrics_from_confusions(
            null_b_flat.reshape(stop - start, classes.size, classes.size)
        )
        for metric in ALL_METRICS:
            null_difference = null_a[metric] - null_b[metric]
            extreme_counts[metric] += int(
                np.count_nonzero(np.abs(null_difference) >= thresholds[metric])
            )

    p_values: dict[str, float] = {}
    probability_a_better: dict[str, float] = {}
    for metric, values in distributions.items():
        probability_a_better[metric] = float(np.mean(values > 0))
        p_values[metric] = float((extreme_counts[metric] + 1) / (n_bootstrap + 1))

    return PairedBootstrapResult(
        point_differences=point_differences,
        confidence_intervals=intervals,
        p_values=p_values,
        probability_a_better=probability_a_better,
        distributions=distributions,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
        n_patients=int(patients.size),
        p_value_method="patient_cluster_random_swap",
    )
