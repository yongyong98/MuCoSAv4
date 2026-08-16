"""Frozen, deterministic independent-seed OOF linear-probe protocol.

The implementation intentionally has no validation split and no early
stopping. Every held-out fold is predicted by independently seeded linear
classifiers trained on the other two folds. Predictions from the three
held-out folds are concatenated separately for every seed. Primary metrics
are then calculated once per complete seed-specific OOF vector. Probabilities,
logits, and predictions are never combined across seeds in the canonical path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import classification_metrics, scalar_metrics
from ..provenance import configure_deterministic_cuda_environment


DEFAULT_SEEDS = (101, 202, 303, 404, 505)
CANONICAL_PROTOCOL_ID = "cv3_independent_seed_oof_v1"
PRIMARY_METRIC_UNIT = "complete_oof_per_seed"
SEED_AGGREGATION = "metric_mean_sd"


@dataclass(frozen=True)
class LinearProbeConfig:
    """Hyperparameters frozen for the PAIR-BST revision benchmark."""

    protocol_id: str = CANONICAL_PROTOCOL_ID
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    device: str = "cpu"
    num_workers: int = 0
    deterministic_algorithms: bool = True
    primary_metric_unit: str = PRIMARY_METRIC_UNIT
    seed_aggregation: str = SEED_AGGREGATION
    probability_ensemble_across_seeds: bool = False
    seed_sd_ddof: int = 1

    def validate(self) -> None:
        if self.protocol_id != CANONICAL_PROTOCOL_ID:
            raise ValueError(
                f"Canonical linear probing requires protocol_id={CANONICAL_PROTOCOL_ID!r}."
            )
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique.")
        if tuple(self.seeds) != DEFAULT_SEEDS:
            raise ValueError(
                f"Canonical seed order must be exactly {DEFAULT_SEEDS}."
            )
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.primary_metric_unit != PRIMARY_METRIC_UNIT:
            raise ValueError(
                f"primary_metric_unit must be {PRIMARY_METRIC_UNIT!r}."
            )
        if self.seed_aggregation != SEED_AGGREGATION:
            raise ValueError(f"seed_aggregation must be {SEED_AGGREGATION!r}.")
        if self.probability_ensemble_across_seeds:
            raise ValueError(
                "Probability ensembling across seeds is disabled in the canonical protocol."
            )
        if self.seed_sd_ddof != 1:
            raise ValueError("seed_sd_ddof must be 1 for sample standard deviation.")


@dataclass
class LinearProbeCVResult:
    """In-memory outputs from a complete three-fold linear-probe run."""

    classes: np.ndarray
    y_true_encoded: np.ndarray
    seeds: np.ndarray
    seed_oof_probabilities: np.ndarray
    seed_oof_predictions_encoded: np.ndarray
    seed_oof_predictions: np.ndarray
    seed_fold_metrics: pd.DataFrame
    seed_oof_metrics: pd.DataFrame
    seed_metric_summary: pd.DataFrame
    seed_pooled_metrics: list[dict[str, Any]]
    provenance: dict[str, Any]
    output_dir: Path | None = None


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _atomic_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _array_sha256(array: np.ndarray) -> str:
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(str(values.dtype).encode("ascii"))
    if values.dtype.kind in {"O", "U", "S"}:
        for value in values.reshape(-1):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        raw = memoryview(np.ascontiguousarray(values)).cast("B")
        chunk_size = 8 * 1024 * 1024
        for start in range(0, len(raw), chunk_size):
            digest.update(raw[start : start + chunk_size])
    return digest.hexdigest()


def _safe_component(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return cleaned or "fold"


def _reject_legacy_output_mixing(directory: Path) -> None:
    legacy_names = {
        "fold_metrics.csv",
        "fold_metric_mean_sd.csv",
        "oof_predictions.csv",
        "oof_probabilities.npz",
        "pooled_metrics.json",
        "pooled_per_class_metrics.csv",
        "pooled_confusion_matrix_counts.csv",
        "pooled_confusion_matrix_row_normalized.csv",
        "seed_and_mean_probabilities.npz",
        "seed_metrics.csv",
    }
    conflicts = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name in legacy_names
    )
    if conflicts:
        raise ValueError(
            "Canonical independent-seed outputs cannot share a directory with "
            f"legacy ensemble artifacts: {conflicts[:10]}."
        )


def _seed_everything(seed: int, deterministic: bool) -> None:
    if deterministic:
        configure_deterministic_cuda_environment()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def _inverse_frequency_weights(encoded_labels: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(encoded_labels, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training fold is missing encoded classes: {missing}.")
    weights = 1.0 / counts
    weights /= weights.mean()
    return weights.astype(np.float32)


def _train_one_seed(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    n_classes: int,
    class_weights: np.ndarray,
    seed: int,
    config: LinearProbeConfig,
) -> tuple[nn.Linear, np.ndarray, list[float]]:
    _seed_everything(seed, config.deterministic_algorithms)
    device = torch.device(config.device)
    model = nn.Linear(train_features.shape[1], n_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_labels.astype(np.int64, copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        drop_last=False,
    )

    epoch_losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        total_loss = 0.0
        total_examples = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=False)
            batch_labels = batch_labels.to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            batch_count = int(batch_labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            total_examples += batch_count
        epoch_losses.append(total_loss / total_examples)

    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        test_tensor = torch.from_numpy(test_features)
        for start in range(0, test_tensor.shape[0], config.batch_size):
            logits = model(test_tensor[start : start + config.batch_size].to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

    model = model.cpu()
    return model, np.concatenate(probabilities, axis=0), epoch_losses


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _validate_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    patient_ids: np.ndarray,
    sample_ids: np.ndarray,
) -> list[Any]:
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"features must be a non-empty 2D array; got {features.shape}.")
    n_samples = features.shape[0]
    for name, values in (
        ("labels", labels),
        ("folds", folds),
        ("patient_ids", patient_ids),
        ("sample_ids", sample_ids),
    ):
        if values.ndim != 1 or values.shape[0] != n_samples:
            raise ValueError(f"{name} must be 1D with {n_samples} entries.")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinite values.")
    if len(np.unique(sample_ids.astype(str))) != n_samples:
        raise ValueError("sample_ids must be unique.")

    fold_values = sorted(np.unique(folds).tolist(), key=str)
    if len(fold_values) != 3:
        raise ValueError(f"Exactly three outer folds are required; found {fold_values}.")
    for patient in np.unique(patient_ids):
        patient_folds = np.unique(folds[patient_ids == patient])
        if patient_folds.size != 1:
            raise ValueError(f"Patient {patient!r} occurs in multiple folds: {patient_folds.tolist()}.")
    return fold_values


def _save_classifier(
    path: Path,
    model: nn.Linear,
    *,
    seed: int,
    classes: np.ndarray,
    feature_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "classes": classes.tolist(),
            "feature_dim": feature_dim,
        },
        temporary,
    )
    os.replace(temporary, path)


def assemble_seed_oof_probabilities(
    fold_seed_probabilities: Sequence[np.ndarray],
    fold_test_indices: Sequence[np.ndarray],
    *,
    n_samples: int,
    seeds: Sequence[int],
) -> np.ndarray:
    """Scatter held-out fold probabilities into complete seed-specific OOF arrays.

    The returned shape is ``(n_seeds, n_samples, n_classes)``. This helper is
    deliberately strict so an overlap, omission, class-order shape mismatch,
    or seed-axis mismatch cannot silently enter canonical results.
    """

    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be a non-empty sequence of unique integers.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if len(fold_seed_probabilities) != len(fold_test_indices):
        raise ValueError("Fold probability arrays and test-index arrays must align.")
    if not fold_seed_probabilities:
        raise ValueError("At least one held-out fold probability array is required.")

    n_classes: int | None = None
    coverage = np.zeros(n_samples, dtype=np.int64)
    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_number, (probabilities, indices) in enumerate(
        zip(fold_seed_probabilities, fold_test_indices, strict=True)
    ):
        values = np.asarray(probabilities, dtype=np.float32)
        positions = np.asarray(indices)
        if values.ndim != 3 or values.shape[0] != len(seed_values):
            raise ValueError(
                f"Fold {fold_number} probabilities must have shape "
                f"({len(seed_values)}, n_test, n_classes); got {values.shape}."
            )
        if positions.ndim != 1 or positions.size != values.shape[1]:
            raise ValueError(
                f"Fold {fold_number} test indices do not match its probability rows."
            )
        if positions.dtype.kind not in {"i", "u"}:
            raise ValueError(f"Fold {fold_number} test indices must be integers.")
        positions = positions.astype(np.int64, copy=False)
        if positions.size == 0 or np.any(positions < 0) or np.any(positions >= n_samples):
            raise ValueError(f"Fold {fold_number} test indices are empty or out of bounds.")
        if np.unique(positions).size != positions.size:
            raise ValueError(f"Fold {fold_number} contains duplicate test indices.")
        if not np.isfinite(values).all():
            raise ValueError(f"Fold {fold_number} probabilities contain non-finite values.")
        if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
            raise ValueError(f"Fold {fold_number} probabilities lie outside [0, 1].")
        if not np.allclose(values.sum(axis=2), 1.0, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Fold {fold_number} probability rows do not sum to one.")
        if n_classes is None:
            n_classes = int(values.shape[2])
            if n_classes < 2:
                raise ValueError("At least two probability classes are required.")
        elif values.shape[2] != n_classes:
            raise ValueError("Class dimension differs across held-out folds.")
        coverage[positions] += 1
        normalized.append((values, positions))

    missing = np.flatnonzero(coverage == 0)
    duplicated = np.flatnonzero(coverage > 1)
    if missing.size or duplicated.size:
        raise ValueError(
            "Held-out folds must partition the complete OOF sample set exactly once; "
            f"missing={missing[:10].tolist()}, duplicated={duplicated[:10].tolist()}."
        )

    assert n_classes is not None
    oof = np.full(
        (len(seed_values), n_samples, n_classes), np.nan, dtype=np.float32
    )
    for values, positions in normalized:
        oof[:, positions, :] = values
    if np.isnan(oof).any():
        raise RuntimeError("Seed-specific OOF probability tensor is incomplete.")
    return oof


def _seed_metric_summary(seed_metrics: pd.DataFrame, *, ddof: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in ("balanced_accuracy", "macro_f1", "accuracy", "weighted_f1"):
        values = seed_metrics[metric].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=ddof)) if values.size > ddof else 0.0,
                "n_seeds": int(values.size),
                "ddof": int(ddof),
            }
        )
    return pd.DataFrame(rows)


def run_outer_cv_linear_probe(
    features: np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    folds: Sequence[Any] | np.ndarray,
    patient_ids: Sequence[Any] | np.ndarray,
    *,
    sample_ids: Sequence[Any] | np.ndarray | None = None,
    config: LinearProbeConfig | None = None,
    output_dir: str | Path | None = None,
) -> LinearProbeCVResult:
    """Run the frozen three-fold PAIR-BST linear-probe evaluation.

    Standardization and inverse-frequency class weights are fitted on the two
    training folds only.  There is no validation set, model selection, or
    early stopping.  When ``output_dir`` is supplied, all required audit
    artifacts are saved without changing the numerical protocol.
    """

    protocol = config or LinearProbeConfig()
    protocol.validate()
    matrix = np.asarray(features, dtype=np.float32)
    raw_labels = np.asarray(labels)
    fold_array = np.asarray(folds)
    patient_array = np.asarray(patient_ids)
    sample_array = (
        np.asarray(sample_ids)
        if sample_ids is not None
        else np.asarray([f"sample_{index:06d}" for index in range(matrix.shape[0])])
    )
    fold_values = _validate_inputs(
        matrix, raw_labels, fold_array, patient_array, sample_array
    )

    encoder = LabelEncoder().fit(raw_labels)
    encoded = encoder.transform(raw_labels).astype(np.int64)
    classes = encoder.classes_
    n_classes = int(classes.size)
    if n_classes < 2:
        raise ValueError("At least two outcome classes are required.")

    target_dir = Path(output_dir).resolve() if output_dir is not None else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)
        _reject_legacy_output_mixing(target_dir)
        _atomic_json(target_dir / "label_encoder.json", {"classes": classes.tolist()})

    fold_seed_rows: list[dict[str, Any]] = []
    fold_probability_arrays: list[np.ndarray] = []
    fold_test_indices: list[np.ndarray] = []

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        for held_fold in fold_values:
            test_mask = fold_array == held_fold
            train_mask = ~test_mask
            train_class_set = set(encoded[train_mask].tolist())
            missing = sorted(set(range(n_classes)) - train_class_set)
            if missing:
                missing_labels = classes[np.asarray(missing)].tolist()
                raise ValueError(
                    f"Training data for held fold {held_fold!r} omit classes {missing_labels}."
                )

            scaler = StandardScaler().fit(matrix[train_mask])
            train_scaled = scaler.transform(matrix[train_mask]).astype(np.float32)
            test_scaled = scaler.transform(matrix[test_mask]).astype(np.float32)
            class_weights = _inverse_frequency_weights(encoded[train_mask], n_classes)
            seed_probabilities: list[np.ndarray] = []
            fold_dir = (
                target_dir / f"fold_{_safe_component(held_fold)}"
                if target_dir is not None
                else None
            )

            if fold_dir is not None:
                fold_dir.mkdir(parents=True, exist_ok=True)
                _atomic_npz(
                    fold_dir / "standard_scaler.npz",
                    mean_=scaler.mean_,
                    scale_=scaler.scale_,
                    var_=scaler.var_,
                    n_features_in_=np.asarray([scaler.n_features_in_], dtype=np.int64),
                    n_samples_seen_=np.atleast_1d(scaler.n_samples_seen_),
                )
                _atomic_json(fold_dir / "label_encoder.json", {"classes": classes.tolist()})

            for seed in protocol.seeds:
                model, probabilities, epoch_losses = _train_one_seed(
                    train_scaled,
                    encoded[train_mask],
                    test_scaled,
                    n_classes=n_classes,
                    class_weights=class_weights,
                    seed=seed,
                    config=protocol,
                )
                seed_probabilities.append(probabilities)
                seed_prediction = probabilities.argmax(axis=1)
                seed_metric_values = classification_metrics(
                    encoded[test_mask], seed_prediction, labels=np.arange(n_classes)
                )
                fold_seed_rows.append(
                    {
                        "protocol_id": protocol.protocol_id,
                        "primary_metric_unit": "held_out_fold_per_seed_audit_only",
                        "held_fold": held_fold,
                        "seed": seed,
                        **scalar_metrics(seed_metric_values),
                        "n_patients": int(np.unique(patient_array[test_mask]).size),
                        "n_classes": n_classes,
                        "final_train_loss": epoch_losses[-1],
                    }
                )
                if fold_dir is not None:
                    _save_classifier(
                        fold_dir / f"classifier_seed_{seed}.pt",
                        model,
                        seed=seed,
                        classes=classes,
                        feature_dim=matrix.shape[1],
                    )
                    _atomic_json(
                        fold_dir / f"training_seed_{seed}.json",
                        {"seed": seed, "epoch_losses": epoch_losses},
                    )

            stacked_seed_probabilities = np.stack(seed_probabilities, axis=0).astype(
                np.float32, copy=False
            )
            held_indices = np.flatnonzero(test_mask).astype(np.int64, copy=False)
            fold_probability_arrays.append(stacked_seed_probabilities)
            fold_test_indices.append(held_indices)

            if fold_dir is not None:
                _atomic_npz(
                    fold_dir / "seed_probabilities.npz",
                    protocol_id=np.asarray(protocol.protocol_id),
                    seed_probabilities=stacked_seed_probabilities,
                    test_indices=held_indices,
                    seeds=np.asarray(protocol.seeds, dtype=np.int64),
                    classes=np.asarray(classes, dtype=str),
                )
                _atomic_json(
                    fold_dir / "training_context.json",
                    {
                        "protocol_id": protocol.protocol_id,
                        "held_fold": held_fold,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "class_weights": class_weights.tolist(),
                        "probability_ensemble_across_seeds": False,
                    },
                )
                _atomic_csv(
                    fold_dir / "seed_fold_metrics.csv",
                    pd.DataFrame(
                        [
                            row
                            for row in fold_seed_rows
                            if row["held_fold"] == held_fold
                        ]
                    ),
                )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)

    seed_oof_probabilities = assemble_seed_oof_probabilities(
        fold_probability_arrays,
        fold_test_indices,
        n_samples=matrix.shape[0],
        seeds=protocol.seeds,
    )
    seed_oof_predictions_encoded = seed_oof_probabilities.argmax(axis=2)
    seed_oof_predictions = np.stack(
        [encoder.inverse_transform(values) for values in seed_oof_predictions_encoded],
        axis=0,
    )

    seed_pooled_metrics: list[dict[str, Any]] = []
    seed_oof_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    raw_confusions: list[np.ndarray] = []
    normalized_confusions: list[np.ndarray] = []
    n_patients = int(np.unique(patient_array.astype(str)).size)
    for seed_index, seed in enumerate(protocol.seeds):
        diagnostic = classification_metrics(
            encoded,
            seed_oof_predictions_encoded[seed_index],
            labels=np.arange(n_classes),
        )
        seed_pooled_metrics.append({"seed": int(seed), **diagnostic})
        seed_oof_rows.append(
            {
                "protocol_id": protocol.protocol_id,
                "primary_metric_unit": protocol.primary_metric_unit,
                "seed": int(seed),
                **scalar_metrics(diagnostic),
                "n_patients": n_patients,
                "n_classes": n_classes,
            }
        )
        raw_confusions.append(np.asarray(diagnostic["confusion_matrix"], dtype=np.int64))
        normalized_confusions.append(
            np.asarray(diagnostic["confusion_matrix_row_normalized"], dtype=np.float64)
        )
        for class_row in diagnostic["per_class"]:
            class_id = int(class_row["label"])
            class_mask = encoded == class_id
            per_class_rows.append(
                {
                    "protocol_id": protocol.protocol_id,
                    "seed": int(seed),
                    "class_id": class_id,
                    "class_name": classes[class_id],
                    "precision": float(class_row["precision"]),
                    "recall": float(class_row["recall"]),
                    "f1": float(class_row["f1"]),
                    "roi_support": int(class_row["support"]),
                    "patient_support": int(
                        np.unique(patient_array[class_mask].astype(str)).size
                    ),
                }
            )

    seed_fold_frame = pd.DataFrame(fold_seed_rows)
    seed_oof_frame = pd.DataFrame(seed_oof_rows)
    seed_summary = _seed_metric_summary(
        seed_oof_frame, ddof=protocol.seed_sd_ddof
    )

    provenance = {
        "protocol": "PAIR-BST deterministic independent-seed OOF linear probe",
        "protocol_id": protocol.protocol_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(protocol),
        "fold_values": fold_values,
        "feature_shape": list(matrix.shape),
        "classes": classes.tolist(),
        "primary_estimator": "mean and sample standard deviation across complete seed-specific OOF metrics",
        "primary_metric_unit": protocol.primary_metric_unit,
        "seed_aggregation": protocol.seed_aggregation,
        "probability_ensemble_across_seeds": False,
        "seed_sd_ddof": protocol.seed_sd_ddof,
        "scaler_fit_scope": "two training folds only",
        "class_weight_scope": "two training folds only; inverse frequency normalized to mean 1",
        "validation_split": None,
        "early_stopping": False,
        "tf32_allowed": False if protocol.deterministic_algorithms else None,
        "input_sha256": {
            "features": _array_sha256(matrix),
            "labels": _array_sha256(raw_labels),
            "folds": _array_sha256(fold_array),
            "patient_ids": _array_sha256(patient_array),
            "sample_ids": _array_sha256(sample_array),
        },
        "package_versions": _package_versions(),
    }

    if target_dir is not None:
        _atomic_csv(target_dir / "seed_fold_metrics.csv", seed_fold_frame)
        _atomic_csv(target_dir / "seed_oof_metrics.csv", seed_oof_frame)
        _atomic_csv(target_dir / "seed_metric_mean_sd.csv", seed_summary)
        _atomic_npz(
            target_dir / "seed_oof_probabilities.npz",
            protocol_id=np.asarray(protocol.protocol_id),
            seeds=np.asarray(protocol.seeds, dtype=np.int64),
            sample_ids=np.asarray(sample_array, dtype=str),
            roi_uids=np.asarray(sample_array, dtype=str),
            patient_ids=np.asarray(patient_array, dtype=str),
            patient_uids=np.asarray(patient_array, dtype=str),
            fold_ids=np.asarray(fold_array),
            true_labels=np.asarray(raw_labels, dtype=str),
            true_labels_encoded=encoded,
            class_names=np.asarray(classes, dtype=str),
            probabilities=seed_oof_probabilities,
            predictions=np.asarray(seed_oof_predictions, dtype=str),
            predictions_encoded=seed_oof_predictions_encoded,
        )
        prediction_frames: list[pd.DataFrame] = []
        for seed_index, seed in enumerate(protocol.seeds):
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "protocol_id": protocol.protocol_id,
                        "seed": int(seed),
                        "sample_id": sample_array,
                        "patient_id": patient_array,
                        "fold": fold_array,
                        "true_label": raw_labels,
                        "predicted_label": seed_oof_predictions[seed_index],
                    }
                )
            )
        _atomic_csv(
            target_dir / "seed_oof_predictions.csv",
            pd.concat(prediction_frames, ignore_index=True),
        )
        _atomic_csv(
            target_dir / "seed_oof_per_class_metrics.csv",
            pd.DataFrame(per_class_rows),
        )
        _atomic_npz(
            target_dir / "seed_oof_confusion_matrices.npz",
            protocol_id=np.asarray(protocol.protocol_id),
            seeds=np.asarray(protocol.seeds, dtype=np.int64),
            class_names=np.asarray(classes, dtype=str),
            raw=np.stack(raw_confusions, axis=0),
            row_normalized=np.stack(normalized_confusions, axis=0),
        )
        _atomic_json(target_dir / "provenance.json", provenance)

    return LinearProbeCVResult(
        classes=classes,
        y_true_encoded=encoded,
        seeds=np.asarray(protocol.seeds, dtype=np.int64),
        seed_oof_probabilities=seed_oof_probabilities,
        seed_oof_predictions_encoded=seed_oof_predictions_encoded,
        seed_oof_predictions=seed_oof_predictions,
        seed_fold_metrics=seed_fold_frame,
        seed_oof_metrics=seed_oof_frame,
        seed_metric_summary=seed_summary,
        seed_pooled_metrics=seed_pooled_metrics,
        provenance=provenance,
        output_dir=target_dir,
    )
