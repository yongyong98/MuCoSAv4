"""Exhaustive, deterministic cosine retrieval with patient exclusion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import aggregate_retrieval_metrics, retrieval_metrics_from_neighbors


DEFAULT_KS = (5, 10, 15, 20)


@dataclass
class ExactCosineSearchResult:
    """Top-neighbor indices and similarities for an exhaustive search."""

    indices: np.ndarray
    similarities: np.ndarray
    eligible_counts: np.ndarray


@dataclass
class RetrievalCVResult:
    """Outputs from patient-disjoint three-fold retrieval."""

    neighbors: pd.DataFrame
    per_query_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    pooled_metrics: pd.DataFrame
    fold_metric_summary: pd.DataFrame
    provenance: dict[str, Any]
    output_dir: Path | None = None


def _normalize_rows(features: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains zero-norm feature vectors.")
    return matrix / norms


def exact_cosine_topk(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_patient_ids: Sequence[Any] | np.ndarray,
    gallery_patient_ids: Sequence[Any] | np.ndarray,
    *,
    gallery_ids: Sequence[Any] | np.ndarray | None = None,
    k: int = 20,
    query_chunk_size: int = 256,
) -> ExactCosineSearchResult:
    """Return exact top-K cosine neighbors after excluding the query patient.

    The search is exhaustive.  Equal similarities are broken by ``gallery_id``
    in lexical order, making neighbor ranks stable across platforms.
    """

    if k <= 0 or query_chunk_size <= 0:
        raise ValueError("k and query_chunk_size must be positive.")
    query = _normalize_rows(query_features, "query_features")
    gallery = _normalize_rows(gallery_features, "gallery_features")
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("Query and gallery feature dimensions must match.")
    query_patients = np.asarray(query_patient_ids)
    gallery_patients = np.asarray(gallery_patient_ids)
    if query_patients.shape != (query.shape[0],):
        raise ValueError("query_patient_ids length does not match query_features.")
    if gallery_patients.shape != (gallery.shape[0],):
        raise ValueError("gallery_patient_ids length does not match gallery_features.")
    ids = (
        np.asarray(gallery_ids).astype(str)
        if gallery_ids is not None
        else np.asarray([f"gallery_{index:09d}" for index in range(gallery.shape[0])])
    )
    if ids.shape != (gallery.shape[0],):
        raise ValueError("gallery_ids length does not match gallery_features.")
    if np.unique(ids).size != ids.size:
        raise ValueError("gallery_ids must be unique for deterministic tie-breaking.")

    output_indices = np.full((query.shape[0], k), -1, dtype=np.int64)
    output_similarities = np.full((query.shape[0], k), np.nan, dtype=np.float32)
    eligible_counts = np.zeros(query.shape[0], dtype=np.int64)
    for start in range(0, query.shape[0], query_chunk_size):
        stop = min(start + query_chunk_size, query.shape[0])
        similarities = query[start:stop] @ gallery.T
        for local_index, query_index in enumerate(range(start, stop)):
            eligible = np.flatnonzero(gallery_patients != query_patients[query_index])
            eligible_counts[query_index] = eligible.size
            if eligible.size == 0:
                continue
            values = similarities[local_index, eligible]
            order = np.lexsort((ids[eligible], -values))
            selected = eligible[order[:k]]
            width = selected.size
            output_indices[query_index, :width] = selected
            output_similarities[query_index, :width] = similarities[local_index, selected]

    valid_rows, valid_columns = np.where(output_indices >= 0)
    if np.any(
        query_patients[valid_rows]
        == gallery_patients[output_indices[valid_rows, valid_columns]]
    ):
        raise AssertionError("Same-patient item survived retrieval exclusion.")
    return ExactCosineSearchResult(output_indices, output_similarities, eligible_counts)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
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
        digest.update(memoryview(np.ascontiguousarray(values)).cast("B"))
    return digest.hexdigest()


def _validate_cv_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    patients: np.ndarray,
    sample_ids: np.ndarray,
) -> list[Any]:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array.")
    n = features.shape[0]
    for name, array in (
        ("labels", labels),
        ("folds", folds),
        ("patient_ids", patients),
        ("sample_ids", sample_ids),
    ):
        if array.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},).")
    if np.unique(sample_ids.astype(str)).size != n:
        raise ValueError("sample_ids must be unique.")
    fold_values = sorted(np.unique(folds).tolist(), key=str)
    if len(fold_values) != 3:
        raise ValueError(f"Exactly three folds are required; found {fold_values}.")
    for patient in np.unique(patients):
        membership = np.unique(folds[patients == patient])
        if membership.size != 1:
            raise ValueError(f"Patient {patient!r} occurs in multiple folds.")
    return fold_values


def _summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "precision_at_k",
        "recall_at_k",
        "hit_at_k",
        "average_precision_at_k",
        "majority_vote_correct",
    ]
    rows: list[dict[str, Any]] = []
    for k, group in fold_metrics.groupby("k", sort=True):
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "k": int(k),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "n_folds": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def run_patient_disjoint_cv_retrieval(
    features: np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    folds: Sequence[Any] | np.ndarray,
    patient_ids: Sequence[Any] | np.ndarray,
    *,
    sample_ids: Sequence[Any] | np.ndarray | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    query_chunk_size: int = 256,
    output_dir: str | Path | None = None,
) -> RetrievalCVResult:
    """Run held-fold-query versus other-two-fold-gallery exact retrieval."""

    matrix = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels)
    fold_array = np.asarray(folds)
    patient_array = np.asarray(patient_ids)
    id_array = (
        np.asarray(sample_ids)
        if sample_ids is not None
        else np.asarray([f"sample_{index:06d}" for index in range(matrix.shape[0])])
    )
    frozen_ks = tuple(sorted({int(k) for k in ks}))
    if frozen_ks != tuple(ks):
        raise ValueError("ks must be positive, unique, and in ascending order.")
    if not frozen_ks or frozen_ks[0] <= 0:
        raise ValueError("ks must contain positive integers.")
    fold_values = _validate_cv_inputs(matrix, label_array, fold_array, patient_array, id_array)
    max_k = max(frozen_ks)

    neighbor_frames: list[pd.DataFrame] = []
    per_query_frames: list[pd.DataFrame] = []
    for held_fold in fold_values:
        query_mask = fold_array == held_fold
        gallery_mask = ~query_mask
        query_indices = np.flatnonzero(query_mask)
        gallery_indices = np.flatnonzero(gallery_mask)
        search = exact_cosine_topk(
            matrix[query_mask],
            matrix[gallery_mask],
            patient_array[query_mask],
            patient_array[gallery_mask],
            gallery_ids=id_array[gallery_mask],
            k=max_k,
            query_chunk_size=query_chunk_size,
        )

        neighbor_labels = np.empty(search.indices.shape, dtype=label_array.dtype)
        if neighbor_labels.dtype.kind in {"U", "S"}:
            neighbor_labels[:] = ""
        elif neighbor_labels.dtype.kind == "O":
            neighbor_labels[:] = None
        else:
            neighbor_labels[:] = 0
        relevant_counts = np.zeros(query_indices.size, dtype=np.int64)
        fold_neighbor_rows: list[dict[str, Any]] = []
        for local_query, global_query in enumerate(query_indices):
            eligible_gallery = gallery_indices[
                patient_array[gallery_indices] != patient_array[global_query]
            ]
            relevant_counts[local_query] = int(
                np.sum(label_array[eligible_gallery] == label_array[global_query])
            )
            for rank in range(max_k):
                local_gallery = int(search.indices[local_query, rank])
                if local_gallery < 0:
                    continue
                global_gallery = int(gallery_indices[local_gallery])
                if patient_array[global_gallery] == patient_array[global_query]:
                    raise AssertionError("Retrieval output includes a same-patient neighbor.")
                neighbor_labels[local_query, rank] = label_array[global_gallery]
                fold_neighbor_rows.append(
                    {
                        "held_fold": held_fold,
                        "query_index": int(global_query),
                        "query_id": id_array[global_query],
                        "query_patient_id": patient_array[global_query],
                        "query_label": label_array[global_query],
                        "rank": rank + 1,
                        "gallery_index": global_gallery,
                        "gallery_id": id_array[global_gallery],
                        "gallery_patient_id": patient_array[global_gallery],
                        "gallery_label": label_array[global_gallery],
                        "cosine_similarity": float(search.similarities[local_query, rank]),
                        "relevant": bool(
                            label_array[global_gallery] == label_array[global_query]
                        ),
                    }
                )
        neighbor_frames.append(pd.DataFrame(fold_neighbor_rows))
        query_metrics = retrieval_metrics_from_neighbors(
            label_array[query_mask],
            neighbor_labels,
            search.similarities,
            relevant_counts,
            ks=frozen_ks,
            query_ids=id_array[query_mask],
            query_patient_ids=patient_array[query_mask],
        )
        query_metrics["query_index"] = np.repeat(query_indices, len(frozen_ks))
        query_metrics.insert(0, "held_fold", held_fold)
        per_query_frames.append(query_metrics)

    neighbors = pd.concat(neighbor_frames, ignore_index=True)
    per_query = pd.concat(per_query_frames, ignore_index=True)
    fold_metrics = aggregate_retrieval_metrics(per_query, ["held_fold", "k"])
    pooled_metrics = aggregate_retrieval_metrics(per_query, ["k"])
    fold_summary = _summarize_fold_metrics(fold_metrics)
    provenance = {
        "protocol": "PAIR-BST patient-disjoint 3-fold exact-cosine retrieval",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fold_values": fold_values,
        "ks": list(frozen_ks),
        "query_role": "held-out fold",
        "gallery_role": "other two folds",
        "same_patient_exclusion": True,
        "similarity": "exhaustive cosine after ROI-feature L2 normalization",
        "tie_break": "gallery sample_id lexical ascending",
        "saved_neighbor_limit": max_k,
        "feature_shape": list(matrix.shape),
        "input_sha256": {
            "features": _array_sha256(matrix),
            "labels": _array_sha256(label_array),
            "folds": _array_sha256(fold_array),
            "patient_ids": _array_sha256(patient_array),
            "sample_ids": _array_sha256(id_array),
        },
    }

    target_dir = Path(output_dir).resolve() if output_dir is not None else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)
        _atomic_csv(target_dir / "top20_neighbors.csv", neighbors)
        _atomic_csv(target_dir / "per_query_metrics.csv", per_query)
        _atomic_csv(target_dir / "fold_metrics.csv", fold_metrics)
        _atomic_csv(target_dir / "pooled_metrics.csv", pooled_metrics)
        _atomic_csv(target_dir / "fold_metric_mean_sd.csv", fold_summary)
        _atomic_json(target_dir / "provenance.json", provenance)

    return RetrievalCVResult(
        neighbors=neighbors,
        per_query_metrics=per_query,
        fold_metrics=fold_metrics,
        pooled_metrics=pooled_metrics,
        fold_metric_summary=fold_summary,
        provenance=provenance,
        output_dir=target_dir,
    )
