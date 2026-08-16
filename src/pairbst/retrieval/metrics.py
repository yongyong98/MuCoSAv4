"""Per-query and aggregate retrieval metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _majority_vote(labels: np.ndarray, similarities: np.ndarray) -> Any:
    """Return a deterministic majority label.

    Ties are broken by summed cosine similarity and then by the string form of
    the label.  This rule is fixed so results do not depend on dictionary order.
    """

    if labels.size == 0:
        return None
    counts = Counter(labels.tolist())
    similarity_sums: dict[Any, float] = defaultdict(float)
    for label, similarity in zip(labels.tolist(), similarities.tolist()):
        similarity_sums[label] += float(similarity)
    return sorted(
        counts,
        key=lambda label: (-counts[label], -similarity_sums[label], str(label)),
    )[0]


def retrieval_metrics_from_neighbors(
    query_labels: Sequence[Any] | np.ndarray,
    neighbor_labels: np.ndarray,
    neighbor_similarities: np.ndarray,
    relevant_gallery_counts: Sequence[int] | np.ndarray,
    *,
    ks: Sequence[int] = (5, 10, 15, 20),
    query_ids: Sequence[Any] | np.ndarray | None = None,
    query_patient_ids: Sequence[Any] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate standard retrieval metrics for each query and each K.

    AP@K uses ``min(number of relevant gallery items, K)`` as its denominator.
    Precision uses the number of returned neighbors when a gallery contains
    fewer than K eligible items.  PAIR-BST galleries are larger than every
    frozen K, but the edge-case behavior is explicit for testing and reuse.
    """

    truth = np.asarray(query_labels)
    labels = np.asarray(neighbor_labels)
    similarities = np.asarray(neighbor_similarities, dtype=float)
    relevant_counts = np.asarray(relevant_gallery_counts, dtype=np.int64)
    if truth.ndim != 1:
        raise ValueError("query_labels must be one-dimensional.")
    if labels.ndim != 2 or similarities.shape != labels.shape:
        raise ValueError("neighbor labels and similarities must be equally shaped 2D arrays.")
    if labels.shape[0] != truth.shape[0] or relevant_counts.shape != truth.shape:
        raise ValueError("Query-level arrays must have matching lengths.")
    frozen_ks = tuple(sorted({int(k) for k in ks}))
    if not frozen_ks or frozen_ks[0] <= 0 or frozen_ks[-1] > labels.shape[1]:
        raise ValueError("ks must be positive and not exceed the saved neighbor width.")

    ids = np.asarray(query_ids) if query_ids is not None else np.arange(truth.size)
    patients = (
        np.asarray(query_patient_ids)
        if query_patient_ids is not None
        else np.asarray([f"query_{index}" for index in range(truth.size)])
    )
    if ids.shape != truth.shape or patients.shape != truth.shape:
        raise ValueError("query_ids and query_patient_ids must match query_labels.")

    rows: list[dict[str, Any]] = []
    for query_index in range(truth.shape[0]):
        valid = np.isfinite(similarities[query_index])
        for k in frozen_ks:
            usable_indices = np.flatnonzero(valid[:k])
            top_labels = labels[query_index, :k][usable_indices]
            top_similarities = similarities[query_index, :k][usable_indices]
            relevant = top_labels == truth[query_index]
            cumulative_relevant = np.cumsum(relevant, dtype=float)
            ranks = np.arange(1, relevant.size + 1, dtype=float)
            precision = float(relevant.mean()) if relevant.size else 0.0
            recall = (
                float(relevant.sum() / relevant_counts[query_index])
                if relevant_counts[query_index] > 0
                else 0.0
            )
            ap_denominator = min(int(relevant_counts[query_index]), k)
            average_precision = (
                float(((cumulative_relevant / ranks) * relevant).sum() / ap_denominator)
                if ap_denominator > 0 and relevant.size
                else 0.0
            )
            majority_label = _majority_vote(top_labels, top_similarities)
            rows.append(
                {
                    "query_index": query_index,
                    "query_id": ids[query_index],
                    "patient_id": patients[query_index],
                    "true_label": truth[query_index],
                    "k": k,
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "hit_at_k": float(bool(relevant.any())),
                    "average_precision_at_k": average_precision,
                    "majority_vote_label": majority_label,
                    "majority_vote_correct": float(majority_label == truth[query_index]),
                    "relevant_gallery_count": int(relevant_counts[query_index]),
                    "neighbors_returned": int(relevant.size),
                }
            )
    return pd.DataFrame(rows)


def aggregate_retrieval_metrics(per_query: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Mean query-level retrieval outcomes by the requested grouping."""

    metric_columns = [
        "precision_at_k",
        "recall_at_k",
        "hit_at_k",
        "average_precision_at_k",
        "majority_vote_correct",
    ]
    grouped = (
        per_query.groupby(group_columns, dropna=False, sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    counts = (
        per_query.groupby(group_columns, dropna=False, sort=True)
        .size()
        .rename("n_queries")
        .reset_index()
    )
    return grouped.merge(counts, on=group_columns, validate="one_to_one")
