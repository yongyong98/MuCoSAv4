#!/usr/bin/env python3
"""Reconstruct canonical independent-seed complete-OOF PAIR-BST results.

This is a Path-A post-processing command.  It reads the immutable, held-out
seed probability arrays written by the verified seven-model run.  It does not
train a classifier, extract a feature, or execute retrieval.

The canonical statistical unit is one complete three-fold OOF evaluation for
one independently trained seed.  Probabilities, logits, and predictions are
never combined across seeds in this path.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


PROTOCOL_ID = "cv3_independent_seed_oof_v1"
ESTIMATOR_ID = "complete_oof_per_seed_metric_mean_sd"
LEGACY_ESTIMATOR_ID = "seed_probability_ensemble_fold_mean_sd"
SEEDS = np.asarray([101, 202, 303, 404, 505], dtype=np.int64)
FOLDS = (0, 1, 2)
TASKS = ("diagnosis", "differentiation", "growth_pattern")
TASK_LABELS = {
    "diagnosis": "Diagnosis",
    "differentiation": "Differentiation",
    "growth_pattern": "Growth Pattern",
}
STRATEGY_LABELS = {
    "center": "Center Crop",
    "mean": "Mean Pooling",
    "max": "Max Pooling",
}
MODEL_ORDER = (
    "resnet50_v2",
    "swin_t",
    "retccl",
    "uni",
    "uni2_h",
    "prov_gigapath",
    "virchow2",
)
STRATEGY_ORDER = ("center", "mean", "max")
EXPECTED_CLASS_COUNTS = {"diagnosis": 33, "differentiation": 11, "growth_pattern": 6}
PREDICTION_COLUMNS = [
    "protocol_id",
    "model_id",
    "model",
    "strategy",
    "task",
    "seed",
    "fold",
    "roi_uid",
    "wsi_uid",
    "patient_uid",
    "true_label",
    "predicted_label",
]
CONFUSION_COLUMNS = [
    "protocol_id",
    "model_id",
    "model",
    "strategy",
    "task",
    "seed",
    "true_class_id",
    "true_class_name",
    "predicted_class_id",
    "predicted_class_name",
    "value",
]
CONFUSION_SUMMARY_COLUMNS = [
    "protocol_id",
    "model_id",
    "model",
    "strategy",
    "task",
    "true_class_id",
    "true_class_name",
    "predicted_class_id",
    "predicted_class_name",
    "value",
]
METRIC_NAMES = ("balanced_accuracy", "macro_f1", "accuracy", "weighted_f1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    shown = path.resolve()
    if relative_to is not None:
        try:
            shown_text = shown.relative_to(relative_to.resolve()).as_posix()
        except ValueError:
            shown_text = path.as_posix()
    else:
        shown_text = path.as_posix()
    return {
        "path": shown_text,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def _labels_hash(classes_by_task: dict[str, np.ndarray]) -> str:
    payload = {
        task: classes_by_task[task].astype(str).tolist()
        for task in TASKS
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest().upper()


def _metadata_hash(frame: pd.DataFrame) -> str:
    columns = ["roi_uid", "wsi_id", "patient_uid", "diagnosis", *TASKS, "fold"]
    payload = frame[columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _encoded_metrics(
    true: np.ndarray, pred: np.ndarray, n_classes: int
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.arange(n_classes, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    total = int(support.sum())
    metrics = {
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "accuracy": float(np.mean(true == pred)),
        "weighted_f1": float(np.sum(f1 * support) / total),
    }
    return metrics, precision, recall, f1, support


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator != 0)


def _append_frame(handle: Any, frame: pd.DataFrame, *, header: bool) -> None:
    frame.to_csv(handle, index=False, header=header, lineterminator="\n")


def _confusion_long_frame(
    matrix: np.ndarray,
    *,
    model_id: str,
    model: str,
    strategy: str,
    task: str,
    classes: np.ndarray,
    seed: int | None,
) -> pd.DataFrame:
    n_classes = len(classes)
    true_id = np.repeat(np.arange(n_classes, dtype=np.int64), n_classes)
    pred_id = np.tile(np.arange(n_classes, dtype=np.int64), n_classes)
    values: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "model_id": model_id,
        "model": model,
        "strategy": strategy,
        "task": task,
        "true_class_id": true_id,
        "true_class_name": classes[true_id],
        "predicted_class_id": pred_id,
        "predicted_class_name": classes[pred_id],
        "value": np.asarray(matrix).reshape(-1),
    }
    if seed is not None:
        values["seed"] = int(seed)
        return pd.DataFrame(values)[CONFUSION_COLUMNS]
    return pd.DataFrame(values)[CONFUSION_SUMMARY_COLUMNS]


def _patient_label_signatures(labels: np.ndarray, patient_ids: np.ndarray) -> np.ndarray:
    table = pd.DataFrame({"patient": patient_ids.astype(str), "label": labels.astype(str)})
    mapping = table.groupby("patient", sort=False)["label"].agg(
        lambda values: json.dumps(
            sorted(set(values.astype(str).tolist())),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return table["patient"].map(mapping).to_numpy(dtype=str)


def _bootstrap_patient_weights(
    patient_ids: np.ndarray,
    strata: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patients, inverse = np.unique(patient_ids.astype(str), return_inverse=True)
    patient_strata = np.empty(len(patients), dtype=object)
    for patient_index in range(len(patients)):
        values = np.unique(strata[inverse == patient_index].astype(str))
        if len(values) != 1:
            raise ValueError(f"Patient-stable bootstrap stratum failed for {patients[patient_index]}")
        patient_strata[patient_index] = values[0]
    groups = [
        np.flatnonzero(patient_strata == value)
        for value in sorted(np.unique(patient_strata).tolist(), key=str)
    ]
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_bootstrap, len(patients)), dtype=np.float32)
    for group in groups:
        size = len(group)
        weights[:, group] = rng.multinomial(
            size,
            np.full(size, 1.0 / size, dtype=np.float64),
            size=n_bootstrap,
        ).astype(np.float32)
    return patients, inverse.astype(np.int64), weights


def _counts_by_patient(
    values: np.ndarray, patient_inverse: np.ndarray, n_patients: int, n_classes: int
) -> np.ndarray:
    code = patient_inverse * n_classes + values
    return np.bincount(code, minlength=n_patients * n_classes).reshape(
        n_patients, n_classes
    ).astype(np.float32)


def _true_positive_counts_by_patient(
    true: np.ndarray,
    pred: np.ndarray,
    patient_inverse: np.ndarray,
    n_patients: int,
    n_classes: int,
) -> np.ndarray:
    mask = true == pred
    code = patient_inverse[mask] * n_classes + true[mask]
    return np.bincount(code, minlength=n_patients * n_classes).reshape(
        n_patients, n_classes
    ).astype(np.float32)


def _bootstrap_ci_rows(
    folds: pd.DataFrame,
    predictions: dict[tuple[str, str, str], np.ndarray],
    systems: pd.DataFrame,
    classes_by_task: dict[str, np.ndarray],
    *,
    n_bootstrap: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Efficient, seed-specific patient-cluster percentile intervals.

    Bootstrap patient draws are shared across systems within each task.  This
    is useful for auditability and paired descriptive comparisons, although no
    new significance claims are generated here.
    """

    rows: list[dict[str, Any]] = []
    tail = (1.0 - confidence_level) / 2.0
    for task in TASKS:
        class_names = classes_by_task[task]
        n_classes = len(class_names)
        mapping = {name: index for index, name in enumerate(class_names.astype(str))}
        true = folds[task].astype(str).map(mapping).to_numpy(dtype=np.int64)
        patient_ids = folds["patient_uid"].astype(str).to_numpy()
        strata = _patient_label_signatures(folds[task].astype(str).to_numpy(), patient_ids)
        patients, patient_inverse, weights = _bootstrap_patient_weights(
            patient_ids,
            strata,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        actual_patient = _counts_by_patient(true, patient_inverse, len(patients), n_classes)
        actual_boot = weights @ actual_patient
        total_boot = actual_boot.sum(axis=1)
        task_systems = systems.loc[systems["task"] == task].reset_index(drop=True)
        evaluation_keys: list[tuple[str, str, str, int, str]] = []
        for system in task_systems.itertuples(index=False):
            for seed_index, seed_value in enumerate(SEEDS):
                evaluation_keys.append(
                    (system.model_id, system.strategy, task, seed_index, system.model)
                )

        # Batching keeps memory bounded while retaining efficient matrix multiplies.
        for start in range(0, len(evaluation_keys), 15):
            batch = evaluation_keys[start : start + 15]
            pred_blocks: list[np.ndarray] = []
            tp_blocks: list[np.ndarray] = []
            for model_id, strategy, task_name, seed_index, _model in batch:
                pred = predictions[(model_id, strategy, task_name)][seed_index]
                pred_blocks.append(
                    _counts_by_patient(pred, patient_inverse, len(patients), n_classes)
                )
                tp_blocks.append(
                    _true_positive_counts_by_patient(
                        true, pred, patient_inverse, len(patients), n_classes
                    )
                )
            pred_boot = (weights @ np.concatenate(pred_blocks, axis=1)).reshape(
                n_bootstrap, len(batch), n_classes
            )
            tp_boot = (weights @ np.concatenate(tp_blocks, axis=1)).reshape(
                n_bootstrap, len(batch), n_classes
            )
            actual = actual_boot[:, None, :]
            recall = np.divide(
                tp_boot,
                actual,
                out=np.zeros_like(tp_boot, dtype=np.float32),
                where=actual != 0,
            )
            precision = np.divide(
                tp_boot,
                pred_boot,
                out=np.zeros_like(tp_boot, dtype=np.float32),
                where=pred_boot != 0,
            )
            f1 = np.divide(
                2.0 * precision * recall,
                precision + recall,
                out=np.zeros_like(recall, dtype=np.float32),
                where=(precision + recall) != 0,
            )
            distributions = {
                "balanced_accuracy": recall.mean(axis=2),
                "macro_f1": f1.mean(axis=2),
                "accuracy": np.divide(
                    tp_boot.sum(axis=2),
                    total_boot[:, None],
                    out=np.zeros((n_bootstrap, len(batch)), dtype=np.float32),
                    where=total_boot[:, None] != 0,
                ),
                "weighted_f1": np.divide(
                    (f1 * actual).sum(axis=2),
                    total_boot[:, None],
                    out=np.zeros((n_bootstrap, len(batch)), dtype=np.float32),
                    where=total_boot[:, None] != 0,
                ),
            }
            for local_index, (model_id, strategy, task_name, seed_index, model) in enumerate(
                batch
            ):
                point, *_ = _encoded_metrics(
                    true, predictions[(model_id, strategy, task_name)][seed_index], n_classes
                )
                for metric_name in METRIC_NAMES:
                    low, high = np.quantile(
                        distributions[metric_name][:, local_index], [tail, 1.0 - tail]
                    )
                    rows.append(
                        {
                            "protocol_id": PROTOCOL_ID,
                            "model_id": model_id,
                            "model": model,
                            "strategy": strategy,
                            "task": task_name,
                            "seed": int(SEEDS[seed_index]),
                            "metric": metric_name,
                            "estimate": point[metric_name],
                            "ci_low": float(low),
                            "ci_high": float(high),
                            "confidence_level": confidence_level,
                            "n_bootstrap": n_bootstrap,
                            "bootstrap_seed": bootstrap_seed,
                            "n_patients": len(patients),
                            "stratified": True,
                            "strata_definition": "patient_unique_task_label_signature",
                        }
                    )
    return pd.DataFrame(rows)


def _display_cell(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def _build_table5(
    seed_metrics: pd.DataFrame,
    *,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], list[tuple[str, str]]]]:
    expected = set(int(value) for value in SEEDS)
    summary_rows: list[dict[str, Any]] = []
    for keys, group in seed_metrics.groupby(
        ["model_id", "model", "strategy", "task"], sort=False
    ):
        if set(group["seed"].astype(int)) != expected or len(group) != len(SEEDS):
            raise ValueError(f"Table 5 requires exactly five canonical seeds for {keys}")
        summary_rows.append(
            {
                "model_id": keys[0],
                "model": keys[1],
                "strategy": keys[2],
                "task": keys[3],
                "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
                "balanced_accuracy_sd": float(group["balanced_accuracy"].std(ddof=1)),
                "macro_f1_mean": float(group["macro_f1"].mean()),
                "macro_f1_sd": float(group["macro_f1"].std(ddof=1)),
                "n_seeds": len(group),
                "ddof": 1,
            }
        )
    summary = pd.DataFrame(summary_rows)
    order_model = {value: index for index, value in enumerate(MODEL_ORDER)}
    order_strategy = {value: index for index, value in enumerate(STRATEGY_ORDER)}
    summary = summary.sort_values(
        ["model_id", "strategy", "task"],
        key=lambda column: column.map(
            order_model if column.name == "model_id" else (
                order_strategy if column.name == "strategy" else {t: i for i, t in enumerate(TASKS)}
            )
        ),
    ).reset_index(drop=True)
    best: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for task in TASKS:
        task_rows = summary.loc[summary["task"] == task]
        for metric in ("balanced_accuracy", "macro_f1"):
            maximum = float(task_rows[f"{metric}_mean"].max())
            winners = task_rows.loc[
                task_rows[f"{metric}_mean"].map(lambda value: value == maximum),
                ["model_id", "strategy"],
            ]
            best[(task, metric)] = list(winners.itertuples(index=False, name=None))

    table_rows: list[dict[str, str]] = []
    for model_id in MODEL_ORDER:
        for strategy in STRATEGY_ORDER:
            system = summary.loc[
                (summary["model_id"] == model_id) & (summary["strategy"] == strategy)
            ]
            if len(system) != len(TASKS):
                raise ValueError(f"Incomplete Table 5 system: {model_id}/{strategy}")
            row: dict[str, str] = {
                "Model": str(system.iloc[0]["model"]),
                "Strategy": STRATEGY_LABELS[strategy],
            }
            for task in TASKS:
                selected = system.loc[system["task"] == task].iloc[0]
                row[f"{TASK_LABELS[task]} B.Acc"] = _display_cell(
                    float(selected["balanced_accuracy_mean"]),
                    float(selected["balanced_accuracy_sd"]),
                )
                row[f"{TASK_LABELS[task]} Macro-F1"] = _display_cell(
                    float(selected["macro_f1_mean"]),
                    float(selected["macro_f1_sd"]),
                )
            table_rows.append(row)
    table = pd.DataFrame(table_rows)
    _write_csv(output / "table5_manuscript.csv", table)
    _write_csv(output / "table5_source_seed_oof_metrics.csv", seed_metrics)

    headers = table.columns.tolist()
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    tex_lines = [
        "\\begin{tabular}{llllllll}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row_index, row in table.iterrows():
        model_id = MODEL_ORDER[row_index // 3]
        strategy = STRATEGY_ORDER[row_index % 3]
        md_values: list[str] = [str(row["Model"]), str(row["Strategy"])]
        tex_values: list[str] = [str(row["Model"]), str(row["Strategy"])]
        for task in TASKS:
            for metric, column_suffix in (
                ("balanced_accuracy", "B.Acc"),
                ("macro_f1", "Macro-F1"),
            ):
                column = f"{TASK_LABELS[task]} {column_suffix}"
                plain = str(row[column])
                is_best = (model_id, strategy) in best[(task, metric)]
                md_values.append(f"**{plain}**" if is_best else plain)
                tex_plain = plain.replace("±", "$\\pm$")
                tex_values.append(f"\\textbf{{{tex_plain}}}" if is_best else tex_plain)
        md_lines.append("| " + " | ".join(md_values) + " |")
        tex_lines.append(" & ".join(tex_values) + " \\\\")
    note = (
        "Values are the mean and sample standard deviation across five independently "
        "trained linear probes. Each seed was evaluated from complete patient-disjoint "
        "three-fold out-of-fold predictions."
    )
    md_lines.extend(["", f"Note: {note}", ""])
    tex_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\par\\smallskip\\footnotesize " + note,
            "",
        ]
    )
    _write_text(output / "table5_manuscript.md", "\n".join(md_lines))
    _write_text(output / "table5_manuscript.tex", "\n".join(tex_lines))
    return table, summary, best


def _legacy_comparison(
    legacy_fold_metrics: Path,
    revised_summary: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    legacy = pd.read_csv(legacy_fold_metrics)
    if len(legacy) != 189 or set(legacy["held_fold"].astype(int)) != set(FOLDS):
        raise ValueError("Legacy fold metrics do not have the verified 189-row schema")
    old_rows: list[dict[str, Any]] = []
    for keys, group in legacy.groupby(["model_id", "model", "strategy", "task"], sort=False):
        old_rows.append(
            {
                "model_id": keys[0],
                "model": keys[1],
                "strategy": keys[2],
                "task": keys[3],
                "legacy_balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
                "legacy_balanced_accuracy_sd": float(group["balanced_accuracy"].std(ddof=1)),
                "legacy_macro_f1_mean": float(group["macro_f1"].mean()),
                "legacy_macro_f1_sd": float(group["macro_f1"].std(ddof=1)),
            }
        )
    comparison = pd.DataFrame(old_rows).merge(
        revised_summary,
        on=["model_id", "model", "strategy", "task"],
        validate="one_to_one",
    )
    for metric in ("balanced_accuracy", "macro_f1"):
        comparison[f"mean_change_{metric}"] = (
            comparison[f"{metric}_mean"] - comparison[f"legacy_{metric}_mean"]
        )
        comparison[f"absolute_mean_change_{metric}"] = comparison[
            f"mean_change_{metric}"
        ].abs()
        comparison[f"sd_change_{metric}"] = (
            comparison[f"{metric}_sd"] - comparison[f"legacy_{metric}_sd"]
        )
        comparison[f"displayed_cell_changed_{metric}"] = [
            _display_cell(new_mean, new_sd) != _display_cell(old_mean, old_sd)
            for new_mean, new_sd, old_mean, old_sd in zip(
                comparison[f"{metric}_mean"],
                comparison[f"{metric}_sd"],
                comparison[f"legacy_{metric}_mean"],
                comparison[f"legacy_{metric}_sd"],
            )
        ]
    summary = {
        "legacy_estimator_id": LEGACY_ESTIMATOR_ID,
        "canonical_estimator_id": ESTIMATOR_ID,
        "mean_absolute_change_balanced_accuracy": float(
            comparison["absolute_mean_change_balanced_accuracy"].mean()
        ),
        "maximum_absolute_change_balanced_accuracy": float(
            comparison["absolute_mean_change_balanced_accuracy"].max()
        ),
        "mean_absolute_change_macro_f1": float(
            comparison["absolute_mean_change_macro_f1"].mean()
        ),
        "maximum_absolute_change_macro_f1": float(
            comparison["absolute_mean_change_macro_f1"].max()
        ),
        "displayed_table5_cells_changed": int(
            comparison[
                [
                    "displayed_cell_changed_balanced_accuracy",
                    "displayed_cell_changed_macro_f1",
                ]
            ].to_numpy().sum()
        ),
        "displayed_table5_cells_total": 126,
        "estimators_are_different": True,
    }
    legacy_dir = output / "legacy_comparison"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(legacy_dir / "table5_estimator_comparison.csv", comparison)
    _write_json(legacy_dir / "table5_estimator_comparison_summary.json", summary)
    note = f"""# Legacy comparison only

The files in this directory are audit evidence and are not canonical manuscript sources.

- Legacy estimator: `{LEGACY_ESTIMATOR_ID}`
- Canonical estimator: `{ESTIMATOR_ID}`
- Displayed Table 5 cells changed at three decimals: {summary['displayed_table5_cells_changed']}/126
- Mean absolute balanced-accuracy change: {summary['mean_absolute_change_balanced_accuracy']:.12f}
- Maximum absolute balanced-accuracy change: {summary['maximum_absolute_change_balanced_accuracy']:.12f}
- Mean absolute macro-F1 change: {summary['mean_absolute_change_macro_f1']:.12f}
- Maximum absolute macro-F1 change: {summary['maximum_absolute_change_macro_f1']:.12f}

The legacy uncertainty is the sample standard deviation of three seed-probability-ensemble fold metrics. The canonical uncertainty is the sample standard deviation of five independently computed complete-OOF seed metrics. These quantities have different estimands and must not be interchanged.
"""
    _write_text(legacy_dir / "README.md", note)
    return comparison, summary


def _copy_retrieval_unchanged(
    prior_package: Path,
    output: Path,
    *,
    workspace: Path,
) -> tuple[list[dict[str, Any]], bool]:
    mapping = [
        ("01_MANUSCRIPT_RESULTS/retrieval_primary.csv", "retrieval/retrieval_primary.csv"),
        ("01_MANUSCRIPT_RESULTS/retrieval_primary.md", "retrieval/retrieval_primary.md"),
        ("01_MANUSCRIPT_RESULTS/retrieval_primary.tex", "retrieval/retrieval_primary.tex"),
        ("02_AUDIT_RESULTS/retrieval/retrieval_fold_metrics.csv", "retrieval/retrieval_fold_metrics.csv"),
        ("02_AUDIT_RESULTS/retrieval/retrieval_per_query_metrics.csv", "retrieval/retrieval_per_query_metrics.csv"),
        ("02_AUDIT_RESULTS/retrieval/retrieval_pooled_metrics.csv", "retrieval/retrieval_pooled_metrics.csv"),
        ("02_AUDIT_RESULTS/statistics/retrieval_paired_comparisons_holm.csv", "retrieval_statistics/retrieval_paired_comparisons_holm.csv"),
        ("02_AUDIT_RESULTS/statistics/retrieval_patient_cluster_ci.csv", "retrieval_statistics/retrieval_patient_cluster_ci.csv"),
    ]
    checks: list[dict[str, Any]] = []
    for source_relative, destination_relative in mapping:
        source = prior_package / Path(source_relative)
        destination = output / Path(destination_relative)
        if not source.is_file():
            raise FileNotFoundError(f"Verified retrieval source missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = _sha256(source)
        destination_hash = _sha256(destination)
        checks.append(
            {
                "source_path": source.resolve().relative_to(workspace.resolve()).as_posix(),
                "destination_path": destination.resolve().relative_to(output.resolve()).as_posix(),
                "source_sha256": source_hash,
                "destination_sha256": destination_hash,
                "size_bytes": int(destination.stat().st_size),
                "comparison_result": "PASS" if source_hash == destination_hash else "FAIL",
            }
        )
    passed = all(row["comparison_result"] == "PASS" for row in checks)
    source_manifest = prior_package / "02_AUDIT_RESULTS/retrieval/retrieval_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Verified retrieval manifest missing: {source_manifest}")
    source_manifest_reference = _identity(source_manifest, relative_to=workspace)
    _write_json(
        output / "RETRIEVAL_UNCHANGED_VERIFICATION.json",
        {
            "schema": "pairbst.retrieval_unchanged_verification.v1",
            "protocol_id": PROTOCOL_ID,
            "retrieval_was_rerun": False,
            "files": checks,
            "source_reference_only": {
                **source_manifest_reference,
                "reason_not_copied": (
                    "The verified legacy manifest contains machine-local absolute paths. "
                    "Its identity is recorded without copying it into the portable result tree."
                ),
            },
            "result": "PASS" if passed else "FAIL",
        },
    )
    return checks, passed


def _git_state(path: Path) -> dict[str, Any]:
    try:
        top = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        )
        return {
            "repository_present": True,
            "top_level": Path(top).name,
            "commit": commit,
            "working_tree_clean": not bool(status.strip()),
            "working_tree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest().upper(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "repository_present": False,
            "commit": None,
            "working_tree_clean": None,
            "note": "No Git metadata was available in the local source tree at generation time.",
        }


def _collect_identities(
    root: Path,
    *,
    include: callable | None = None,
    exclude_names: Iterable[str] = (),
) -> list[dict[str, Any]]:
    excluded = set(exclude_names)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in excluded
        and (include(path) if include is not None else True)
    ]
    return [_identity(path, relative_to=root) for path in sorted(files)]


def _parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    repository = script.parents[1]
    workspace = repository.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classification-root",
        type=Path,
        default=repository / "outputs/runs/classification/official_model_specific_7model_v1",
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=repository / "locks/folds_cv3_v1.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repository / "configs/protocol_cv3_independent_seed_oof_v1.yaml",
    )
    parser.add_argument(
        "--prior-package",
        type=Path,
        default=workspace
        / "PAIR_BST_7MODEL_revision_results_with_reviewer_evidence_INTERNAL_CONTROLLED_20260816_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace
        / "PAIR_BST_7MODEL_revision_results_independent_seed_oof_20260816_v4",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    script_path = Path(__file__).resolve()
    repository = script_path.parents[1]
    workspace = repository.parent
    classification_root = args.classification_root.resolve()
    folds_path = args.folds.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    prior_package = args.prior_package.resolve()
    if not classification_root.is_dir() or not folds_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Classification root, fold manifest, or canonical config is missing")
    output.mkdir(parents=True, exist_ok=True)
    probability_dir = output / "classification_seed_oof_probabilities"
    confusion_dir = output / "confusion_matrices"
    probability_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)

    generated_utc = datetime.now(timezone.utc).isoformat()
    folds = pd.read_csv(folds_path)
    required_fold_columns = {
        "roi_uid",
        "wsi_id",
        "patient_uid",
        "diagnosis",
        "differentiation",
        "growth_pattern",
        "fold",
    }
    if missing := sorted(required_fold_columns.difference(folds.columns)):
        raise ValueError(f"Fold manifest lacks required columns: {missing}")
    if len(folds) != 2252 or folds["roi_uid"].duplicated().any():
        raise ValueError("Expected 2,252 unique ROI rows in the frozen fold manifest")
    if set(folds["fold"].astype(int)) != set(FOLDS):
        raise ValueError("Frozen fold manifest does not contain folds 0, 1, and 2")

    legacy_pooled = pd.read_csv(classification_root / "classification_pooled_metrics.csv")
    systems = legacy_pooled[["model_id", "model", "strategy", "task"]].drop_duplicates()
    if len(systems) != 63:
        raise ValueError(f"Expected 63 systems, found {len(systems)}")
    expected_grid = {
        (model_id, strategy, task)
        for model_id in MODEL_ORDER
        for strategy in STRATEGY_ORDER
        for task in TASKS
    }
    observed_grid = set(systems[["model_id", "strategy", "task"]].itertuples(index=False, name=None))
    if observed_grid != expected_grid:
        raise ValueError("The live classification grid is not the expected 7 x 3 x 3 grid")
    model_names = systems.drop_duplicates("model_id").set_index("model_id")["model"].to_dict()

    prior_manifest_path = classification_root / "classification_manifest.json"
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    prior_outputs = {
        str(Path(item["path"]).resolve()).casefold(): item
        for item in prior_manifest["output_identities"]
    }

    metrics_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    source_identities: list[dict[str, Any]] = []
    classes_by_task: dict[str, np.ndarray] = {}
    predictions: dict[tuple[str, str, str], np.ndarray] = {}
    validation_counts: defaultdict[str, int] = defaultdict(int)

    prediction_path = output / "classification_seed_oof_predictions.csv.gz"
    raw_confusion_path = confusion_dir / "confusion_matrices_seed_raw_long.csv.gz"
    normalized_confusion_path = (
        confusion_dir / "confusion_matrices_seed_row_normalized_long.csv.gz"
    )
    mean_confusion_path = confusion_dir / "confusion_matrices_mean_row_normalized.csv.gz"
    sd_confusion_path = confusion_dir / "confusion_matrices_sd_row_normalized.csv.gz"
    for managed in (
        prediction_path,
        raw_confusion_path,
        normalized_confusion_path,
        mean_confusion_path,
        sd_confusion_path,
    ):
        if managed.exists():
            managed.unlink()

    with (
        gzip.open(prediction_path, "wt", encoding="utf-8", newline="") as prediction_handle,
        gzip.open(raw_confusion_path, "wt", encoding="utf-8", newline="") as raw_handle,
        gzip.open(normalized_confusion_path, "wt", encoding="utf-8", newline="") as normalized_handle,
        gzip.open(mean_confusion_path, "wt", encoding="utf-8", newline="") as mean_handle,
        gzip.open(sd_confusion_path, "wt", encoding="utf-8", newline="") as sd_handle,
    ):
        first_system = True
        ordered_systems = systems.assign(
            _model=systems["model_id"].map({v: i for i, v in enumerate(MODEL_ORDER)}),
            _strategy=systems["strategy"].map({v: i for i, v in enumerate(STRATEGY_ORDER)}),
            _task=systems["task"].map({v: i for i, v in enumerate(TASKS)}),
        ).sort_values(["_model", "_strategy", "_task"])
        for system in ordered_systems.itertuples(index=False):
            model_id, model, strategy, task = (
                system.model_id,
                system.model,
                system.strategy,
                system.task,
            )
            system_dir = classification_root / model_id / strategy / task
            encoder_path = system_dir / "label_encoder.json"
            encoder = json.loads(encoder_path.read_text(encoding="utf-8"))
            classes = np.asarray(encoder["classes"], dtype=str)
            if len(classes) != EXPECTED_CLASS_COUNTS[task]:
                raise ValueError(f"Unexpected class count for {model_id}/{strategy}/{task}")
            if task in classes_by_task and not np.array_equal(classes_by_task[task], classes):
                raise ValueError(f"Class order differs across systems for task {task}")
            classes_by_task.setdefault(task, classes)
            mapping = {name: index for index, name in enumerate(classes.tolist())}
            try:
                true = folds[task].astype(str).map(mapping).to_numpy(dtype=np.int64)
            except (TypeError, ValueError) as error:
                raise ValueError(f"A true label is absent from the encoder for {task}") from error
            if pd.isna(folds[task].astype(str).map(mapping)).any():
                raise ValueError(f"A true label is absent from the encoder for {task}")
            n_samples, n_classes = len(folds), len(classes)
            probabilities = np.full((len(SEEDS), n_samples, n_classes), np.nan, dtype=np.float32)
            fold_assignments = np.full(n_samples, -1, dtype=np.int16)
            filled = np.zeros(n_samples, dtype=bool)

            for held_fold in FOLDS:
                fold_dir = system_dir / f"fold_{held_fold}"
                npz_path = fold_dir / "seed_and_mean_probabilities.npz"
                identity = _identity(npz_path, relative_to=workspace)
                bound = prior_outputs.get(str(npz_path.resolve()).casefold())
                if bound is None:
                    raise ValueError(f"Source NPZ is not bound by prior classification manifest: {npz_path}")
                if identity["sha256"] != str(bound["sha256"]).upper() or identity["size_bytes"] != int(
                    bound["size_bytes"]
                ):
                    raise ValueError(f"Source NPZ differs from prior manifest: {npz_path}")
                source_identities.append(identity)
                expected_indices = np.flatnonzero(folds["fold"].to_numpy(dtype=int) == held_fold)
                with np.load(npz_path, allow_pickle=False) as archive:
                    required_keys = {"seed_probabilities", "mean_probabilities", "test_indices", "seeds"}
                    if set(archive.files) != required_keys:
                        raise ValueError(f"Unexpected NPZ schema at {npz_path}: {archive.files}")
                    seed_probabilities = archive["seed_probabilities"]
                    mean_probabilities = archive["mean_probabilities"]
                    test_indices = archive["test_indices"]
                    source_seeds = archive["seeds"]
                if not np.array_equal(source_seeds, SEEDS):
                    raise ValueError(f"Seed order differs at {npz_path}")
                expected_shape = (len(SEEDS), len(expected_indices), n_classes)
                if seed_probabilities.shape != expected_shape or seed_probabilities.dtype != np.float32:
                    raise ValueError(
                        f"Probability shape/dtype mismatch at {npz_path}: {seed_probabilities.shape}"
                    )
                if not np.array_equal(test_indices, expected_indices):
                    raise ValueError(f"Held-out sample order differs at {npz_path}")
                fold_encoder = json.loads(
                    (fold_dir / "label_encoder.json").read_text(encoding="utf-8")
                )
                if fold_encoder["classes"] != encoder["classes"]:
                    raise ValueError(f"Class order differs across folds at {fold_dir}")
                recalculated_mean = seed_probabilities.mean(axis=0)
                if not np.array_equal(mean_probabilities, recalculated_mean):
                    raise ValueError(f"Stored legacy mean probability failed verification at {npz_path}")
                if not np.isfinite(seed_probabilities).all():
                    raise ValueError(f"NaN or Inf in seed probabilities at {npz_path}")
                if float(np.max(np.abs(seed_probabilities.sum(axis=2) - 1.0))) > 1e-5:
                    raise ValueError(f"Probability rows do not sum to one at {npz_path}")
                if (seed_probabilities < -1e-6).any() or (seed_probabilities > 1.000001).any():
                    raise ValueError(f"Probability outside [0,1] at {npz_path}")
                if filled[test_indices].any():
                    raise ValueError(f"Held-out folds overlap for {model_id}/{strategy}/{task}")
                probabilities[:, test_indices, :] = seed_probabilities
                filled[test_indices] = True
                fold_assignments[test_indices] = held_fold
                for seed_index, seed_value in enumerate(SEEDS):
                    fold_pred = np.argmax(seed_probabilities[seed_index], axis=1).astype(np.int64)
                    fold_metrics, *_ = _encoded_metrics(
                        true[test_indices], fold_pred, n_classes
                    )
                    training_record = json.loads(
                        (fold_dir / f"training_seed_{int(seed_value)}.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    if int(training_record["seed"]) != int(seed_value):
                        raise ValueError(f"Training record seed mismatch at {fold_dir}")
                    fold_metric_rows.append(
                        {
                            "protocol_id": PROTOCOL_ID,
                            "model_id": model_id,
                            "model": model,
                            "strategy": strategy,
                            "task": task,
                            "seed": int(seed_value),
                            "held_fold": held_fold,
                            **fold_metrics,
                            "n_samples": len(test_indices),
                            "n_patients": int(
                                folds.iloc[test_indices]["patient_uid"].astype(str).nunique()
                            ),
                            "n_classes": n_classes,
                            "final_train_loss": float(training_record["epoch_losses"][-1]),
                            "primary_metric_unit": False,
                        }
                    )
            if not filled.all() or (fold_assignments < 0).any() or not np.isfinite(probabilities).all():
                raise ValueError(f"Incomplete complete-OOF reconstruction for {model_id}/{strategy}/{task}")
            system_predictions = np.argmax(probabilities, axis=2).astype(np.int16)
            predictions[(model_id, strategy, task)] = system_predictions
            normalized_confusions: list[np.ndarray] = []
            for seed_index, seed_value in enumerate(SEEDS):
                pred = system_predictions[seed_index].astype(np.int64)
                metric_values, precision, recall, f1, support = _encoded_metrics(
                    true, pred, n_classes
                )
                metrics_rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "model_id": model_id,
                        "model": model,
                        "strategy": strategy,
                        "task": task,
                        "seed": int(seed_value),
                        **metric_values,
                        "n_samples": n_samples,
                        "n_patients": int(folds["patient_uid"].astype(str).nunique()),
                        "n_classes": n_classes,
                    }
                )
                prediction_frame = pd.DataFrame(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "model_id": model_id,
                        "model": model,
                        "strategy": strategy,
                        "task": task,
                        "seed": int(seed_value),
                        "fold": fold_assignments,
                        "roi_uid": folds["roi_uid"].astype(str).to_numpy(),
                        "wsi_uid": folds["wsi_id"].astype(str).to_numpy(),
                        "patient_uid": folds["patient_uid"].astype(str).to_numpy(),
                        "true_label": classes[true],
                        "predicted_label": classes[pred],
                    }
                )[PREDICTION_COLUMNS]
                _append_frame(prediction_handle, prediction_frame, header=first_system and seed_index == 0)
                for class_id, class_name in enumerate(classes):
                    class_mask = true == class_id
                    per_class_rows.append(
                        {
                            "protocol_id": PROTOCOL_ID,
                            "model_id": model_id,
                            "model": model,
                            "strategy": strategy,
                            "task": task,
                            "seed": int(seed_value),
                            "class_id": class_id,
                            "class_name": class_name,
                            "precision": float(precision[class_id]),
                            "recall": float(recall[class_id]),
                            "f1": float(f1[class_id]),
                            "roi_support": int(support[class_id]),
                            "patient_support": int(
                                folds.loc[class_mask, "patient_uid"].astype(str).nunique()
                            ),
                        }
                    )
                counts = confusion_matrix(true, pred, labels=np.arange(n_classes)).astype(np.int64)
                normalized = _row_normalize(counts)
                normalized_confusions.append(normalized)
                _append_frame(
                    raw_handle,
                    _confusion_long_frame(
                        counts,
                        model_id=model_id,
                        model=model,
                        strategy=strategy,
                        task=task,
                        classes=classes,
                        seed=int(seed_value),
                    ),
                    header=first_system and seed_index == 0,
                )
                _append_frame(
                    normalized_handle,
                    _confusion_long_frame(
                        normalized,
                        model_id=model_id,
                        model=model,
                        strategy=strategy,
                        task=task,
                        classes=classes,
                        seed=int(seed_value),
                    ),
                    header=first_system and seed_index == 0,
                )
            normalized_stack = np.stack(normalized_confusions, axis=0)
            _append_frame(
                mean_handle,
                _confusion_long_frame(
                    normalized_stack.mean(axis=0),
                    model_id=model_id,
                    model=model,
                    strategy=strategy,
                    task=task,
                    classes=classes,
                    seed=None,
                ),
                header=first_system,
            )
            _append_frame(
                sd_handle,
                _confusion_long_frame(
                    normalized_stack.std(axis=0, ddof=1),
                    model_id=model_id,
                    model=model,
                    strategy=strategy,
                    task=task,
                    classes=classes,
                    seed=None,
                ),
                header=first_system,
            )
            probability_path = probability_dir / f"{model_id}__{strategy}__{task}.npz"
            np.savez_compressed(
                probability_path,
                protocol_id=np.asarray(PROTOCOL_ID),
                model_id=np.asarray(model_id),
                model=np.asarray(model),
                strategy=np.asarray(strategy),
                task=np.asarray(task),
                seeds=SEEDS,
                roi_uids=folds["roi_uid"].to_numpy(dtype=str),
                wsi_uids=folds["wsi_id"].to_numpy(dtype=str),
                patient_uids=folds["patient_uid"].to_numpy(dtype=str),
                fold_ids=fold_assignments,
                true_labels=classes[true],
                true_label_ids=true.astype(np.int16),
                class_names=classes,
                probabilities=probabilities,
                predictions=system_predictions,
                predicted_labels=classes[system_predictions],
                probability_ensemble_across_seeds=np.asarray(False),
            )
            validation_counts["systems"] += 1
            validation_counts["source_npz"] += len(FOLDS)
            validation_counts["seed_oof"] += len(SEEDS)
            validation_counts["prediction_rows"] += len(SEEDS) * n_samples
            validation_counts["per_class_rows"] += len(SEEDS) * n_classes
            first_system = False

    seed_metrics = pd.DataFrame(metrics_rows)
    seed_fold_metrics = pd.DataFrame(fold_metric_rows)
    per_class = pd.DataFrame(per_class_rows)
    if len(seed_metrics) != 315 or len(seed_fold_metrics) != 945 or len(per_class) != 5250:
        raise ValueError(
            f"Unexpected row counts: metrics={len(seed_metrics)}, fold={len(seed_fold_metrics)}, per-class={len(per_class)}"
        )
    _write_csv(output / "classification_seed_oof_metrics.csv", seed_metrics)
    _write_csv(output / "classification_seed_fold_metrics.csv", seed_fold_metrics)
    _write_csv(output / "classification_per_class_seed_oof.csv", per_class)
    per_class_summary = (
        per_class.groupby(
            [
                "protocol_id",
                "model_id",
                "model",
                "strategy",
                "task",
                "class_id",
                "class_name",
            ],
            sort=False,
            as_index=False,
        )
        .agg(
            f1_mean=("f1", "mean"),
            f1_sd=("f1", lambda values: values.std(ddof=1)),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", lambda values: values.std(ddof=1)),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", lambda values: values.std(ddof=1)),
            roi_support=("roi_support", "first"),
            patient_support=("patient_support", "first"),
            n_seeds=("seed", "nunique"),
        )
    )
    if len(per_class_summary) != 1050 or not (per_class_summary["n_seeds"] == 5).all():
        raise ValueError("Unexpected per-class seed summary row count or seed count")
    _write_csv(output / "classification_per_class_seed_summary.csv", per_class_summary)

    table5, table5_summary, best = _build_table5(seed_metrics, output=output)
    comparison, change_summary = _legacy_comparison(
        classification_root / "classification_fold_metrics.csv", table5_summary, output
    )
    legacy_stats_source = (
        repository
        / "outputs/runs/statistics/official_model_specific_7model_v1/classification_paired_comparisons_holm.csv"
    )
    if legacy_stats_source.is_file():
        shutil.copy2(
            legacy_stats_source,
            output
            / "legacy_comparison/classification_paired_comparisons_holm_LEGACY_ENSEMBLE.csv",
        )

    ci_frame = _bootstrap_ci_rows(
        folds,
        predictions,
        systems,
        classes_by_task,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    if len(ci_frame) != 1260:
        raise ValueError(f"Expected 1,260 seed-specific CI rows, found {len(ci_frame)}")
    _write_csv(output / "classification_patient_cluster_ci_by_seed.csv", ci_frame)

    retrieval_checks, retrieval_passed = _copy_retrieval_unchanged(
        prior_package, output, workspace=workspace
    )
    if not retrieval_passed:
        raise ValueError("Retrieval unchanged verification failed")

    class_order_hash = _labels_hash(classes_by_task)
    source_hashes = {item["path"]: item["sha256"] for item in source_identities}
    if len(source_hashes) != 189:
        raise ValueError(f"Expected 189 distinct source NPZ identities, found {len(source_hashes)}")
    protocol_record = {
        "schema": "pairbst.classification_protocol.v1",
        "protocol_id": PROTOCOL_ID,
        "estimator_id": ESTIMATOR_ID,
        "canonical": True,
        "primary_metric_unit": "complete_oof_per_seed",
        "seed_aggregation": "metric_mean_sd",
        "probability_ensemble_across_seeds": False,
        "hard_vote_across_seeds": False,
        "seed_sd_ddof": 1,
        "seeds": SEEDS.tolist(),
        "folds": list(FOLDS),
        "n_systems": 63,
        "n_samples_per_seed_system": len(folds),
        "n_patients": int(folds["patient_uid"].nunique()),
        "class_counts": EXPECTED_CLASS_COUNTS,
        "source_probability_files": 189,
        "linear_head_retrained": False,
        "features_reextracted": False,
        "retrieval_rerun": False,
    }
    _write_json(output / "classification_protocol.json", protocol_record)

    classification_file_names = {
        "classification_seed_oof_metrics.csv",
        "classification_seed_fold_metrics.csv",
        "classification_seed_oof_predictions.csv.gz",
        "classification_per_class_seed_oof.csv",
        "classification_per_class_seed_summary.csv",
        "classification_patient_cluster_ci_by_seed.csv",
        "table5_manuscript.csv",
        "table5_manuscript.md",
        "table5_manuscript.tex",
        "table5_source_seed_oof_metrics.csv",
        "classification_protocol.json",
    }
    classification_outputs = _collect_identities(
        output,
        include=lambda path: (
            (path.parent == output and path.name in classification_file_names)
            or probability_dir in path.parents
            or confusion_dir in path.parents
        ),
    )
    classification_manifest = {
        "schema": "pairbst.classification_manifest.v2",
        "action": "classify.run",
        "profile": "official_model_specific",
        "protocol_id": PROTOCOL_ID,
        "estimator_id": ESTIMATOR_ID,
        "generated_utc": generated_utc,
        "source_stage": "path_a_verified_seed_probabilities",
        "source_npz_count": 189,
        "source_npz_identities": source_identities,
        "source_legacy_manifest": _identity(prior_manifest_path, relative_to=workspace),
        "config_identity": _identity(config_path, relative_to=workspace),
        "fold_manifest_identity": _identity(folds_path, relative_to=workspace),
        "metadata_sha256": _metadata_hash(folds),
        "class_order_sha256": class_order_hash,
        "seeds": SEEDS.tolist(),
        "sample_counts": {
            "systems": 63,
            "seed_complete_oof_evaluations": 315,
            "seed_fold_audit_evaluations": 945,
            "seed_oof_prediction_rows": 709380,
            "per_class_seed_rows": 5250,
            "per_class_seed_summary_rows": 1050,
        },
        "primary_metric_unit": "complete_oof_per_seed",
        "seed_aggregation": "metric_mean_sd",
        "seed_sd_ddof": 1,
        "probability_ensemble_across_seeds": False,
        "linear_head_retrained": False,
        "feature_extraction_rerun": False,
        "output_identities": classification_outputs,
    }
    _write_json(output / "classification_manifest.json", classification_manifest)
    statistics_manifest = {
        "schema": "pairbst.statistics_manifest.v2",
        "protocol_id": PROTOCOL_ID,
        "estimator_id": ESTIMATOR_ID,
        "generated_utc": generated_utc,
        "primary_uncertainty": "sample_sd_across_five_complete_oof_seed_metrics",
        "seed_sd_ddof": 1,
        "patient_cluster_bootstrap": {
            "scope": "separately_for_each_seed_specific_complete_oof_prediction_set",
            "n_bootstrap": args.n_bootstrap,
            "confidence_level": args.confidence_level,
            "bootstrap_seed": args.bootstrap_seed,
            "strata_definition": "patient_unique_task_label_signature",
            "intervals_are_not_collapsed_across_seeds": True,
        },
        "paired_statistical_comparisons": {
            "canonical_significance_analysis_generated": False,
            "legacy_ensemble_result_location": "legacy_comparison",
        },
        "input_seed_prediction_identity": _identity(prediction_path, relative_to=output),
        "output_identities": [
            _identity(output / "classification_patient_cluster_ci_by_seed.csv", relative_to=output)
        ],
    }
    _write_json(output / "statistics_manifest.json", statistics_manifest)

    best_records: dict[str, Any] = {}
    for task in TASKS:
        best_records[task] = {}
        for metric in ("balanced_accuracy", "macro_f1"):
            winner_keys = best[(task, metric)]
            best_records[task][metric] = [
                {
                    "model_id": model_id,
                    "model": model_names[model_id],
                    "strategy": strategy,
                    "mean": float(
                        table5_summary.loc[
                            (table5_summary["model_id"] == model_id)
                            & (table5_summary["strategy"] == strategy)
                            & (table5_summary["task"] == task),
                            f"{metric}_mean",
                        ].iloc[0]
                    ),
                    "sd": float(
                        table5_summary.loc[
                            (table5_summary["model_id"] == model_id)
                            & (table5_summary["strategy"] == strategy)
                            & (table5_summary["task"] == task),
                            f"{metric}_sd",
                        ].iloc[0]
                    ),
                }
                for model_id, strategy in winner_keys
            ]

    checks = {
        "protocol_id_exact": PROTOCOL_ID == "cv3_independent_seed_oof_v1",
        "systems_63": len(systems) == 63,
        "source_npz_189": len(source_identities) == 189,
        "seed_oof_metrics_315": len(seed_metrics) == 315,
        "seed_fold_metrics_945": len(seed_fold_metrics) == 945,
        "prediction_rows_709380": validation_counts["prediction_rows"] == 709380,
        "per_class_seed_rows_5250": len(per_class) == 5250,
        "per_class_summary_rows_1050": len(per_class_summary) == 1050,
        "probability_npz_63": len(list(probability_dir.glob("*.npz"))) == 63,
        "five_seeds_each_system": bool(
            (seed_metrics.groupby(["model_id", "strategy", "task"])["seed"].nunique() == 5).all()
        ),
        "complete_roi_once_per_seed_system": validation_counts["prediction_rows"] == 63 * 5 * 2252,
        "probability_ensemble_disabled": True,
        "table5_source_is_315_seed_oof_rows": len(seed_metrics) == 315,
        "table5_ddof_1": bool((table5_summary["ddof"] == 1).all()),
        "retrieval_hashes_unchanged": retrieval_passed,
        "legacy_primary_filenames_absent": not any(
            (output / name).exists()
            for name in (
                "classification_fold_metrics.csv",
                "classification_pooled_metrics.csv",
                "classification_oof_predictions.csv",
                "classification_paired_comparisons_holm.csv",
            )
        ),
    }
    validation_passed = all(checks.values())
    validation = {
        "schema": "pairbst.seven_model_validation.v2",
        "protocol_id": PROTOCOL_ID,
        "estimator_id": ESTIMATOR_ID,
        "generated_utc": generated_utc,
        "status": "PASS" if validation_passed else "FAIL",
        "checks": checks,
        "counts": {
            "systems": len(systems),
            "source_probability_npz": len(source_identities),
            "seed_complete_oof_metrics": len(seed_metrics),
            "seed_fold_audit_metrics": len(seed_fold_metrics),
            "seed_oof_predictions": validation_counts["prediction_rows"],
            "per_class_seed_rows": len(per_class),
            "per_class_seed_summary_rows": len(per_class_summary),
            "patient_cluster_ci_by_seed_rows": len(ci_frame),
        },
        "best_systems": best_records,
        "table5_change_summary": change_summary,
        "remaining_external_dependency": (
            "Future validation against a globally unique patient identifier supplied by the data team"
        ),
    }
    _write_json(output / "SEVEN_MODEL_VALIDATION.json", validation)
    provenance = {
        "schema": "pairbst.path_a_reconstruction_provenance.v1",
        "protocol_id": PROTOCOL_ID,
        "estimator_id": ESTIMATOR_ID,
        "generated_utc": generated_utc,
        "script_identity": _identity(script_path, relative_to=workspace),
        "config_identity": _identity(config_path, relative_to=workspace),
        "fold_manifest_identity": _identity(folds_path, relative_to=workspace),
        "class_order_sha256": class_order_hash,
        "metadata_sha256": _metadata_hash(folds),
        "source_npz_count": len(source_identities),
        "source_npz_hashes": source_hashes,
        "repository_state": _git_state(repository),
        "execution": {
            "path": "A",
            "trained": False,
            "features_extracted": False,
            "retrieval_executed": False,
            "seed_probabilities_averaged": False,
            "seed_predictions_voted": False,
            "primary_metric_unit": "complete_oof_per_seed",
            "aggregation": "arithmetic_mean_and_sample_sd_across_five_seed_metrics",
            "ddof": 1,
        },
    }
    _write_json(output / "provenance.json", provenance)

    final_results_manifest = {
        "schema": "pairbst.final_results_manifest.v2",
        "protocol_id": PROTOCOL_ID,
        "estimator_id": ESTIMATOR_ID,
        "generated_utc": generated_utc,
        "classification_manifest_identity": _identity(
            output / "classification_manifest.json", relative_to=output
        ),
        "statistics_manifest_identity": _identity(
            output / "statistics_manifest.json", relative_to=output
        ),
        "retrieval_verification_identity": _identity(
            output / "RETRIEVAL_UNCHANGED_VERIFICATION.json", relative_to=output
        ),
        "validation_identity": _identity(output / "SEVEN_MODEL_VALIDATION.json", relative_to=output),
        "canonical_table5_source": "classification_seed_oof_metrics.csv",
        "canonical_table5_outputs": [
            "table5_manuscript.csv",
            "table5_manuscript.md",
            "table5_manuscript.tex",
        ],
        "legacy_comparison_only": "legacy_comparison",
        "retrieval_unchanged": True,
        "output_identities": _collect_identities(
            output,
            exclude_names=(
                "final_results_manifest.json",
                "INDEPENDENT_VALIDATION.json",
                "reconstruction_stdout.log",
                "reconstruction_stderr.log",
            ),
        ),
    }
    _write_json(output / "final_results_manifest.json", final_results_manifest)

    report = f"""# Path-A independent-seed OOF reconstruction execution report

- Status: {'PASS' if validation_passed else 'FAIL'}
- Protocol: `{PROTOCOL_ID}`
- Estimator: `{ESTIMATOR_ID}`
- Systems: {len(systems)}
- Complete seed-specific OOF metrics: {len(seed_metrics)}
- Seed-and-fold audit metrics: {len(seed_fold_metrics)}
- Seed-specific OOF prediction rows: {validation_counts['prediction_rows']:,}
- Per-class seed rows: {len(per_class):,}
- Per-class seed summary rows: {len(per_class_summary):,}
- Seed-specific patient-cluster CI rows: {len(ci_frame):,}
- Source probability NPZ files: {len(source_identities)}
- Table 5 displayed cells changed: {change_summary['displayed_table5_cells_changed']}/126
- Linear-head retraining: not performed
- Feature extraction: not performed
- Retrieval execution: not performed
- Retrieval unchanged verification: {'PASS' if retrieval_passed else 'FAIL'}

The only external dependency left open is future validation against a globally unique patient identifier supplied by the data team. The frozen fold manifest and labels were not changed.
"""
    _write_text(output / "PATH_A_RECONSTRUCTION_EXECUTION_REPORT.md", report)
    report_ko = f"""# Path-A 독립 seed OOF 재구성 실행 보고서

- 상태: {'PASS' if validation_passed else 'FAIL'}
- 프로토콜: `{PROTOCOL_ID}`
- 추정 방식: `{ESTIMATOR_ID}`
- 시스템 수: {len(systems)}
- 완전한 seed별 OOF metric 행: {len(seed_metrics)}
- seed 및 fold 감사 metric 행: {len(seed_fold_metrics)}
- seed별 OOF 예측 행: {validation_counts['prediction_rows']:,}
- class별 seed 행: {len(per_class):,}
- class별 seed 요약 행: {len(per_class_summary):,}
- seed별 patient-cluster CI 행: {len(ci_frame):,}
- 원본 probability NPZ 파일: {len(source_identities)}
- 변경된 Table 5 표시 cell: {change_summary['displayed_table5_cells_changed']}/126
- linear-head 재학습: 수행하지 않음
- feature 추출: 수행하지 않음
- retrieval 실행: 수행하지 않음
- retrieval 불변 검증: {'PASS' if retrieval_passed else 'FAIL'}

남아 있는 유일한 외부 의존 사항은 데이터 팀이 향후 제공할 전역 고유 patient ID에 대한 검증입니다. 기존 fold manifest와 label은 변경하지 않았습니다.
"""
    _write_text(output / "PATH_A_RECONSTRUCTION_EXECUTION_REPORT_KO.md", report_ko)
    if not validation_passed:
        raise RuntimeError("Independent-seed OOF validation failed")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
