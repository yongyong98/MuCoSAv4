#!/usr/bin/env python3
"""Fail-closed validation for canonical PAIR-BST seven-model results.

The default mode validates ``cv3_independent_seed_oof_v1``.  The former
probability-ensemble release can be checked only with the explicit
``--legacy`` flag and is never accepted as a canonical manuscript source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
V4_DIRECTORY_NAME = "PAIR_BST_7MODEL_revision_results_independent_seed_oof_20260816_v4"
PROTOCOL_ID = "cv3_independent_seed_oof_v1"
ESTIMATOR_ID = "complete_oof_per_seed_metric_mean_sd"
SEEDS = (101, 202, 303, 404, 505)
FOLDS = (0, 1, 2)
MODELS = (
    "resnet50_v2",
    "swin_t",
    "retccl",
    "uni",
    "uni2_h",
    "prov_gigapath",
    "virchow2",
)
STRATEGIES = ("center", "mean", "max")
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
EXPECTED_CLASS_COUNTS = {"diagnosis": 33, "differentiation": 11, "growth_pattern": 6}
PACKAGED_V4 = ROOT.parent if ROOT.name == "CODE" and (ROOT.parent / "classification_protocol.json").is_file() else None
DEFAULT_V4 = PACKAGED_V4 if PACKAGED_V4 is not None else WORKSPACE / V4_DIRECTORY_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_csv(path: Path, expected_rows: int) -> pd.DataFrame:
    if not path.is_file():
        raise AssertionError(f"Missing canonical file: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    if len(frame) != expected_rows:
        raise AssertionError(f"{path.name}: rows={len(frame)}, expected={expected_rows}")
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


def _verify_identity(identity: dict[str, Any], base: Path) -> None:
    relative = Path(str(identity["path"]))
    if relative.is_absolute():
        raise AssertionError(f"Portable manifest contains an absolute path: {relative}")
    path = (base / relative).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as error:
        raise AssertionError(f"Manifest path escapes its root: {relative}") from error
    if not path.is_file():
        raise AssertionError(f"Manifest-bound file is missing: {relative}")
    if int(identity["size_bytes"]) != path.stat().st_size:
        raise AssertionError(f"Manifest size mismatch: {relative}")
    if str(identity["sha256"]).upper() != _sha256(path):
        raise AssertionError(f"Manifest SHA-256 mismatch: {relative}")


def _validate_external_identity_record(identity: dict[str, Any]) -> Path:
    relative = Path(str(identity["path"]))
    if relative.is_absolute():
        raise AssertionError(f"Source identity must be workspace-relative: {relative}")
    if ".." in relative.parts:
        raise AssertionError(f"Source identity escapes its recorded root: {relative}")
    digest = str(identity.get("sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789ABCDEF" for character in digest.upper()):
        raise AssertionError(f"Invalid recorded source SHA-256: {relative}")
    if int(identity.get("size_bytes", 0)) <= 0:
        raise AssertionError(f"Invalid recorded source size: {relative}")
    return relative


def _portable_repository_path(relative: Path, result: Path) -> Path:
    if relative.parts and relative.parts[0] == "pairbst_benchmark":
        return result / "CODE" / Path(*relative.parts[1:])
    return result / "CODE" / relative


def _verify_repository_identity(identity: dict[str, Any], result: Path, *, portable: bool) -> None:
    relative = _validate_external_identity_record(identity)
    path = _portable_repository_path(relative, result) if portable else WORKSPACE / relative
    if not path.is_file():
        raise AssertionError(f"Repository-bound file is missing: {relative}")
    if int(identity["size_bytes"]) != path.stat().st_size:
        raise AssertionError(f"Repository-bound file size mismatch: {relative}")
    if str(identity["sha256"]).upper() != _sha256(path):
        raise AssertionError(f"Repository-bound file SHA-256 mismatch: {relative}")


def _verify_source_identity(
    identity: dict[str, Any], result: Path, *, portable: bool
) -> bool:
    relative = _validate_external_identity_record(identity)
    if portable:
        candidate = _portable_repository_path(relative, result)
        if not candidate.is_file():
            return False
        path = candidate.resolve()
    else:
        path = (WORKSPACE / relative).resolve()
    if not portable:
        try:
            path.relative_to(WORKSPACE.resolve())
        except ValueError as error:
            raise AssertionError(f"Source identity escapes workspace: {relative}") from error
    if not path.is_file():
        raise AssertionError(f"Source artifact is missing: {relative}")
    if int(identity["size_bytes"]) != path.stat().st_size:
        raise AssertionError(f"Source artifact size mismatch: {relative}")
    if str(identity["sha256"]).upper() != _sha256(path):
        raise AssertionError(f"Source artifact SHA-256 mismatch: {relative}")
    return True


def _assert_exact_grid(frame: pd.DataFrame) -> None:
    observed = set(frame[["model_id", "strategy", "task"]].itertuples(index=False, name=None))
    expected = {
        (model_id, strategy, task)
        for model_id in MODELS
        for strategy in STRATEGIES
        for task in TASKS
    }
    if observed != expected:
        raise AssertionError("Canonical classification grid is not the exact 7 x 3 x 3 grid")


def _validate_table5(metrics: pd.DataFrame, result: Path) -> None:
    source = _read_csv(result / "table5_source_seed_oof_metrics.csv", 315)
    pd.testing.assert_frame_equal(
        metrics.reset_index(drop=True),
        source.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
        obj="Table 5 canonical source",
    )
    table = _read_csv(result / "table5_manuscript.csv", 21)
    if "±" not in (result / "table5_manuscript.csv").read_text(encoding="utf-8"):
        raise AssertionError("Table 5 CSV does not contain U+00B1")
    if "짹" in (result / "table5_manuscript.csv").read_text(encoding="utf-8"):
        raise AssertionError("Table 5 CSV contains a mojibake plus-minus marker")
    grouped = metrics.groupby(["model_id", "model", "strategy", "task"], sort=False)
    summary = grouped.agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", lambda values: values.std(ddof=1)),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", lambda values: values.std(ddof=1)),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    if not (summary["n_seeds"] == 5).all():
        raise AssertionError("Table 5 source does not contain five seeds per system")
    row_index = 0
    for model_id in MODELS:
        for strategy in STRATEGIES:
            table_row = table.iloc[row_index]
            expected_model = str(
                summary.loc[summary["model_id"] == model_id, "model"].iloc[0]
            )
            if table_row["Model"] != expected_model:
                raise AssertionError(f"Table 5 model order mismatch at row {row_index}")
            if table_row["Strategy"] != STRATEGY_LABELS[strategy]:
                raise AssertionError(f"Table 5 strategy order mismatch at row {row_index}")
            for task in TASKS:
                selected = summary.loc[
                    (summary["model_id"] == model_id)
                    & (summary["strategy"] == strategy)
                    & (summary["task"] == task)
                ]
                if len(selected) != 1:
                    raise AssertionError(f"Missing Table 5 source row for {model_id}/{strategy}/{task}")
                selected_row = selected.iloc[0]
                expected_ba = (
                    f"{selected_row['balanced_accuracy_mean']:.3f} ± "
                    f"{selected_row['balanced_accuracy_sd']:.3f}"
                )
                expected_f1 = (
                    f"{selected_row['macro_f1_mean']:.3f} ± "
                    f"{selected_row['macro_f1_sd']:.3f}"
                )
                if table_row[f"{TASK_LABELS[task]} B.Acc"] != expected_ba:
                    raise AssertionError(f"Table 5 balanced accuracy mismatch for {model_id}/{strategy}/{task}")
                if table_row[f"{TASK_LABELS[task]} Macro-F1"] != expected_f1:
                    raise AssertionError(f"Table 5 macro-F1 mismatch for {model_id}/{strategy}/{task}")
            row_index += 1


def _validate_probability_npz(result: Path) -> None:
    directory = result / "classification_seed_oof_probabilities"
    paths = sorted(directory.glob("*.npz"))
    if len(paths) != 63:
        raise AssertionError(f"Probability NPZ files={len(paths)}, expected=63")
    seen: set[tuple[str, str, str]] = set()
    required = {
        "protocol_id",
        "model_id",
        "strategy",
        "task",
        "seeds",
        "roi_uids",
        "wsi_uids",
        "patient_uids",
        "fold_ids",
        "true_labels",
        "class_names",
        "probabilities",
        "predictions",
        "probability_ensemble_across_seeds",
    }
    forbidden = {"mean_probabilities", "ensemble_probabilities", "ensemble_predictions"}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                raise AssertionError(f"Required keys missing from {path.name}")
            if forbidden.intersection(archive.files):
                raise AssertionError(f"Legacy ensemble arrays present in {path.name}")
            protocol = str(archive["protocol_id"].item())
            model_id = str(archive["model_id"].item())
            strategy = str(archive["strategy"].item())
            task = str(archive["task"].item())
            key = (model_id, strategy, task)
            seen.add(key)
            if protocol != PROTOCOL_ID or not np.array_equal(archive["seeds"], SEEDS):
                raise AssertionError(f"Protocol or seed order mismatch in {path.name}")
            expected_shape = (5, 2252, EXPECTED_CLASS_COUNTS[task])
            if archive["probabilities"].shape != expected_shape:
                raise AssertionError(f"Probability shape mismatch in {path.name}")
            if archive["predictions"].shape != (5, 2252):
                raise AssertionError(f"Prediction shape mismatch in {path.name}")
            if len(np.unique(archive["roi_uids"])) != 2252:
                raise AssertionError(f"ROI identifiers are missing or duplicated in {path.name}")
            if set(archive["fold_ids"].astype(int).tolist()) != set(FOLDS):
                raise AssertionError(f"Fold coverage mismatch in {path.name}")
            if bool(archive["probability_ensemble_across_seeds"].item()):
                raise AssertionError(f"Probability ensembling enabled in {path.name}")
    expected_grid = {
        (model_id, strategy, task)
        for model_id in MODELS
        for strategy in STRATEGIES
        for task in TASKS
    }
    if seen != expected_grid:
        raise AssertionError("Probability NPZ system grid is incomplete")


def _validate_retrieval(result: Path) -> None:
    verification = _read_json(result / "RETRIEVAL_UNCHANGED_VERIFICATION.json")
    if verification.get("result") != "PASS" or verification.get("retrieval_was_rerun") is not False:
        raise AssertionError("Retrieval unchanged verification is not PASS")
    if not verification.get("source_reference_only"):
        raise AssertionError("Legacy retrieval manifest source identity is not recorded")
    if (result / "retrieval/retrieval_manifest_LEGACY_SOURCE.json").exists():
        raise AssertionError("Machine-local legacy retrieval manifest was copied into v4")
    for item in verification.get("files", []):
        destination = result / str(item["destination_path"])
        if not destination.is_file():
            raise AssertionError(f"Verified retrieval output is missing: {destination}")
        digest = _sha256(destination)
        if digest != str(item["destination_sha256"]).upper():
            raise AssertionError(f"Retrieval destination hash mismatch: {destination.name}")
        if digest != str(item["source_sha256"]).upper():
            raise AssertionError(f"Retrieval source/destination mismatch: {destination.name}")


def _validate_manifests(result: Path) -> str:
    portable = PACKAGED_V4 is not None and result.resolve() == PACKAGED_V4.resolve()
    protocol = _read_json(result / "classification_protocol.json")
    classification = _read_json(result / "classification_manifest.json")
    statistics = _read_json(result / "statistics_manifest.json")
    final = _read_json(result / "final_results_manifest.json")
    validation = _read_json(result / "SEVEN_MODEL_VALIDATION.json")
    provenance = _read_json(result / "provenance.json")
    for name, value in (
        ("classification_protocol", protocol),
        ("classification_manifest", classification),
        ("statistics_manifest", statistics),
        ("final_results_manifest", final),
        ("SEVEN_MODEL_VALIDATION", validation),
    ):
        if value.get("protocol_id") != PROTOCOL_ID:
            raise AssertionError(f"Cross-protocol mixing detected in {name}")
    for name, value in (
        ("classification_manifest", classification),
        ("statistics_manifest", statistics),
        ("final_results_manifest", final),
        ("SEVEN_MODEL_VALIDATION", validation),
    ):
        if value.get("estimator_id") != ESTIMATOR_ID:
            raise AssertionError(f"Cross-estimator mixing detected in {name}")
    if classification.get("action") != "classify.run" or classification.get("profile") != "official_model_specific":
        raise AssertionError("Classification manifest action/profile mismatch")
    if classification.get("probability_ensemble_across_seeds") is not False:
        raise AssertionError("Classification manifest does not disable probability ensembling")
    if classification.get("seed_sd_ddof") != 1:
        raise AssertionError("Classification manifest does not specify ddof=1")
    if len(classification.get("source_npz_identities", [])) != 189:
        raise AssertionError("Classification manifest does not bind 189 source NPZ files")
    source_records = classification["source_npz_identities"]
    if len({str(identity["path"]) for identity in source_records}) != 189:
        raise AssertionError("Source NPZ identity paths are not unique")
    source_files_verified = sum(
        _verify_source_identity(identity, result, portable=portable)
        for identity in source_records
    )
    if portable:
        if source_files_verified not in (0, 189):
            raise AssertionError("Portable source NPZ availability is partial")
        source_mode = (
            "portable_external_hash_ledger"
            if source_files_verified == 0
            else "portable_source_files_verified"
        )
    else:
        if source_files_verified != 189:
            raise AssertionError("Live source NPZ verification is incomplete")
        source_mode = "live_source_files_verified"
    _verify_repository_identity(
        classification["config_identity"], result, portable=portable
    )
    _verify_repository_identity(
        classification["fold_manifest_identity"], result, portable=portable
    )
    if provenance.get("protocol_id") != PROTOCOL_ID or provenance.get("estimator_id") != ESTIMATOR_ID:
        raise AssertionError("Provenance protocol or estimator mismatch")
    _verify_repository_identity(provenance["script_identity"], result, portable=portable)
    for identity in classification.get("output_identities", []):
        _verify_identity(identity, result)
    for identity in statistics.get("output_identities", []):
        _verify_identity(identity, result)
    for identity in final.get("output_identities", []):
        _verify_identity(identity, result)
    if validation.get("status") != "PASS" or not all(validation.get("checks", {}).values()):
        raise AssertionError("Embedded seven-model validation is not PASS")
    return source_mode


def _validate_canonical(result: Path) -> dict[str, Any]:
    result = result.resolve()
    if not result.is_dir():
        raise AssertionError(f"Canonical result directory does not exist: {result}")
    legacy_names = (
        "classification_fold_metrics.csv",
        "classification_seed_metrics.csv",
        "classification_pooled_metrics.csv",
        "classification_oof_predictions.csv",
        "classification_per_class_metrics.csv",
        "classification_paired_comparisons_holm.csv",
    )
    mixed = [name for name in legacy_names if (result / name).exists()]
    if mixed:
        raise AssertionError(f"Legacy primary files mixed into canonical root: {mixed}")

    metrics = _read_csv(result / "classification_seed_oof_metrics.csv", 315)
    fold_metrics = _read_csv(result / "classification_seed_fold_metrics.csv", 945)
    predictions = _read_csv(result / "classification_seed_oof_predictions.csv.gz", 709380)
    per_class = _read_csv(result / "classification_per_class_seed_oof.csv", 5250)
    per_class_summary = _read_csv(result / "classification_per_class_seed_summary.csv", 1050)
    ci = _read_csv(result / "classification_patient_cluster_ci_by_seed.csv", 1260)
    _assert_exact_grid(metrics)
    metric_groups = metrics.groupby(["model_id", "strategy", "task"], sort=False)
    if len(metric_groups) != 63:
        raise AssertionError("Seed metric file does not contain 63 systems")
    if not (metric_groups.size() == 5).all():
        raise AssertionError("Every system must have exactly five complete OOF metrics")
    if not metric_groups["seed"].agg(lambda values: set(values.astype(int)) == set(SEEDS)).all():
        raise AssertionError("A system has an incomplete or unexpected seed set")
    fold_groups = fold_metrics.groupby(["model_id", "strategy", "task", "seed"], sort=False)
    if len(fold_groups) != 315 or not (fold_groups.size() == 3).all():
        raise AssertionError("Seed-and-fold audit metric grouping is incomplete")
    if not fold_groups["held_fold"].agg(lambda values: set(values.astype(int)) == set(FOLDS)).all():
        raise AssertionError("A seed-and-system group does not cover the three held-out folds")
    prediction_groups = predictions.groupby(
        ["model_id", "strategy", "task", "seed"], sort=False
    )["roi_uid"].agg(["size", "nunique"])
    if len(prediction_groups) != 315:
        raise AssertionError("Prediction file does not contain 315 seed-system groups")
    if not ((prediction_groups["size"] == 2252) & (prediction_groups["nunique"] == 2252)).all():
        raise AssertionError("An OOF vector has a missing or duplicate ROI")
    if set(predictions["fold"].astype(int)) != set(FOLDS):
        raise AssertionError("Prediction file does not cover all three held-out folds")
    if set(predictions["protocol_id"]) != {PROTOCOL_ID}:
        raise AssertionError("Prediction file contains a different protocol")
    if set(metrics["protocol_id"]) != {PROTOCOL_ID}:
        raise AssertionError("Metric file contains a different protocol")
    if per_class.groupby(["model_id", "strategy", "task", "seed"]).ngroups != 315:
        raise AssertionError("Per-class metrics were not calculated independently by seed")
    if not (per_class_summary["n_seeds"].astype(int) == 5).all():
        raise AssertionError("Per-class summaries do not use five independent seeds")
    if len(ci.groupby(["model_id", "strategy", "task", "seed", "metric"])) != 1260:
        raise AssertionError("Seed-specific patient-cluster CI grouping is incomplete")
    if not (ci["n_bootstrap"].astype(int) == 10000).all():
        raise AssertionError("Patient-cluster CI does not use the canonical 10,000 replicates")

    _validate_probability_npz(result)
    _validate_table5(metrics, result)
    raw_confusion = _read_csv(
        result / "confusion_matrices/confusion_matrices_seed_raw_long.csv.gz", 130830
    )
    normalized_confusion = _read_csv(
        result / "confusion_matrices/confusion_matrices_seed_row_normalized_long.csv.gz",
        130830,
    )
    _read_csv(
        result / "confusion_matrices/confusion_matrices_mean_row_normalized.csv.gz", 26166
    )
    _read_csv(
        result / "confusion_matrices/confusion_matrices_sd_row_normalized.csv.gz", 26166
    )
    if set(raw_confusion["seed"].astype(int)) != set(SEEDS):
        raise AssertionError("Raw confusion matrices do not retain five seeds")
    if set(normalized_confusion["seed"].astype(int)) != set(SEEDS):
        raise AssertionError("Normalized confusion matrices do not retain five seeds")
    _validate_retrieval(result)
    source_identity_mode = _validate_manifests(result)

    return {
        "schema": "pairbst.independent_validation.v1",
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "estimator_id": ESTIMATOR_ID,
        "result_directory": result.name,
        "source_identity_mode": source_identity_mode,
        "counts": {
            "systems": 63,
            "seed_oof_metrics": len(metrics),
            "seed_fold_metrics": len(fold_metrics),
            "seed_oof_predictions": len(predictions),
            "per_class_seed_rows": len(per_class),
            "per_class_seed_summary_rows": len(per_class_summary),
            "patient_cluster_ci_by_seed_rows": len(ci),
            "probability_npz": 63,
        },
        "checks": {
            "no_probability_ensemble": "PASS",
            "complete_oof_once_per_seed": "PASS",
            "table5_recalculated_from_315_rows": "PASS",
            "table5_sample_sd_ddof_1": "PASS",
            "per_class_independent_by_seed": "PASS",
            "confusion_independent_by_seed": "PASS",
            "retrieval_hashes_unchanged": "PASS",
            "manifest_protocol_mixing_rejected": "PASS",
            "source_npz_provenance_verified": "PASS",
            "legacy_primary_files_absent": "PASS",
        },
    }


def _validate_legacy() -> dict[str, Any]:
    """Retain the old release check behind an explicit noncanonical flag."""

    tag = "official_model_specific_7model_v1"
    classification = ROOT / "outputs/runs/classification" / tag
    retrieval = ROOT / "outputs/runs/retrieval" / tag
    statistics = ROOT / "outputs/runs/statistics" / tag
    final = ROOT / "outputs/final_7model_v1"
    expected = {
        classification / "classification_fold_metrics.csv": 189,
        classification / "classification_seed_metrics.csv": 945,
        classification / "classification_pooled_metrics.csv": 63,
        classification / "classification_per_class_metrics.csv": 1050,
        classification / "classification_oof_predictions.csv": 141876,
        retrieval / "retrieval_fold_metrics.csv": 756,
        retrieval / "retrieval_pooled_metrics.csv": 252,
        retrieval / "retrieval_per_query_metrics.csv": 567504,
        statistics / "classification_patient_cluster_ci.csv": 252,
        statistics / "classification_paired_comparisons_holm.csv": 192,
        final / "table5_manuscript.csv": 21,
    }
    for path, rows in expected.items():
        _read_csv(path, rows)
    return {
        "status": "PASS",
        "mode": "LEGACY_EXPLICIT_ONLY",
        "canonical_manuscript_source": False,
        "estimator_id": "seed_probability_ensemble_fold_mean_sd",
        "row_counts": {str(path.relative_to(ROOT)): rows for path, rows in expected.items()},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_V4,
        help="Canonical v4 directory. Ignored with --legacy.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Validate the noncanonical probability-ensemble legacy release explicitly.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.legacy:
        summary = _validate_legacy()
    else:
        summary = _validate_canonical(args.result_dir)
        if PACKAGED_V4 is None:
            destination = args.result_dir.resolve() / "INDEPENDENT_VALIDATION.json"
            destination.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
