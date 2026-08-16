"""Preparation audit for datasets, checkpoints, folds, and recovery inputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import h5py
import numpy as np
from PIL import Image

from .config import get_path
from .datasets import load_roi_records, records_fingerprint
from .provenance import utc_now, write_json_atomic
from .qa import PAIRBST_EXPECTED_FOLD_COUNTS, PAIRBST_EXPECTED_FOLDS, audit_cv_split


@dataclass(frozen=True)
class AuditCheck:
    name: str
    status: str
    detail: str
    blocking: bool = True


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _locked_artifact_check(
    name: str,
    artifact_path: str | Path,
    lock: Mapping[str, Any],
    lock_key: str,
) -> AuditCheck:
    """Verify an artifact against a required SHA-256 entry in a lock."""

    path = Path(artifact_path).resolve()
    expected = str(lock.get(lock_key, "")).strip().upper()
    if not path.is_file():
        return AuditCheck(name, "FAIL", f"missing locked artifact: {path}")
    if not expected:
        return AuditCheck(name, "FAIL", f"lock is missing required hash {lock_key!r}: {path}")
    observed = _sha256(path)
    if observed != expected:
        return AuditCheck(name, "FAIL", f"SHA-256 {observed} != {expected}: {path}")
    return AuditCheck(name, "PASS", f"SHA-256 verified: {path}")


def _all_pass(checks: Iterable[AuditCheck]) -> bool:
    return all(check.status == "PASS" for check in checks)


def _check_file(
    name: str,
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    hash_contents: bool = False,
    blocking: bool = True,
) -> AuditCheck:
    if not path.is_file():
        return AuditCheck(name, "FAIL", f"missing: {path}", blocking)
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        return AuditCheck(
            name,
            "FAIL",
            f"size {path.stat().st_size:,} != expected {int(expected_bytes):,}: {path}",
            blocking,
        )
    if hash_contents and expected_sha256:
        observed = _sha256(path)
        if observed != expected_sha256.upper():
            return AuditCheck(name, "FAIL", f"SHA-256 {observed} != {expected_sha256}: {path}", blocking)
        return AuditCheck(name, "PASS", f"size and SHA-256 verified: {path}", blocking)
    if expected_sha256:
        return AuditCheck(
            name,
            "DEFERRED",
            f"present with expected size, but SHA-256 was not calculated: {path}",
            blocking,
        )
    return AuditCheck(name, "PASS", f"present with expected size: {path}", blocking)


def audit_dataset(
    paths_config: Mapping[str, Any],
    dataset_lock: Mapping[str, Any],
    *,
    decode_all: bool = False,
    image_qa_path: str | Path | None = None,
    image_qa_lock_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
    dataset_manifest_lock_path: str | Path | None = None,
) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    released: pd.DataFrame | None = None
    metadata_path = get_path(paths_config, "dataset", "metadata_csv")
    required_manifest_value = paths_config.get("dataset", {}).get("required_manifest")
    if required_manifest_value is not None or dataset_lock.get("required_manifest_sha256"):
        if required_manifest_value is None:
            checks.append(
                AuditCheck(
                    "dataset.required_manifest",
                    "FAIL",
                    "dataset.required_manifest path is missing from configuration",
                )
            )
        else:
            required_manifest_path = get_path(paths_config, "dataset", "required_manifest")
            checks.append(
                _check_file(
                    "dataset.required_manifest",
                    required_manifest_path,
                    expected_sha256=dataset_lock.get("required_manifest_sha256"),
                    hash_contents=True,
                )
            )
            if required_manifest_path.is_file():
                released = pd.read_csv(
                    required_manifest_path,
                    sep="\t",
                    header=None,
                    names=("article_id", "file_id", "filename", "bytes", "md5", "url"),
                    dtype={"filename": "string"},
                )
                expected_manifest_rows = int(dataset_lock.get("required_manifest_rows", -1))
                row_count_ok = len(released) == expected_manifest_rows
                checks.append(
                    AuditCheck(
                        "dataset.required_manifest_rows",
                        "PASS" if row_count_ok else "FAIL",
                        f"rows={len(released)}, expected={expected_manifest_rows}",
                    )
                )
    checks.append(
        _check_file(
            "dataset.metadata",
            metadata_path,
            expected_sha256=dataset_lock.get("metadata_sha256"),
            hash_contents=True,
        )
    )
    manifest_lock: dict[str, Any] | None = None
    manifest_hash_ok = False
    if (
        dataset_manifest_path is not None
        and dataset_manifest_lock_path is not None
        and Path(dataset_manifest_lock_path).is_file()
    ):
        manifest_lock = _load_json(dataset_manifest_lock_path)
        manifest_check = _check_file(
            "dataset.manifest_integrity",
            Path(dataset_manifest_path).resolve(),
            expected_sha256=manifest_lock.get("manifest_sha256"),
            hash_contents=True,
        )
        checks.append(manifest_check)
        expected_metadata_hash = str(dataset_lock.get("metadata_sha256", "")).upper()
        locked_metadata_hash = str(manifest_lock.get("metadata_sha256", "")).upper()
        binding_ok = bool(expected_metadata_hash) and locked_metadata_hash == expected_metadata_hash
        checks.append(
            AuditCheck(
                "dataset.manifest_metadata_binding",
                "PASS" if binding_ok else "FAIL",
                (
                    f"manifest metadata SHA-256={locked_metadata_hash}"
                    if binding_ok
                    else f"manifest metadata SHA-256 {locked_metadata_hash or '<missing>'} "
                    f"!= dataset lock {expected_metadata_hash or '<missing>'}"
                ),
            )
        )
        manifest_hash_ok = manifest_check.status == "PASS" and binding_ok
    else:
        checks.append(
            AuditCheck(
                "dataset.manifest_integrity",
                "DEFERRED",
                "dataset manifest and its lock were not both supplied",
            )
        )
    if not metadata_path.is_file():
        return checks
    metadata = pd.read_csv(metadata_path)
    expected_rows = int(dataset_lock["metadata_rows"])
    checks.append(
        AuditCheck(
            "dataset.metadata_rows",
            "PASS" if len(metadata) == expected_rows else "FAIL",
            f"rows={len(metadata):,}, expected={expected_rows:,}",
        )
    )
    required_columns = {
        "slide_name",
        "patient_idx",
        "roi_idx",
        "diagnosis",
        "differentiation",
        "growth_pattern",
    }
    missing = sorted(required_columns.difference(metadata.columns))
    checks.append(
        AuditCheck(
            "dataset.columns",
            "PASS" if not missing else "FAIL",
            "required columns present" if not missing else f"missing columns: {missing}",
        )
    )
    if not missing:
        patient_count = metadata[["diagnosis", "patient_idx"]].drop_duplicates().shape[0]
        slide_count = metadata["slide_name"].str.replace(r"\.svs_roi_\d+\.png$", ".svs", regex=True).nunique()
        counts = {
            "diagnosis": metadata["diagnosis"].nunique(),
            "differentiation": metadata["differentiation"].nunique(),
            "growth_pattern": metadata["growth_pattern"].nunique(),
        }
        expected_classes = {key: int(value) for key, value in dataset_lock["class_counts"].items()}
        identity_ok = (
            patient_count == int(dataset_lock["expected_patient_count"])
            and slide_count == int(dataset_lock["expected_slide_count"])
            and counts == expected_classes
        )
        checks.append(
            AuditCheck(
                "dataset.identity_and_labels",
                "PASS" if identity_ok else "FAIL",
                f"patients={patient_count}, slides={slide_count}, classes={counts}",
            )
        )
    roi_root = get_path(paths_config, "dataset", "roi_root")
    roi_files = sorted(roi_root.glob("part_*/*.png")) if roi_root.is_dir() else []
    expected_roi_count = int(dataset_lock["roi_png_count"])
    checks.append(
        AuditCheck(
            "dataset.roi_count",
            "PASS" if len(roi_files) == expected_roi_count else "FAIL",
            f"PNG files={len(roi_files):,}, expected={expected_roi_count:,}: {roi_root}",
        )
    )
    if required_manifest_value is not None and released is not None:
        expected_sizes = {
            str(row.filename): int(row.bytes)
            for row in released.itertuples(index=False)
        }
        local_sizes = {path.name: path.stat().st_size for path in roi_files}
        if metadata_path.is_file():
            local_sizes[metadata_path.name] = metadata_path.stat().st_size
        missing_names = sorted(set(expected_sizes) - set(local_sizes))
        extra_names = sorted(set(local_sizes) - set(expected_sizes))
        wrong_sizes = sorted(
            name
            for name in set(expected_sizes).intersection(local_sizes)
            if expected_sizes[name] != local_sizes[name]
        )
        size_set_ok = not missing_names and not extra_names and not wrong_sizes
        checks.append(
            AuditCheck(
                "dataset.required_file_sizes",
                "PASS" if size_set_ok else "FAIL",
                "all released filenames and byte sizes match the frozen Figshare manifest"
                if size_set_ok
                else (
                    f"missing={missing_names[:5]}, extra={extra_names[:5]}, "
                    f"wrong_size={wrong_sizes[:5]}"
                ),
            )
        )
    image_qa: dict[str, Any] | None = None
    image_qa_integrity_ok = False
    image_qa_source_ok = False
    if image_qa_path is not None and Path(image_qa_path).is_file():
        image_qa = _load_json(image_qa_path)
        if image_qa_lock_path is not None and Path(image_qa_lock_path).is_file():
            image_lock = _load_json(image_qa_lock_path)
            artifact_checks = [
                _locked_artifact_check(
                    "dataset.image_qa_summary_integrity",
                    image_qa_path,
                    image_lock,
                    "image_center_summary_sha256",
                ),
                _locked_artifact_check(
                    "dataset.image_qa_rows_integrity",
                    Path(image_qa_path).parent / "image_center_checks.csv",
                    image_lock,
                    "image_center_checks_sha256",
                ),
            ]
            checks.extend(artifact_checks)
            image_qa_integrity_ok = _all_pass(artifact_checks)

            locked_source_hash = str(image_lock.get("dataset_manifest_sha256", "")).upper()
            actual_source_hash = (
                _sha256(Path(dataset_manifest_path))
                if dataset_manifest_path is not None and Path(dataset_manifest_path).is_file()
                else ""
            )
            primary_locked_hash = (
                str(manifest_lock.get("manifest_sha256", "")).upper()
                if manifest_lock is not None
                else ""
            )
            image_qa_source_ok = (
                manifest_hash_ok
                and bool(locked_source_hash)
                and locked_source_hash == actual_source_hash == primary_locked_hash
            )
            checks.append(
                AuditCheck(
                    "dataset.image_qa_source_manifest",
                    "PASS" if image_qa_source_ok else "FAIL",
                    (
                        f"image QA is bound to manifest SHA-256 {actual_source_hash}"
                        if image_qa_source_ok
                        else "image QA manifest binding mismatch: "
                        f"qa_lock={locked_source_hash or '<missing>'}, "
                        f"actual={actual_source_hash or '<missing>'}, "
                        f"manifest_lock={primary_locked_hash or '<missing>'}"
                    ),
                )
            )
        else:
            checks.append(
                AuditCheck(
                    "dataset.image_qa_summary_integrity",
                    "DEFERRED",
                    "frozen image QA lock was not supplied",
                )
            )

    if decode_all and len(roi_files) == expected_roi_count:
        expected_width, expected_height, channels = dataset_lock["roi_shape"]
        bad: list[str] = []
        for path in roi_files:
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.size != (expected_width, expected_height) or image.mode != "RGB" or channels != 3:
                        bad.append(str(path))
            except Exception:
                bad.append(str(path))
            if len(bad) >= 20:
                break
        checks.append(
            AuditCheck(
                "dataset.full_decode",
                "PASS" if not bad else "FAIL",
                "all ROI PNGs fully decoded as RGB 4096x4096" if not bad else f"invalid examples: {bad}",
            )
        )
    elif image_qa is not None:
        image_pass = (
            image_qa_integrity_ok
            and image_qa_source_ok
            and int(image_qa.get("roi_checked", -1)) == expected_roi_count
            and int(image_qa.get("fully_decoded", -1)) == expected_roi_count
            and int(image_qa.get("rgb_4096", -1)) == expected_roi_count
        )
        checks.append(
            AuditCheck(
                "dataset.full_decode",
                "PASS" if image_pass else "FAIL",
                f"frozen image QA: {Path(image_qa_path).resolve()}",
            )
        )
    else:
        checks.append(
            AuditCheck(
                "dataset.full_decode",
                "DEFERRED",
                "use --decode-all before the approved full experiment",
                True,
            )
        )
    if image_qa is None:
        checks.append(
            AuditCheck(
                "dataset.center_equivalence",
                "DEFERRED",
                "frozen center-comparison QA was not supplied",
            )
        )
    else:
        comparison_performed = image_qa.get("comparison_performed") is True
        center_pass = (
            image_qa_integrity_ok
            and image_qa_source_ok
            and comparison_performed
            and image_qa.get("center_equivalence_status") == "PASS"
            and int(image_qa.get("legacy_centers_found", -1)) == expected_roi_count
            and int(image_qa.get("legacy_centers_pixel_exact", -1)) == expected_roi_count
        )
        if not comparison_performed:
            center_status = "DEFERRED"
            center_detail = "legacy/public center comparison was not performed"
        else:
            center_status = "PASS" if center_pass else "FAIL"
            center_detail = (
                "pixel-exact legacy/public center checks="
                f"{image_qa.get('legacy_centers_pixel_exact')}; "
                f"QA integrity={image_qa_integrity_ok}; source binding={image_qa_source_ok}"
            )
        checks.append(
            AuditCheck(
                "dataset.center_equivalence",
                center_status,
                center_detail,
            )
        )
    return checks


MODEL_LOCK_TO_REGISTRY = {
    "resnet50_in1k_v2": "resnet50_v2",
    "swin_t_in1k_v1": "swin_t",
    "retccl_resnet50": "retccl",
    "uni_vitl16": "uni",
    "uni2_h": "uni2_h",
    "prov_gigapath": "prov_gigapath",
    "virchow2": "virchow2",
}


def audit_models(
    paths_config: Mapping[str, Any],
    model_lock: Mapping[str, Any],
    *,
    hash_contents: bool = False,
) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    for model_id, expected in model_lock["models"].items():
        if model_id not in MODEL_LOCK_TO_REGISTRY:
            checks.append(
                AuditCheck(
                    f"model.{model_id}",
                    "FAIL",
                    "model lock ID has no registry/path mapping",
                )
            )
            continue
        checkpoint = get_path(paths_config, "weights", MODEL_LOCK_TO_REGISTRY[model_id])
        expected_hash = expected.get("expected_sha256")
        if expected_hash is None:
            if checkpoint.is_file():
                checks.append(
                    AuditCheck(
                        f"model.{model_id}",
                        "FAIL",
                        f"checkpoint exists but its SHA-256 has not been frozen: {checkpoint}",
                    )
                )
            else:
                checks.append(AuditCheck(f"model.{model_id}", "FAIL", f"checkpoint not acquired: {checkpoint}"))
            continue
        checks.append(
            _check_file(
                f"model.{model_id}",
                checkpoint,
                expected_bytes=expected.get("expected_bytes"),
                expected_sha256=expected_hash,
                hash_contents=hash_contents,
            )
        )
        expected_config_hash = expected.get("upstream_config_sha256")
        if expected_config_hash is not None:
            checks.append(
                _check_file(
                    f"model.{model_id}.upstream_config",
                    checkpoint.parent / "config.json",
                    expected_sha256=str(expected_config_hash),
                    hash_contents=True,
                )
            )
        if model_id == "retccl_resnet50":
            expected_source_hash = expected.get("architecture_source_sha256")
            configured_source = paths_config.get("weights", {}).get(
                "retccl_architecture_source"
            )
            source_path = (
                get_path(paths_config, "weights", "retccl_architecture_source")
                if configured_source is not None
                else get_path(paths_config, "weights", "root") / "RetCCL" / "ResNet.py"
            )
            if not expected_source_hash:
                checks.append(
                    AuditCheck(
                        "model.retccl_resnet50.architecture_source",
                        "FAIL",
                        "RetCCL architecture_source_sha256 is not frozen in the model lock",
                    )
                )
            else:
                checks.append(
                    _check_file(
                        "model.retccl_resnet50.architecture_source",
                        source_path,
                        expected_sha256=expected_source_hash,
                        hash_contents=hash_contents,
                    )
                )
    return checks


def audit_recovery_inputs(
    paths_config: Mapping[str, Any],
    recovery_lock: Mapping[str, Any] | None = None,
) -> list[AuditCheck]:
    items = (
        ("recovery.root", get_path(paths_config, "recovery", "root"), "directory"),
        ("recovery.historical_split", get_path(paths_config, "recovery", "historical_split_csv"), "file"),
        ("recovery.reference_root", get_path(paths_config, "recovery", "reference_root"), "directory"),
        ("recovery.handoff_zip", get_path(paths_config, "recovery", "handoff_zip"), "file"),
        ("recovery.uni2_grid_zip", get_path(paths_config, "recovery", "uni2_grid_zip"), "file"),
    )
    checks: list[AuditCheck] = []
    for name, path, kind in items:
        exists = path.is_dir() if kind == "directory" else path.is_file()
        checks.append(AuditCheck(name, "PASS" if exists else "FAIL", f"{kind}: {path}"))
    if recovery_lock is not None:
        genuine = recovery_lock.get("genuine_uni2_h_grid", {})
        for name, config_key, hash_key in (
            ("recovery.genuine_uni2_grid_mean", "genuine_uni2_grid_mean", "mean_h5_sha256"),
            ("recovery.genuine_uni2_grid_max", "genuine_uni2_grid_max", "max_h5_sha256"),
        ):
            checks.append(
                _check_file(
                    name,
                    get_path(paths_config, "recovery", config_key),
                    expected_sha256=genuine.get(hash_key),
                    hash_contents=True,
                )
            )
    return checks


def _expected_pairbst_fold_rows() -> list[dict[str, int]]:
    return [
        {
            "fold": int(fold),
            "patients": int(PAIRBST_EXPECTED_FOLD_COUNTS[fold]["patients"]),
            "wsi": int(PAIRBST_EXPECTED_FOLD_COUNTS[fold]["wsi"]),
            "roi": int(PAIRBST_EXPECTED_FOLD_COUNTS[fold]["roi"]),
        }
        for fold in PAIRBST_EXPECTED_FOLDS
    ]


def _normalize_qa_fold_rows(value: Any) -> list[dict[str, int]] | None:
    if not isinstance(value, list):
        return None
    rows: list[dict[str, int]] = []
    try:
        for row in value:
            rows.append(
                {
                    "fold": int(row["fold"]),
                    "patients": int(row["patients"]),
                    "wsi": int(row["wsi"]),
                    "roi": int(row["roi"]),
                }
            )
    except (KeyError, TypeError, ValueError):
        return None
    return sorted(rows, key=lambda row: row["fold"])


def _normalize_split_lock_fold_rows(value: Any) -> list[dict[str, int]] | None:
    if not isinstance(value, list):
        return None
    rows: list[dict[str, int]] = []
    try:
        for row in value:
            rows.append(
                {
                    "fold": int(row["fold"]),
                    "patients": int(row["n_patient"]),
                    "wsi": int(row["n_wsi"]),
                    "roi": int(row["n_roi"]),
                }
            )
    except (KeyError, TypeError, ValueError):
        return None
    return sorted(rows, key=lambda row: row["fold"])


def audit_cv_split_artifacts(
    *,
    split_csv_path: str | Path,
    split_lock_path: str | Path,
    split_qa_path: str | Path,
    split_qa_lock_path: str | Path,
    patient_coverage_path: str | Path | None = None,
    roi_coverage_path: str | Path | None = None,
) -> list[AuditCheck]:
    """Fail-closed audit of the frozen split, QA outputs, and their hash locks."""

    split_csv = Path(split_csv_path).resolve()
    split_lock_file = Path(split_lock_path).resolve()
    split_qa_file = Path(split_qa_path).resolve()
    split_qa_lock_file = Path(split_qa_lock_path).resolve()
    patient_coverage = (
        Path(patient_coverage_path).resolve()
        if patient_coverage_path is not None
        else split_qa_file.parent / "class_coverage_patients.csv"
    )
    roi_coverage = (
        Path(roi_coverage_path).resolve()
        if roi_coverage_path is not None
        else split_qa_file.parent / "class_coverage_rois.csv"
    )
    checks: list[AuditCheck] = []
    required = {
        "split CSV": split_csv,
        "split lock": split_lock_file,
        "split QA": split_qa_file,
        "split QA lock": split_qa_lock_file,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        return [AuditCheck("split.cv3_artifacts", "FAIL", "; ".join(missing))]

    split_lock = _load_json(split_lock_file)
    qa_lock = _load_json(split_qa_lock_file)
    split_qa = _load_json(split_qa_file)
    artifact_checks = [
        _locked_artifact_check(
            "split.qa_summary_integrity", split_qa_file, qa_lock, "split_qa_sha256"
        ),
        _locked_artifact_check(
            "split.patient_coverage_integrity",
            patient_coverage,
            qa_lock,
            "patient_coverage_sha256",
        ),
        _locked_artifact_check(
            "split.roi_coverage_integrity",
            roi_coverage,
            qa_lock,
            "roi_coverage_sha256",
        ),
    ]
    checks.extend(artifact_checks)

    qa_source_check = _locked_artifact_check(
        "split.qa_source_split", split_csv, qa_lock, "fold_manifest_sha256"
    )
    primary_source_check = _locked_artifact_check(
        "split.primary_lock_source", split_csv, split_lock, "roi_split_sha256"
    )
    checks.extend((qa_source_check, primary_source_check))
    qa_source_hash = str(qa_lock.get("fold_manifest_sha256", "")).upper()
    primary_source_hash = str(split_lock.get("roi_split_sha256", "")).upper()
    source_binding_ok = (
        qa_source_check.status == "PASS"
        and primary_source_check.status == "PASS"
        and bool(qa_source_hash)
        and qa_source_hash == primary_source_hash
    )
    checks.append(
        AuditCheck(
            "split.source_lock_binding",
            "PASS" if source_binding_ok else "FAIL",
            (
                f"QA and primary locks bind split SHA-256 {qa_source_hash}"
                if source_binding_ok
                else f"QA lock split hash {qa_source_hash or '<missing>'} != "
                f"primary lock {primary_source_hash or '<missing>'}"
            ),
        )
    )

    expected_rows = _expected_pairbst_fold_rows()
    qa_rows = _normalize_qa_fold_rows(split_qa.get("folds"))
    lock_rows = _normalize_split_lock_fold_rows(split_lock.get("folds"))
    declared_counts_ok = qa_rows == expected_rows and lock_rows == expected_rows
    checks.append(
        AuditCheck(
            "split.declared_fold_counts",
            "PASS" if declared_counts_ok else "FAIL",
            f"expected={expected_rows}; qa={qa_rows}; split_lock={lock_rows}",
        )
    )

    try:
        split_frame = pd.read_csv(split_csv)
        recomputed = audit_cv_split(split_frame)
        recomputed_rows = _normalize_qa_fold_rows(recomputed.get("folds"))
        summary_matches = all(
            split_qa.get(key) == recomputed.get(key)
            for key in (
                "unique_patients",
                "unique_wsi",
                "unique_roi",
                "patients_in_multiple_folds",
                "wsi_in_multiple_folds",
                "roi_duplicates",
                "missing_class_fold_pairs",
            )
        ) and qa_rows == recomputed_rows
        semantic_ok = (
            recomputed.get("status") == "PASS"
            and split_qa.get("status") == "PASS"
            and qa_lock.get("status") == "PASS"
            and summary_matches
            and declared_counts_ok
        )
        detail = (
            f"fold IDs={recomputed.get('observed_fold_ids')}; "
            f"counts={recomputed_rows}; summary_matches={summary_matches}"
        )
    except Exception as exc:
        semantic_ok = False
        detail = f"failed to recompute split QA: {type(exc).__name__}: {exc}"
    checks.append(AuditCheck("split.cv3", "PASS" if semantic_ok else "FAIL", detail))
    return checks


def audit_environment_lock(
    environment_lock_path: str | Path,
    pip_lock_path: str | Path,
) -> list[AuditCheck]:
    """Verify the exact runtime and its frozen package-list file."""

    environment_path = Path(environment_lock_path).resolve()
    requirements_path = Path(pip_lock_path).resolve()
    if not environment_path.is_file():
        return [AuditCheck("environment.lock", "FAIL", f"missing: {environment_path}")]
    lock = _load_json(environment_path)
    checks = [
        _check_file(
            "environment.pip_lock_integrity",
            requirements_path,
            expected_sha256=lock.get("pip_lock_sha256"),
            hash_contents=True,
        )
    ]
    actual: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    module_names = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "sklearn",
        "scipy": "scipy",
        "h5py": "h5py",
        "PIL": "PIL",
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
    }
    for key, module_name in module_names.items():
        module = importlib.import_module(module_name)
        actual[key] = getattr(module, "__version__", "unknown")
    import torch

    actual["cuda_available"] = bool(torch.cuda.is_available())
    actual["cuda_version"] = torch.version.cuda
    actual["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    mismatches = {
        key: {"observed": value, "expected": lock.get(key)}
        for key, value in actual.items()
        if value != lock.get(key)
    }
    checks.append(
        AuditCheck(
            "environment.runtime_identity",
            "PASS" if not mismatches else "FAIL",
            "exact Python/package/CUDA/GPU runtime matches lock"
            if not mismatches
            else f"runtime mismatches: {mismatches}",
        )
    )
    return checks


def audit_release_integrity_artifacts(
    *,
    required_manifest_path: str | Path,
    dataset_lock: Mapping[str, Any],
    summary_path: str | Path,
    lock_path: str | Path,
    workers: int = 4,
) -> list[AuditCheck]:
    """Validate frozen artifacts and re-hash every released file at time of use."""

    if workers < 1:
        raise ValueError("workers must be at least 1")

    summary = Path(summary_path).resolve()
    lock_file = Path(lock_path).resolve()
    rows_path = summary.parent / "release_file_integrity.csv"
    if not summary.is_file() or not lock_file.is_file() or not rows_path.is_file():
        return [
            AuditCheck(
                "dataset.release_file_md5_integrity",
                "DEFERRED",
                "run `pairbst images verify-release` before experiment approval",
            )
        ]
    lock = _load_json(lock_file)
    artifact_checks = [
        _check_file(
            "dataset.release_integrity_summary",
            summary,
            expected_sha256=lock.get("summary_sha256"),
            hash_contents=True,
        ),
        _check_file(
            "dataset.release_integrity_rows",
            rows_path,
            expected_sha256=lock.get("rows_sha256"),
            hash_contents=True,
        ),
    ]
    required_manifest_sha = _sha256(Path(required_manifest_path).resolve())
    expected_manifest_sha = str(dataset_lock.get("required_manifest_sha256", "")).upper()
    source_ok = (
        required_manifest_sha
        == expected_manifest_sha
        == str(lock.get("required_manifest_sha256", "")).upper()
    )
    artifact_checks.append(
        AuditCheck(
            "dataset.release_integrity_source_manifest",
            "PASS" if source_ok else "FAIL",
            f"required manifest SHA-256={required_manifest_sha}",
        )
    )
    summary_data = _load_json(summary)
    expected_count = int(dataset_lock.get("required_manifest_rows", -1))
    summary_ok = (
        summary_data.get("status") == "PASS"
        and int(summary_data.get("file_count", -1)) == expected_count
        and int(summary_data.get("matched_files", -1)) == expected_count
        and int(summary_data.get("failure_count", -1)) == 0
        and int(lock.get("file_count", -1)) == expected_count
        and int(lock.get("matched_files", -1)) == expected_count
        and lock.get("status") == "PASS"
    )
    released = pd.read_csv(
        Path(required_manifest_path).resolve(),
        sep="\t",
        header=None,
        names=("article_id", "file_id", "filename", "bytes", "md5", "url"),
        dtype={"filename": "string", "md5": "string"},
    )
    frozen_rows = pd.read_csv(rows_path, keep_default_na=False)
    manifest_contract_ok = False
    if (
        len(released) == expected_count
        and not released["filename"].duplicated().any()
        and {"filename", "expected_bytes", "expected_md5"}.issubset(
            frozen_rows.columns
        )
        and len(frozen_rows) == expected_count
        and not frozen_rows["filename"].duplicated().any()
    ):
        authoritative = {
            str(row.filename): (int(row.bytes), str(row.md5).lower())
            for row in released.itertuples(index=False)
        }
        frozen_contract = {
            str(row.filename): (
                int(row.expected_bytes),
                str(row.expected_md5).lower(),
            )
            for row in frozen_rows.itertuples(index=False)
        }
        manifest_contract_ok = frozen_contract == authoritative
    artifact_checks.append(
        AuditCheck(
            "dataset.release_integrity_manifest_contract",
            "PASS" if manifest_contract_ok else "FAIL",
            "frozen verification rows exactly match Figshare filename/bytes/MD5"
            if manifest_contract_ok
            else "frozen verification rows differ from the authoritative Figshare TSV",
        )
    )
    current_identity_ok = False
    current_identity_detail = "artifact integrity failed"
    if _all_pass(artifact_checks) and summary_ok:
        rows = frozen_rows
        required_columns = {"path", "expected_bytes", "expected_md5", "matches"}
        if required_columns.issubset(rows.columns) and len(rows) == expected_count:
            def verify_current(row: Any) -> str | None:
                file_path = Path(str(row.path))
                try:
                    return (
                        None
                        if (
                            file_path.is_file()
                            and file_path.stat().st_size == int(row.expected_bytes)
                            and _md5(file_path) == str(row.expected_md5).lower()
                            and str(row.matches).casefold() in {"true", "1"}
                        )
                        else str(file_path)
                    )
                except OSError:
                    return str(file_path)

            row_values = list(rows.itertuples(index=False))
            if workers == 1:
                observed = map(verify_current, row_values)
            else:
                pool = ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="pairbst-audit-md5"
                )
                observed = pool.map(verify_current, row_values)
            try:
                changed = [value for value in observed if value is not None][:5]
            finally:
                if workers != 1:
                    pool.shutdown(wait=True)
            current_identity_ok = not changed
            current_identity_detail = (
                "all 2,253 current files match the published byte size and MD5"
                if current_identity_ok
                else f"current file identities differ from Figshare: {changed}"
            )
        else:
            current_identity_detail = "release integrity rows have an invalid schema or count"
    artifact_checks.append(
        AuditCheck(
            "dataset.release_file_md5_integrity",
            "PASS" if summary_ok and current_identity_ok else "FAIL",
            current_identity_detail,
        )
    )
    return artifact_checks


def audit_deterministic_pilot(
    pilot_path: str | Path,
    model_lock: Mapping[str, Any],
    *,
    expected_dataset_manifest_sha256: str,
    expected_roi_uids: list[str],
    expected_records_fingerprint: str,
    expected_gpu_name: str,
    expected_cuda_version: str,
    required_input_paths: Iterable[str | Path] = (),
) -> AuditCheck:
    """Fail-closed validation of the complete deterministic model pilot."""

    path = Path(pilot_path).resolve()
    try:
        pilot = _load_json(path)
        expected_models = {
            registry_id: model_lock["models"][lock_id]
            for lock_id, registry_id in MODEL_LOCK_TO_REGISTRY.items()
            if lock_id in model_lock.get("models", {})
        }
        if (
            pilot.get("action") != "features.pilot"
            or pilot.get("status") != "PASS"
            or pilot.get("profile") != "official_model_specific"
            or pilot.get("device") != "cuda"
            or int(pilot.get("roi_count", -1)) != 2
            or float(pilot.get("atol", -1)) != 0.0
            or float(pilot.get("rtol", -1)) != 0.0
        ):
            raise ValueError("pilot header does not match the frozen exact-FP32 protocol")
        if (
            pilot.get("identity_schema") != "pairbst.stage_identity.v1"
            or str(pilot.get("dataset_manifest_sha256", "")).upper()
            != expected_dataset_manifest_sha256.upper()
            or pilot.get("pilot_roi_uids") != expected_roi_uids
            or pilot.get("ordered_records_fingerprint") != expected_records_fingerprint
            or pilot.get("release_integrity", {}).get("status") != "PASS"
        ):
            raise ValueError("pilot dataset/release identity does not match the canonical inputs")
        input_identities = pilot.get("input_identities")
        if not isinstance(input_identities, list) or not input_identities:
            raise ValueError("pilot input identities are missing")
        input_index: dict[str, Mapping[str, Any]] = {}
        for identity in input_identities:
            if not isinstance(identity, Mapping):
                raise ValueError("pilot input identity is not an object")
            input_path = Path(str(identity.get("path", ""))).resolve()
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            if (
                input_path.stat().st_size != int(identity.get("size_bytes", -1))
                or _sha256(input_path) != str(identity.get("sha256", "")).upper()
            ):
                raise ValueError(f"pilot input identity mismatch: {input_path}")
            input_index[str(input_path).casefold()] = identity
        missing_required_inputs = [
            str(Path(required).resolve())
            for required in required_input_paths
            if str(Path(required).resolve()).casefold() not in input_index
        ]
        if missing_required_inputs:
            raise ValueError(
                f"pilot does not bind required inputs: {missing_required_inputs}"
            )
        results = pilot.get("models")
        if not isinstance(results, list):
            raise ValueError("pilot models must be a result list")
        by_model = {str(result.get("model_id")): result for result in results}
        if set(by_model) != set(expected_models) or len(results) != len(expected_models):
            raise ValueError(f"pilot model set is {sorted(by_model)}")
        dataset_paths = ("features/center", "features/grid_mean", "features/grid_max")
        for model_id, expected in expected_models.items():
            result = by_model[model_id]
            representations = result.get("representations")
            if (
                result.get("status") != "PASS"
                or result.get("resume_reopen_bit_exact") is not True
                or not isinstance(representations, dict)
                or set(representations) != {"center", "mean", "max"}
            ):
                raise ValueError(f"pilot outcome failed for {model_id}")
            for strategy, comparison in representations.items():
                if (
                    comparison.get("finite") is not True
                    or comparison.get("matches") is not True
                    or float(comparison.get("max_abs_difference", -1)) != 0.0
                ):
                    raise ValueError(f"non-exact pilot comparison for {model_id}/{strategy}")
            output_files = result.get("output_files")
            if not isinstance(output_files, dict) or set(output_files) != {
                "repeat_a", "repeat_b", "resume"
            }:
                raise ValueError(f"pilot output identities are incomplete for {model_id}")
            opened: list[h5py.File] = []
            try:
                for output_name in ("repeat_a", "repeat_b", "resume"):
                    identity = output_files[output_name]
                    output_path = Path(str(identity["path"])).resolve()
                    if not output_path.is_file():
                        raise FileNotFoundError(output_path)
                    observed_sha = _sha256(output_path)
                    if observed_sha != str(identity["sha256"]).upper():
                        raise ValueError(f"pilot H5 SHA-256 mismatch: {output_path}")
                    handle = h5py.File(output_path, "r")
                    opened.append(handle)
                    if (
                        str(handle.attrs.get("schema_name", "")) != "pairbst_roi_features"
                        or str(handle.attrs.get("status", "")) != "complete"
                        or int(handle.attrs.get("num_rois", -1)) != 2
                        or not np.asarray(handle["completed"][:], dtype=bool).all()
                    ):
                        raise ValueError(f"invalid completed pilot H5: {output_path}")
                    provenance = json.loads(str(handle.attrs.get("provenance_json", "{}")))
                    hardware = provenance.get("hardware", {})
                    if (
                        provenance.get("model_name") != model_id
                        or provenance.get("checkpoint_sha256")
                        != expected["expected_sha256"]
                        or provenance.get("transform_profile") != "official_model_specific"
                        or provenance.get("deterministic_algorithms") is not True
                        or provenance.get("encoder_autocast_dtype") is not None
                        or provenance.get("tf32_allowed") is not False
                        or str(provenance.get("dataset_manifest_sha256", "")).upper()
                        != expected_dataset_manifest_sha256.upper()
                        or provenance.get("ordered_records_fingerprint")
                        != expected_records_fingerprint
                        or not isinstance(hardware, Mapping)
                        or hardware.get("requested_device") != "cuda"
                        or not str(hardware.get("resolved_device", "")).startswith(
                            "cuda"
                        )
                        or hardware.get("gpu_name") != expected_gpu_name
                        or str(hardware.get("cuda_version"))
                        != expected_cuda_version
                    ):
                        raise ValueError(f"pilot provenance mismatch for {model_id}")
                reference = opened[0]
                reference_roi = [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in reference["metadata/roi_uid"][:]
                ]
                if reference_roi != expected_roi_uids:
                    raise ValueError(
                        f"pilot ROI identities are not canonical for {model_id}"
                    )
                for handle in opened[1:]:
                    roi_ids = [
                        value.decode("utf-8") if isinstance(value, bytes) else str(value)
                        for value in handle["metadata/roi_uid"][:]
                    ]
                    if roi_ids != reference_roi:
                        raise ValueError(f"pilot ROI order mismatch for {model_id}")
                    for dataset_path in dataset_paths:
                        if not np.array_equal(reference[dataset_path][:], handle[dataset_path][:]):
                            raise ValueError(
                                f"pilot payload mismatch for {model_id}/{dataset_path}"
                            )
            finally:
                for handle in opened:
                    handle.close()
    except Exception as exc:
        return AuditCheck("model.deterministic_pilot", "FAIL", f"{path}: {exc}")
    return AuditCheck(
        "model.deterministic_pilot",
        "PASS",
        f"complete exact-FP32 repeat/resume payloads and provenance verified: {path}",
    )


def summarize_checks(checks: Iterable[AuditCheck]) -> dict[str, Any]:
    rows = list(checks)
    blocking_failures = [row.name for row in rows if row.blocking and row.status != "PASS"]
    return {
        "created_utc": utc_now(),
        "status": "READY" if not blocking_failures else "NOT_READY",
        "blocking_failures": blocking_failures,
        "checks": [asdict(row) for row in rows],
    }


def run_preparation_audit(
    paths_config: Mapping[str, Any],
    *,
    dataset_lock_path: str | Path,
    model_lock_path: str | Path,
    recovery_lock_path: str | Path | None = None,
    output_json: str | Path | None = None,
    decode_all: bool = False,
    image_qa_path: str | Path | None = None,
    image_qa_lock_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
    dataset_manifest_lock_path: str | Path | None = None,
    split_qa_path: str | Path | None = None,
    split_qa_lock_path: str | Path | None = None,
    split_csv_path: str | Path | None = None,
    split_lock_path: str | Path | None = None,
    model_pilot_path: str | Path | None = None,
    execution_hold_path: str | Path | None = None,
    environment_lock_path: str | Path | None = None,
    pip_lock_path: str | Path | None = None,
    release_integrity_summary_path: str | Path | None = None,
    release_integrity_lock_path: str | Path | None = None,
    hash_models: bool = False,
    extra_checks: Iterable[AuditCheck] = (),
) -> dict[str, Any]:
    model_lock = _load_json(model_lock_path)
    checks = audit_dataset(
        paths_config,
        _load_json(dataset_lock_path),
        decode_all=decode_all,
        image_qa_path=image_qa_path,
        image_qa_lock_path=image_qa_lock_path,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_lock_path=dataset_manifest_lock_path,
    )
    checks.extend(audit_models(paths_config, model_lock, hash_contents=hash_models))
    checks.extend(
        audit_recovery_inputs(
            paths_config,
            _load_json(recovery_lock_path) if recovery_lock_path is not None else None,
        )
    )
    if environment_lock_path is not None and pip_lock_path is not None:
        checks.extend(audit_environment_lock(environment_lock_path, pip_lock_path))
    if (
        release_integrity_summary_path is not None
        and release_integrity_lock_path is not None
    ):
        checks.extend(
            audit_release_integrity_artifacts(
                required_manifest_path=get_path(
                    paths_config, "dataset", "required_manifest"
                ),
                dataset_lock=_load_json(dataset_lock_path),
                summary_path=release_integrity_summary_path,
                lock_path=release_integrity_lock_path,
            )
        )
    if all(
        value is not None
        for value in (split_qa_path, split_qa_lock_path, split_csv_path, split_lock_path)
    ):
        checks.extend(
            audit_cv_split_artifacts(
                split_csv_path=split_csv_path,
                split_lock_path=split_lock_path,
                split_qa_path=split_qa_path,
                split_qa_lock_path=split_qa_lock_path,
            )
        )
    else:
        checks.append(
            AuditCheck(
                "split.cv3",
                "DEFERRED",
                "split CSV, split lock, QA summary, and QA lock were not all supplied",
            )
        )
    if model_pilot_path is not None and Path(model_pilot_path).is_file():
        try:
            if dataset_manifest_lock_path is None:
                raise ValueError("dataset manifest lock is required to audit the pilot")
            if environment_lock_path is None:
                raise ValueError("environment lock is required to audit the pilot")
            manifest_lock = _load_json(dataset_manifest_lock_path)
            environment_lock = _load_json(environment_lock_path)
            all_records = load_roi_records(
                get_path(paths_config, "dataset", "metadata_csv"),
                get_path(paths_config, "dataset", "roi_root"),
                strict=True,
            )
            pilot_records = sorted(all_records, key=lambda record: record.roi_uid)[:2]
            required_pilot_inputs = [
                value
                for value in (
                    dataset_manifest_path,
                    dataset_manifest_lock_path,
                    split_csv_path,
                    split_lock_path,
                    release_integrity_summary_path,
                    release_integrity_lock_path,
                    environment_lock_path,
                )
                if value is not None
            ]
            checks.append(
                audit_deterministic_pilot(
                    model_pilot_path,
                    model_lock,
                    expected_dataset_manifest_sha256=str(
                        manifest_lock["manifest_sha256"]
                    ),
                    expected_roi_uids=[record.roi_uid for record in pilot_records],
                    expected_records_fingerprint=records_fingerprint(pilot_records),
                    expected_gpu_name=str(environment_lock["gpu"]),
                    expected_cuda_version=str(environment_lock["cuda_version"]),
                    required_input_paths=required_pilot_inputs,
                )
            )
        except Exception as exc:
            checks.append(
                AuditCheck(
                    "model.deterministic_pilot",
                    "FAIL",
                    f"could not establish canonical pilot inputs: {exc}",
                )
            )
    else:
        checks.append(
            AuditCheck(
                "model.deterministic_pilot",
                "DEFERRED",
                "intentionally not run before implementation review",
            )
        )
    if execution_hold_path is not None and Path(execution_hold_path).is_file():
        hold = _load_json(execution_hold_path)
        active = bool(hold.get("hold", True))
        checks.append(
            AuditCheck(
                "execution.review_hold",
                "FAIL" if active else "PASS",
                str(hold.get("reason", "execution hold")),
            )
        )
    checks.extend(extra_checks)
    result = summarize_checks(checks)
    if output_json is not None:
        write_json_atomic(result, output_json)
    return result
