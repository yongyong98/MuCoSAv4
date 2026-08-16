"""Safe orchestration for the PAIR-BST revision benchmark.

Preparation functions in this module are intentionally usable while the
execution hold is active.  Functions that can create experimental results
check ``locks/EXECUTION_HOLD.json`` before opening a checkpoint, training a
classifier, running retrieval, bootstrapping results, or writing final tables.

Canonical feature files are *always* aligned to the frozen split by
``roi_uid``.  HDF5 row order is never treated as cohort order, and all patient
and outcome metadata are compared after the join.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
import json
import os
import re
import shutil
import tempfile
import time

import h5py
import numpy as np
import pandas as pd

from .audit import (
    audit_models,
    audit_release_integrity_artifacts,
    run_preparation_audit,
    summarize_checks,
)
from .classification.linear_probe import (
    CANONICAL_PROTOCOL_ID,
    LinearProbeConfig,
    run_outer_cv_linear_probe,
)
from .config import load_yaml, resolve_path
from .data_checks import verify_images_and_centers, verify_release_files_against_manifest
from .datasets import load_roi_records, records_fingerprint
from .features.extract import (
    ExtractionConfig,
    ModelExtractionRun,
    extract_models_sequentially,
)
from .features.h5store import ResumableFeatureStore, validate_feature_file
from .hashing import sha256_file
from .manifest import create_manifest_artifacts
from .models.registry import get_model_spec, get_transform_spec, list_model_names
from .qa import write_cv_split_qa
from .reporting.tables import write_final_results_bundle
from .retrieval.search import run_patient_disjoint_cv_retrieval
from .splits import make_patient_folds, write_cv_split_artifacts
from .statistics.bootstrap import (
    cluster_bootstrap_classification,
    cluster_bootstrap_mean,
)
from .statistics.comparisons import (
    apply_holm_correction,
    paired_model_comparison,
    paired_query_metric_comparison,
)


MODEL_IDS = tuple(list_model_names())
STRATEGIES = ("center", "mean", "max")
FEATURE_DATASETS = {
    "center": "features/center",
    "mean": "features/grid_mean",
    "max": "features/grid_max",
}
REQUIRED_JOIN_COLUMNS = (
    "roi_uid",
    "patient_uid",
    "diagnosis",
    "differentiation",
    "growth_pattern",
)
MODEL_ID_ALIASES = {
    "resnet50_in1k_v2": "resnet50_v2",
    "swin_t_in1k_v1": "swin_t",
    "retccl_resnet50": "retccl",
    "uni_vitl16": "uni",
    "uni2-h": "uni2_h",
    "uni2h": "uni2_h",
    "prov-gigapath": "prov_gigapath",
    "gigapath": "prov_gigapath",
    "virchow-2": "virchow2",
    "virchow_v2": "virchow2",
}
STRATEGY_ALIASES = {
    "center_crop": "center",
    "grid_mean": "mean",
    "mean_pooling": "mean",
    "grid_max": "max",
    "max_pooling": "max",
}


class ExecutionHoldError(RuntimeError):
    """Raised when an experimental stage is requested before review release."""


@dataclass(frozen=True)
class PipelineContext:
    """Resolved configuration and conventional pipeline paths."""

    paths_config_path: Path
    protocol_path: Path
    models_config_path: Path
    comparisons_config_path: Path
    paths: Mapping[str, Any]
    protocol: Mapping[str, Any]
    models_config: Mapping[str, Any]
    comparisons: Mapping[str, Any]
    project_root: Path

    @classmethod
    def load(
        cls,
        paths_config: str | Path = "configs/paths.local.yaml",
        protocol: str | Path = "configs/protocol_cv3_independent_seed_oof_v1.yaml",
        models_config: str | Path = "configs/models.yaml",
        comparisons: str | Path = "configs/comparisons.yaml",
    ) -> "PipelineContext":
        path_cfg = load_yaml(paths_config)
        source_dir = Path(path_cfg["_source_dir"])
        project_root = resolve_path(path_cfg.get("project_root", ".."), base=source_dir)

        def resolve_config_path(value: str | Path) -> Path:
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                return candidate.resolve()
            # CLI defaults are normally relative to the working directory.  If
            # not found there, resolve them relative to the project root.
            cwd_candidate = candidate.resolve()
            return cwd_candidate if cwd_candidate.exists() else (project_root / candidate).resolve()

        protocol_path = resolve_config_path(protocol)
        models_path = resolve_config_path(models_config)
        comparisons_path = resolve_config_path(comparisons)
        return cls(
            paths_config_path=Path(path_cfg["_source_path"]),
            protocol_path=protocol_path,
            models_config_path=models_path,
            comparisons_config_path=comparisons_path,
            paths=path_cfg,
            protocol=load_yaml(protocol_path),
            models_config=load_yaml(models_path),
            comparisons=load_yaml(comparisons_path),
            project_root=project_root,
        )

    def path(self, section: str, key: str) -> Path:
        value = self.paths.get(section, {}).get(key)
        if value is None:
            raise KeyError(f"Missing paths configuration: {section}.{key}")
        candidate = Path(str(value)).expanduser()
        config_dir = Path(str(self.paths.get("_source_dir", self.project_root)))
        return candidate.resolve() if candidate.is_absolute() else (config_dir / candidate).resolve()

    @property
    def locks_dir(self) -> Path:
        return self.path("output", "locks")

    @property
    def runs_dir(self) -> Path:
        return self.path("output", "runs")

    @property
    def final_dir(self) -> Path:
        return self.path("output", "final")

    @property
    def hold_path(self) -> Path:
        return self.locks_dir / "EXECUTION_HOLD.json"

    @property
    def split_csv(self) -> Path:
        return self.locks_dir / "folds_cv3_v1.csv"

    @property
    def split_lock(self) -> Path:
        return self.locks_dir / "folds_cv3_v1.lock.json"

    @property
    def manifest_csv(self) -> Path:
        return self.locks_dir / "dataset_manifest_v1.csv"

    @property
    def manifest_lock(self) -> Path:
        return self.locks_dir / "dataset_manifest_v1.lock.json"

    def absolute_paths_mapping(self) -> dict[str, Any]:
        """Return a config copy compatible with legacy ``get_path`` consumers."""

        result: dict[str, Any] = {"_source_dir": str(self.project_root)}
        for section in ("dataset", "recovery", "weights", "output"):
            values = self.paths.get(section, {})
            result[section] = {
                key: str(self.path(section, key)) for key in values if not str(key).startswith("_")
            }
        return result


@dataclass(frozen=True)
class FeatureAlignment:
    """Verified mapping from split rows to HDF5 rows."""

    feature_path: Path
    split: pd.DataFrame
    h5_rows: np.ndarray
    provenance: Mapping[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        Path(temporary_name).write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def _atomic_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False, encoding="utf-8-sig", lineterminator="\n")
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def _atomic_csv_gzip(path: str | Path, frame: pd.DataFrame) -> Path:
    """Write a deterministic gzip-compressed CSV and replace atomically."""

    destination = Path(path).resolve()
    if destination.suffixes[-2:] != [".csv", ".gz"]:
        raise ValueError("Compressed CSV destinations must end in .csv.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(
            temporary_name,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        )
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    """Write a compressed NumPy archive and replace atomically."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with Path(temporary_name).open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def file_identity(path: str | Path) -> dict[str, Any]:
    """Return a cryptographic identity for a concrete stage input or output."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def collect_output_identities(
    directory: str | Path,
    *,
    excluded_names: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Hash every non-temporary file produced by one pipeline stage."""

    root = Path(directory).resolve()
    excluded = set(excluded_names)
    identities = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.name not in excluded
            and not path.name.startswith(".")
            and not path.name.endswith(".tmp")
        ):
            identities.append(file_identity(path))
    if not identities:
        raise ValueError(f"No stage outputs were found under {root}")
    return identities


def _verify_identities(identities: Any, *, label: str) -> None:
    if not isinstance(identities, list) or not identities:
        raise ValueError(f"{label} must be a non-empty identity list")
    observed_paths: set[str] = set()
    for identity in identities:
        if not isinstance(identity, Mapping):
            raise ValueError(f"{label} contains a non-object identity")
        path = Path(str(identity.get("path", ""))).resolve()
        normalized = str(path).casefold()
        if normalized in observed_paths:
            raise ValueError(f"{label} contains duplicate path {path}")
        observed_paths.add(normalized)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = int(identity.get("size_bytes", -1))
        if path.stat().st_size != expected_size:
            raise ValueError(
                f"{label} size mismatch for {path}: {path.stat().st_size} != {expected_size}"
            )
        observed_hash = sha256_file(path)
        if observed_hash != str(identity.get("sha256", "")).upper():
            raise ValueError(f"{label} SHA-256 mismatch for {path}")


def verify_stage_manifest(
    manifest_path: str | Path,
    *,
    expected_action: str,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    """Verify a stage manifest and every recursively bound file identity."""

    return _verify_stage_manifest_tree(
        Path(manifest_path).resolve(),
        expected_action=expected_action,
        expected_profile=expected_profile,
        cache={},
        active=set(),
    )


def _verify_stage_manifest_tree(
    path: Path,
    *,
    expected_action: str,
    expected_profile: str | None,
    cache: dict[str, dict[str, Any]],
    active: set[str],
) -> dict[str, Any]:
    normalized_path = str(path).casefold()
    if normalized_path in active:
        raise ValueError(f"Stage manifest lineage contains a cycle: {path}")
    if normalized_path in cache:
        cached = cache[normalized_path]
        if cached.get("action") != expected_action or (
            expected_profile is not None and cached.get("profile") != expected_profile
        ):
            raise ValueError(f"Cached stage manifest identity is incompatible: {path}")
        return cached

    if not path.is_file():
        raise FileNotFoundError(f"Required upstream stage manifest is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("action") != expected_action:
        raise ValueError(f"Unexpected stage manifest action in {path}")
    if expected_profile is not None and data.get("profile") != expected_profile:
        raise ValueError(
            f"Stage manifest profile {data.get('profile')!r} != {expected_profile!r}: {path}"
        )
    if data.get("identity_schema") != "pairbst.stage_identity.v1":
        raise ValueError(f"Stage manifest has no supported identity schema: {path}")
    for key in ("input_identities", "output_identities"):
        identities = data.get(key)
        if identities:
            _verify_identities(identities, label=f"{expected_action}.{key}")
        elif key == "output_identities":
            raise ValueError(f"Stage manifest lacks output identities: {path}")
    upstream = data.get("input_stage_manifests", [])
    if not isinstance(upstream, list):
        raise ValueError(f"input_stage_manifests must be a list: {path}")
    if upstream:
        _verify_identities(upstream, label=f"{expected_action}.input_stage_manifests")
    active.add(normalized_path)
    try:
        for identity in upstream:
            upstream_path = Path(str(identity["path"])).resolve()
            upstream_data = json.loads(upstream_path.read_text(encoding="utf-8"))
            if not isinstance(upstream_data, dict) or not upstream_data.get("action"):
                raise ValueError(f"Invalid upstream stage manifest: {upstream_path}")
            _verify_stage_manifest_tree(
                upstream_path,
                expected_action=str(upstream_data["action"]),
                expected_profile=str(data["profile"]) if data.get("profile") else None,
                cache=cache,
                active=active,
            )
    finally:
        active.remove(normalized_path)
    cache[normalized_path] = data
    return data


def _identity_fingerprint(identity: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(Path(str(identity.get("path", ""))).resolve()).casefold(),
        int(identity.get("size_bytes", -1)),
        str(identity.get("sha256", "")).upper(),
    )


def require_manifest_bound_paths(
    manifest: Mapping[str, Any],
    paths: Sequence[str | Path],
    *,
    identity_field: str,
    label: str,
) -> list[Mapping[str, Any]]:
    """Require each concrete consumer input to be bound by its producer."""

    identities = manifest.get(identity_field)
    if not isinstance(identities, list) or not identities:
        raise ValueError(f"{label} has no {identity_field}")
    index = {
        str(Path(str(identity["path"])).resolve()).casefold(): identity
        for identity in identities
    }
    requested = [Path(path).resolve() for path in paths]
    missing = [str(path) for path in requested if str(path).casefold() not in index]
    if missing:
        raise ValueError(f"{label} does not bind consumed files: {missing}")
    return [index[str(path).casefold()] for path in requested]


def verify_compatible_feature_lineage(
    classification_manifest: Mapping[str, Any],
    retrieval_manifest: Mapping[str, Any],
) -> None:
    """Require classification and retrieval to derive from one exact H5 set."""

    for field in ("input_stage_manifests", "input_identities"):
        classification = classification_manifest.get(field)
        retrieval = retrieval_manifest.get(field)
        if (
            not isinstance(classification, list)
            or not classification
            or not isinstance(retrieval, list)
            or not retrieval
            or {_identity_fingerprint(value) for value in classification}
            != {_identity_fingerprint(value) for value in retrieval}
        ):
            raise ValueError(
                "Classification/retrieval feature lineage mismatch in "
                f"{field}; refusing to mix runs"
            )


def require_exact_stage_manifest_inputs(
    manifest: Mapping[str, Any],
    expected_manifest_paths: Sequence[str | Path],
    *,
    label: str,
) -> None:
    observed = manifest.get("input_stage_manifests")
    if not isinstance(observed, list) or not observed:
        raise ValueError(f"{label} has no bound upstream stage manifests")
    expected = [file_identity(path) for path in expected_manifest_paths]
    if {_identity_fingerprint(value) for value in observed} != {
        _identity_fingerprint(value) for value in expected
    }:
        raise ValueError(f"{label} was not produced from the selected upstream runs")


def execution_context_identities(context: PipelineContext) -> list[dict[str, Any]]:
    """Bind a result to the exact configs, split, manifest, environment, and code."""

    candidates = [
        context.paths_config_path,
        context.protocol_path,
        context.models_config_path,
        context.comparisons_config_path,
        context.split_csv,
        context.split_lock,
        context.manifest_csv,
        context.manifest_lock,
        context.locks_dir / "environment.current.json",
        context.project_root / "requirements.mucosa-cu128.lock.txt",
    ]
    candidates.extend(sorted((context.project_root / "src" / "pairbst").rglob("*.py")))
    unique: dict[str, Path] = {}
    for candidate in candidates:
        path = Path(candidate).resolve()
        unique[str(path).casefold()] = path
    return [file_identity(path) for path in unique.values()]


def _safe_name(value: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return result or "item"


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def normalize_model_id(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "_")
    normalized = MODEL_ID_ALIASES.get(normalized, normalized)
    return get_model_spec(normalized).name


def normalize_strategy(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = STRATEGY_ALIASES.get(normalized, normalized)
    if normalized not in STRATEGIES:
        raise KeyError(f"Unknown strategy {value!r}; expected {STRATEGIES}")
    return normalized


def select_models(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values or any(value.strip().lower() == "all" for value in values):
        return MODEL_IDS
    selected: list[str] = []
    for value in values:
        for item in value.split(","):
            model = normalize_model_id(item)
            if model not in selected:
                selected.append(model)
    return tuple(selected)


def experiment_grid(
    models: Sequence[str], tasks: Sequence[str]
) -> list[dict[str, str]]:
    return [
        {"model_id": model, "strategy": strategy, "task": task}
        for model in models
        for strategy in STRATEGIES
        for task in tasks
    ]


def require_execution_permission(
    hold_path: str | Path,
    *,
    action: str,
    override_hold: bool = False,
) -> dict[str, Any]:
    """Fail closed unless the owner released the hold or explicitly overrides it."""

    path = Path(hold_path).resolve()
    if override_hold:
        return {"allowed": True, "override": True, "hold_path": str(path), "action": action}
    if not path.is_file():
        raise ExecutionHoldError(
            f"Execution hold file is missing; refusing {action!r} without --override-hold: {path}"
        )
    hold = json.loads(path.read_text(encoding="utf-8"))
    if bool(hold.get("hold", True)):
        raise ExecutionHoldError(
            f"Execution is on hold for {action}: {hold.get('reason', 'review required')} "
            "Use --override-hold only after dataset-owner approval."
        )
    return {"allowed": True, "override": False, "hold_path": str(path), "action": action}


def verify_frozen_split(split_csv: str | Path, split_lock: str | Path) -> dict[str, Any]:
    split_path = Path(split_csv).resolve()
    lock_path = Path(split_lock).resolve()
    if not split_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError(f"Frozen split or lock is missing: {split_path}, {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    actual = sha256_file(split_path).upper()
    expected = str(lock.get("roi_split_sha256", "")).upper()
    if not expected or actual != expected:
        raise ValueError(f"Frozen split SHA-256 mismatch: {actual} != {expected}")
    return {"split_csv": str(split_path), "sha256": actual, "lock": str(lock_path)}


def build_manifest(
    context: PipelineContext,
    *,
    verify_dimensions: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = {
        "action": "manifest.build",
        "metadata_csv": context.path("dataset", "metadata_csv"),
        "roi_root": context.path("dataset", "roi_root"),
        "manifest_csv": context.manifest_csv,
        "lock_json": context.manifest_lock,
        "verify_dimensions": verify_dimensions,
    }
    if dry_run:
        return {"dry_run": True, **plan}
    return create_manifest_artifacts(
        plan["metadata_csv"],
        plan["roi_root"],
        plan["manifest_csv"],
        plan["lock_json"],
        verify_files=True,
        strict_file_set=True,
        verify_dimensions=verify_dimensions,
    )


def build_splits(
    context: PipelineContext,
    *,
    optimize_balance: bool = True,
    max_balance_passes: int = 30,
    dry_run: bool = False,
) -> dict[str, Any]:
    outer = context.protocol["outer_cv"]
    n_splits = int(outer["folds"])
    seed = int(outer["assignment_seed"])
    plan = {
        "action": "splits.build",
        "manifest_csv": context.manifest_csv,
        "roi_split_csv": context.split_csv,
        "patient_folds_csv": context.locks_dir / "folds_cv3_patients_v1.csv",
        "lock_json": context.split_lock,
        "n_splits": n_splits,
        "seed": seed,
        "optimize_balance": optimize_balance,
        "max_balance_passes": max_balance_passes,
    }
    if dry_run:
        return {"dry_run": True, **plan}
    manifest = pd.read_csv(
        context.manifest_csv,
        dtype={"patient_idx": "string", "roi_idx": "string"},
        keep_default_na=False,
    )
    patient_folds = make_patient_folds(
        manifest,
        n_splits=n_splits,
        seed=seed,
        optimize_balance=optimize_balance,
        max_balance_passes=max_balance_passes,
    )
    expected_counts = sorted(int(value) for value in outer.get("target_patient_counts", []))
    observed_counts = sorted(patient_folds.groupby("fold")["patient_uid"].size().astype(int).tolist())
    if expected_counts and observed_counts != expected_counts:
        raise ValueError(
            f"Generated fold patient counts {observed_counts} do not match protocol {expected_counts}"
        )
    lock = write_cv_split_artifacts(
        manifest,
        patient_folds,
        roi_split_csv=plan["roi_split_csv"],
        patient_folds_csv=plan["patient_folds_csv"],
        lock_json=plan["lock_json"],
        seed=seed,
    )
    qa_dir = context.locks_dir / "folds_cv3_v1_qa"
    qa = write_cv_split_qa(
        plan["roi_split_csv"], qa_dir, tasks=tuple(context.protocol["tasks"])
    )
    return {"lock": lock, "qa": qa, "qa_directory": qa_dir}


def verify_images(
    context: PipelineContext,
    *,
    legacy_center_root: str | Path | None = None,
    workers: int = 1,
    output_directory: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    output = Path(output_directory).resolve() if output_directory else context.locks_dir / "data_image_center_qa"
    configured_center = context.paths.get("legacy_center_root")
    resolved_center = (
        Path(legacy_center_root).resolve()
        if legacy_center_root is not None
        else resolve_path(configured_center, base=context.paths["_source_dir"])
        if configured_center
        else None
    )
    plan = {
        "action": "images.verify",
        "metadata_csv": context.path("dataset", "metadata_csv"),
        "roi_root": context.path("dataset", "roi_root"),
        "legacy_center_root": resolved_center,
        "workers": workers,
        "output_directory": output,
    }
    if dry_run:
        return {"dry_run": True, **plan}
    records = load_roi_records(plan["metadata_csv"], plan["roi_root"], strict=True)
    return verify_images_and_centers(
        records,
        legacy_center_root=plan["legacy_center_root"],
        output_directory=output,
        workers=workers,
    )


def verify_release_integrity(
    context: PipelineContext,
    *,
    workers: int = 4,
    output_directory: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Hash all 2,252 PNGs and metadata against the published Figshare manifest."""

    canonical_output = (context.locks_dir / "figshare_release_integrity").resolve()
    output = Path(output_directory).resolve() if output_directory else canonical_output
    if output != canonical_output:
        raise ValueError(
            "Figshare release-integrity artifacts are canonical lock inputs and must be "
            f"written to {canonical_output}"
        )
    plan = {
        "action": "images.verify_release",
        "metadata_csv": context.path("dataset", "metadata_csv"),
        "roi_root": context.path("dataset", "roi_root"),
        "required_manifest": context.path("dataset", "required_manifest"),
        "workers": workers,
        "output_directory": output,
    }
    if dry_run:
        return {"dry_run": True, **plan}
    records = load_roi_records(plan["metadata_csv"], plan["roi_root"], strict=True)
    result = verify_release_files_against_manifest(
        records,
        metadata_csv=plan["metadata_csv"],
        required_manifest=plan["required_manifest"],
        output_directory=output,
        workers=workers,
    )
    summary_path = output / "release_file_integrity_summary.json"
    rows_path = output / "release_file_integrity.csv"
    lock = {
        "schema_version": 1,
        "status": result["status"],
        "required_manifest_sha256": result["required_manifest_sha256"],
        "file_count": result["file_count"],
        "matched_files": result["matched_files"],
        "summary_sha256": sha256_file(summary_path),
        "rows_sha256": sha256_file(rows_path),
    }
    lock_path = context.locks_dir / "figshare_release_integrity.lock.json"
    _atomic_json(lock_path, lock)
    return {**plan, **result, "lock_path": lock_path}


def verify_models(
    context: PipelineContext,
    *,
    hash_contents: bool = True,
    output_json: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    model_lock = context.locks_dir / "models.expected.json"
    plan = {
        "action": "models.verify",
        "model_lock": model_lock,
        "hash_contents": hash_contents,
        "output_json": Path(output_json).resolve() if output_json else context.runs_dir / "preflight" / "models_verify.json",
    }
    if dry_run:
        return {"dry_run": True, **plan}
    lock = json.loads(model_lock.read_text(encoding="utf-8"))
    result = summarize_checks(
        audit_models(context.absolute_paths_mapping(), lock, hash_contents=hash_contents)
    )
    _atomic_json(plan["output_json"], result)
    return result


def run_audit(
    context: PipelineContext,
    *,
    decode_all: bool = False,
    hash_models: bool = False,
    output_json: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    destination = Path(output_json).resolve() if output_json else context.runs_dir / "preflight" / "preparation_audit.json"
    plan = {
        "action": "audit.run",
        "decode_all": decode_all,
        "hash_models": hash_models,
        "output_json": destination,
    }
    if dry_run:
        return {"dry_run": True, **plan}
    return run_preparation_audit(
        context.absolute_paths_mapping(),
        dataset_lock_path=context.locks_dir / "dataset.expected.json",
        model_lock_path=context.locks_dir / "models.expected.json",
        recovery_lock_path=context.locks_dir / "recovery.expected.json",
        output_json=destination,
        decode_all=decode_all,
        image_qa_path=context.locks_dir / "data_image_center_qa" / "image_center_summary.json",
        image_qa_lock_path=context.locks_dir / "data_image_center_qa.lock.json",
        dataset_manifest_path=context.manifest_csv,
        dataset_manifest_lock_path=context.manifest_lock,
        split_qa_path=context.locks_dir / "folds_cv3_v1_qa" / "split_qa.json",
        split_qa_lock_path=context.locks_dir / "folds_cv3_v1_qa.lock.json",
        split_csv_path=context.split_csv,
        split_lock_path=context.split_lock,
        model_pilot_path=context.locks_dir / "model_deterministic_pilot.json",
        execution_hold_path=context.hold_path,
        environment_lock_path=context.locks_dir / "environment.current.json",
        pip_lock_path=context.project_root / "requirements.mucosa-cu128.lock.txt",
        release_integrity_summary_path=(
            context.locks_dir
            / "figshare_release_integrity"
            / "release_file_integrity_summary.json"
        ),
        release_integrity_lock_path=(
            context.locks_dir / "figshare_release_integrity.lock.json"
        ),
        hash_models=hash_models,
    )


def _load_manifest_hash(context: PipelineContext) -> str:
    lock = json.loads(context.manifest_lock.read_text(encoding="utf-8"))
    expected = str(lock.get("manifest_sha256", "")).upper()
    actual = sha256_file(context.manifest_csv).upper()
    if not expected or actual != expected:
        raise ValueError(f"Dataset manifest SHA-256 mismatch: {actual} != {expected}")
    return actual


def require_release_integrity(context: PipelineContext) -> dict[str, Any]:
    """Fail closed unless every published input retained its verified identity."""

    dataset_lock_path = context.locks_dir / "dataset.expected.json"
    dataset_lock = json.loads(dataset_lock_path.read_text(encoding="utf-8"))
    checks = audit_release_integrity_artifacts(
        required_manifest_path=context.path("dataset", "required_manifest"),
        dataset_lock=dataset_lock,
        summary_path=(
            context.locks_dir
            / "figshare_release_integrity"
            / "release_file_integrity_summary.json"
        ),
        lock_path=context.locks_dir / "figshare_release_integrity.lock.json",
    )
    failures = [check for check in checks if check.status != "PASS"]
    if failures:
        detail = "; ".join(
            f"{check.name}={check.status}: {check.detail}" for check in failures
        )
        raise ValueError(
            "Figshare release-integrity gate is not PASS; refusing feature inference: "
            f"{detail}"
        )
    return {
        "status": "PASS",
        "checks": [asdict(check) for check in checks],
    }


def feature_directory(
    context: PipelineContext,
    profile: str,
    explicit: str | Path | None = None,
) -> Path:
    return Path(explicit).resolve() if explicit else context.runs_dir / "features" / _safe_name(profile)


def feature_file_path(feature_dir: str | Path, model_id: str) -> Path:
    return Path(feature_dir).resolve() / f"{normalize_model_id(model_id)}.h5"


def extract_features(
    context: PipelineContext,
    *,
    models: Sequence[str] | None = None,
    profile: str | None = None,
    device: str = "cuda",
    batch_size: int | None = None,
    autocast_dtype: str | None = None,
    output_directory: str | Path | None = None,
    override_hold: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected = select_models(models)
    chosen_profile = profile or str(context.protocol["feature_protocol"]["primary_transform"])
    if device != "cuda":
        raise ValueError("Benchmark feature extraction is frozen to the audited CUDA device")
    if autocast_dtype is not None:
        raise ValueError("Benchmark feature extraction is frozen to FP32 with autocast disabled")
    if batch_size is not None:
        raise ValueError("Benchmark extraction uses the frozen per-model batch sizes")
    output = feature_directory(context, chosen_profile, output_directory)
    default_batch = int(
        context.models_config.get("execution", {}).get("default_grid_batch_size", 32)
    )
    configured_models = context.models_config.get("models", {})
    planned = [
        {
            "model_id": model,
            "output_h5": feature_file_path(output, model),
            "batch_size": int(
                batch_size
                or configured_models.get(model, {}).get(
                    "extraction_batch_size_rtx3090_fp32", default_batch
                )
            ),
        }
        for model in selected
    ]
    plan = {
        "action": "features.extract",
        "models": list(selected),
        "profile": chosen_profile,
        "device": device,
        "output_directory": output,
        "runs": planned,
    }
    if dry_run:
        return {"dry_run": True, **plan}
    permission = require_execution_permission(
        context.hold_path, action="real feature extraction", override_hold=override_hold
    )
    release_integrity = require_release_integrity(context)
    verify_frozen_split(context.split_csv, context.split_lock)
    manifest_sha = _load_manifest_hash(context)
    context_inputs = execution_context_identities(context)
    records = load_roi_records(
        context.path("dataset", "metadata_csv"), context.path("dataset", "roi_root"), strict=True
    )
    runs: list[ModelExtractionRun] = []
    for model in selected:
        checkpoint = context.path("weights", {
            "resnet50_v2": "resnet50_v2",
            "swin_t": "swin_t",
            "retccl": "retccl",
            "uni": "uni",
            "uni2_h": "uni2_h",
            "prov_gigapath": "prov_gigapath",
            "virchow2": "virchow2",
        }[model])
        retccl_source = context.path("weights", "root") / "RetCCL" / "ResNet.py" if model == "retccl" else None
        runs.append(
            ModelExtractionRun(
                model_name=model,
                checkpoint_path=checkpoint,
                output_path=feature_file_path(output, model),
                retccl_source_path=retccl_source,
            )
        )
    # The large encoders cannot safely share one global batch size on a 24 GB
    # RTX 3090.  Each one-model call releases the model before the next run and
    # honors its frozen FP32 batch size unless the CLI explicitly overrides it.
    outputs: list[Path] = []
    for run, item in zip(runs, planned, strict=True):
        extraction_config = ExtractionConfig(
            batch_size=int(item["batch_size"]),
            device=device,
            transform_profile=chosen_profile,
            autocast_dtype=autocast_dtype,
            deterministic_algorithms=True,
        )
        started_at = time.monotonic()

        def progress(
            model_name: str,
            completed: int,
            total: int,
            _record: Any,
        ) -> None:
            if completed != 1 and completed != total and completed % 10 != 0:
                return
            elapsed = max(time.monotonic() - started_at, 1e-9)
            rate = completed / elapsed
            eta_seconds = (total - completed) / rate if rate > 0 else float("inf")
            print(
                "[features.extract] "
                f"model={model_name} session_rois={completed}/{total} "
                f"rate={rate:.3f}_roi_s eta={eta_seconds / 3600:.2f}_h",
                flush=True,
            )

        outputs.extend(
            extract_models_sequentially(
                records,
                [run],
                dataset_manifest_sha256=manifest_sha,
                config=extraction_config,
                progress=progress,
            )
        )
    result = {
        **plan,
        "permission": permission,
        "release_integrity": release_integrity,
        "outputs": outputs,
        "identity_schema": "pairbst.stage_identity.v1",
        "input_identities": [
            *context_inputs,
            file_identity(
                context.locks_dir
                / "figshare_release_integrity"
                / "release_file_integrity_summary.json"
            ),
            file_identity(context.locks_dir / "figshare_release_integrity.lock.json"),
        ],
        "input_stage_manifests": [],
        "output_identities": [file_identity(path) for path in outputs],
    }
    _atomic_json(output / "extraction_manifest.json", result)
    return result


def _feature_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            strategy: np.asarray(handle[dataset][:], dtype=np.float32)
            for strategy, dataset in FEATURE_DATASETS.items()
        }


def run_feature_pilot(
    context: PipelineContext,
    *,
    models: Sequence[str] | None = None,
    profile: str | None = None,
    device: str = "cuda",
    batch_size: int | None = None,
    autocast_dtype: str | None = None,
    roi_count: int = 2,
    atol: float = 0.0,
    rtol: float = 0.0,
    output_directory: str | Path | None = None,
    summary_json: str | Path | None = None,
    override_hold: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Repeat a tiny extraction and verify determinism plus H5 resume behavior."""

    if roi_count < 1:
        raise ValueError("roi_count must be positive")
    if atol < 0 or rtol < 0:
        raise ValueError("pilot tolerances must be non-negative")
    selected = select_models(models)
    chosen_profile = profile or str(context.protocol["feature_protocol"]["primary_transform"])
    if device != "cuda":
        raise ValueError("The deterministic benchmark pilot is frozen to the audited CUDA device")
    if autocast_dtype is not None:
        raise ValueError("The deterministic benchmark pilot is frozen to FP32")
    if batch_size is not None:
        raise ValueError("The benchmark pilot uses the frozen per-model batch sizes")
    output = Path(output_directory).resolve() if output_directory else context.runs_dir / "pilot" / _safe_name(chosen_profile)
    summary_path = Path(summary_json).resolve() if summary_json else context.locks_dir / "model_deterministic_pilot.json"
    default_batch = int(
        context.models_config.get("execution", {}).get("default_grid_batch_size", 32)
    )
    configured_models = context.models_config.get("models", {})
    planned = [
        {
            "model_id": model,
            "batch_size": int(
                batch_size
                or configured_models.get(model, {}).get(
                    "extraction_batch_size_rtx3090_fp32", default_batch
                )
            ),
            "repeat_a": output / model / "repeat_a.h5",
            "repeat_b": output / model / "repeat_b.h5",
            "resume": output / model / "resume.h5",
        }
        for model in selected
    ]
    plan = {
        "action": "features.pilot",
        "models": list(selected),
        "profile": chosen_profile,
        "device": device,
        "roi_count": roi_count,
        "atol": atol,
        "rtol": rtol,
        "output_directory": output,
        "summary_json": summary_path,
        "runs": planned,
    }
    if dry_run:
        return {"dry_run": True, **plan}
    permission = require_execution_permission(
        context.hold_path, action="deterministic feature pilot", override_hold=override_hold
    )
    release_integrity = require_release_integrity(context)
    completed_pilot_files = [
        Path(item[key])
        for item in planned
        for key in ("repeat_a", "repeat_b", "resume")
        if Path(item[key]).exists()
    ]
    if completed_pilot_files:
        raise FileExistsError(
            "Pilot requires fresh independent outputs; choose a new --output-dir. "
            f"Existing examples: {completed_pilot_files[:3]}"
        )
    verify_frozen_split(context.split_csv, context.split_lock)
    manifest_sha = _load_manifest_hash(context)
    context_inputs = execution_context_identities(context)
    all_records = load_roi_records(
        context.path("dataset", "metadata_csv"), context.path("dataset", "roi_root"), strict=True
    )
    records = sorted(all_records, key=lambda record: record.roi_uid)[:roi_count]
    pilot_roi_uids = [record.roi_uid for record in records]
    pilot_records_fingerprint = records_fingerprint(records)
    results: list[dict[str, Any]] = []
    for item in planned:
        model_id = str(item["model_id"])
        checkpoint = context.path("weights", model_id)
        retccl_source = context.path("weights", "root") / "RetCCL" / "ResNet.py" if model_id == "retccl" else None
        config = ExtractionConfig(
            batch_size=int(item["batch_size"]),
            device=device,
            transform_profile=chosen_profile,
            autocast_dtype=autocast_dtype,
            deterministic_algorithms=True,
        )
        repeat_runs = [
            ModelExtractionRun(model_id, checkpoint, Path(item[name]), retccl_source)
            for name in ("repeat_a", "repeat_b")
        ]
        extract_models_sequentially(
            records,
            repeat_runs,
            dataset_manifest_sha256=manifest_sha,
            config=config,
        )
        arrays_a = _feature_arrays(item["repeat_a"])
        arrays_b = _feature_arrays(item["repeat_b"])
        comparisons: dict[str, Any] = {}
        for strategy in STRATEGIES:
            difference = np.abs(arrays_a[strategy] - arrays_b[strategy])
            matches = bool(
                np.array_equal(arrays_a[strategy], arrays_b[strategy])
                if atol == 0 and rtol == 0
                else np.allclose(
                    arrays_a[strategy], arrays_b[strategy], atol=atol, rtol=rtol
                )
            )
            comparisons[strategy] = {
                "finite": bool(
                    np.isfinite(arrays_a[strategy]).all()
                    and np.isfinite(arrays_b[strategy]).all()
                ),
                "matches": matches,
                "max_abs_difference": float(difference.max(initial=0.0)),
            }

        # Exercise actual partial-file reopen/finalize semantics using the first
        # deterministic repeat as the payload.  No third inference is needed.
        repeat_validation = validate_feature_file(item["repeat_a"], require_complete=True)
        resume_path = Path(item["resume"])
        if not resume_path.exists():
            resume_store = ResumableFeatureStore(
                resume_path,
                records,
                get_model_spec(model_id).embedding_dim,
                repeat_validation["provenance"],
            )
            with resume_store:
                resume_store.write_row(
                    0,
                    center=arrays_a["center"][0],
                    mean=arrays_a["mean"][0],
                    max_=arrays_a["max"][0],
                )
            reopened = ResumableFeatureStore(
                resume_path,
                records,
                get_model_spec(model_id).embedding_dim,
                repeat_validation["provenance"],
            )
            with reopened:
                pending = reopened.incomplete_indices().tolist()
                if 0 in pending:
                    raise AssertionError("Completed pilot row was not preserved across reopen")
                for index in pending:
                    reopened.write_row(
                        index,
                        center=arrays_a["center"][index],
                        mean=arrays_a["mean"][index],
                        max_=arrays_a["max"][index],
                    )
            reopened.finalize()
        arrays_resume = _feature_arrays(resume_path)
        resume_matches = all(
            np.array_equal(arrays_a[strategy], arrays_resume[strategy])
            for strategy in STRATEGIES
        )
        status = "PASS" if resume_matches and all(
            value["finite"] and value["matches"] for value in comparisons.values()
        ) else "FAIL"
        results.append(
            {
                "model_id": model_id,
                "status": status,
                "batch_size": item["batch_size"],
                "representations": comparisons,
                "resume_reopen_bit_exact": resume_matches,
                "output_files": {
                    name: {
                        "path": str(Path(item[name]).resolve()),
                        "sha256": sha256_file(item[name]),
                    }
                    for name in ("repeat_a", "repeat_b", "resume")
                },
            }
        )
        del arrays_a, arrays_b, arrays_resume
    summary = {
        **plan,
        "permission": permission,
        "release_integrity": release_integrity,
        "identity_schema": "pairbst.stage_identity.v1",
        "input_identities": [
            *context_inputs,
            file_identity(
                context.locks_dir
                / "figshare_release_integrity"
                / "release_file_integrity_summary.json"
            ),
            file_identity(context.locks_dir / "figshare_release_integrity.lock.json"),
        ],
        "dataset_manifest_sha256": manifest_sha,
        "pilot_roi_uids": pilot_roi_uids,
        "ordered_records_fingerprint": pilot_records_fingerprint,
        "status": "PASS" if all(result["status"] == "PASS" for result in results) else "FAIL",
        "models": results,
    }
    _atomic_json(summary_path, summary)
    return summary


def _load_split_frame(split_csv: str | Path) -> pd.DataFrame:
    split = pd.read_csv(
        split_csv,
        dtype={column: "string" for column in REQUIRED_JOIN_COLUMNS},
        keep_default_na=False,
    )
    missing = sorted(set(REQUIRED_JOIN_COLUMNS).union({"fold"}).difference(split.columns))
    if missing:
        raise ValueError(f"Split CSV is missing required columns: {missing}")
    if split["roi_uid"].duplicated().any():
        raise ValueError("Split CSV contains duplicate roi_uid values")
    split["fold"] = pd.to_numeric(split["fold"], errors="raise").astype(int)
    if sorted(split["fold"].unique().tolist()) != [0, 1, 2]:
        raise ValueError("Split CSV must contain exactly folds 0, 1, and 2")
    membership = split.groupby("patient_uid", sort=False)["fold"].nunique()
    if (membership != 1).any():
        raise ValueError("Split CSV contains patients in more than one fold")
    return split


def align_feature_file(
    feature_path: str | Path,
    split_csv: str | Path,
    *,
    expected_model_name: str,
    expected_transform_profile: str,
    expected_manifest_sha256: str | None = None,
    expected_gpu_name: str | None = None,
    expected_cuda_version: str | None = None,
    expected_extraction_batch_size: int | None = None,
) -> FeatureAlignment:
    """Validate and join a canonical feature H5 to the split by ``roi_uid``."""

    path = Path(feature_path).resolve()
    spec = get_model_spec(expected_model_name)
    split = _load_split_frame(split_csv)
    validation = validate_feature_file(
        path,
        expected_rows=len(split),
        expected_dim=spec.embedding_dim,
        require_complete=True,
    )
    if validation["status"] != "complete":
        raise ValueError(f"Canonical feature file status is not complete: {path}")
    provenance = validation["provenance"]
    expected_identity = {
        "model_name": spec.name,
        "architecture": spec.architecture,
        "embedding_dim": spec.embedding_dim,
        "checkpoint_sha256": spec.checkpoint.sha256,
        "transform_profile": expected_transform_profile,
        "pooling_dtype": "float32",
        "encoder_autocast_dtype": None,
        "deterministic_algorithms": True,
        "tf32_allowed": False,
    }
    if expected_extraction_batch_size is not None:
        expected_identity["extraction_batch_size"] = expected_extraction_batch_size
    identity_mismatches = {
        key: {"observed": provenance.get(key), "expected": expected}
        for key, expected in expected_identity.items()
        if provenance.get(key) != expected
    }
    observed_transform = json.dumps(
        provenance.get("transform", {}), sort_keys=True, separators=(",", ":")
    )
    expected_transform = json.dumps(
        get_transform_spec(spec, expected_transform_profile).as_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    if observed_transform != expected_transform:
        identity_mismatches["transform"] = {
            "observed": provenance.get("transform"),
            "expected": get_transform_spec(spec, expected_transform_profile).as_dict(),
        }
    if spec.architecture_source_sha256 is not None and provenance.get(
        "architecture_source_sha256"
    ) != spec.architecture_source_sha256:
        identity_mismatches["architecture_source_sha256"] = {
            "observed": provenance.get("architecture_source_sha256"),
            "expected": spec.architecture_source_sha256,
        }
    hardware = provenance.get("hardware", {})
    if expected_gpu_name is not None and (
        not isinstance(hardware, Mapping)
        or hardware.get("gpu_name") != expected_gpu_name
        or not str(hardware.get("resolved_device", "")).startswith("cuda")
    ):
        identity_mismatches["hardware.gpu_name"] = {
            "observed": hardware,
            "expected": expected_gpu_name,
        }
    if expected_cuda_version is not None and (
        not isinstance(hardware, Mapping)
        or str(hardware.get("cuda_version")) != expected_cuda_version
    ):
        identity_mismatches["hardware.cuda_version"] = {
            "observed": hardware,
            "expected": expected_cuda_version,
        }
    if identity_mismatches:
        raise ValueError(
            f"Feature model/profile identity mismatch for {path}: {identity_mismatches}"
        )
    if expected_manifest_sha256 is not None:
        observed = str(provenance.get("dataset_manifest_sha256", "")).upper()
        if observed != expected_manifest_sha256.upper():
            raise ValueError(
                f"Feature dataset manifest hash mismatch: {observed} != {expected_manifest_sha256}"
            )
    with h5py.File(path, "r") as handle:
        metadata_group = handle.get("metadata")
        if metadata_group is None:
            raise ValueError(f"Feature file has no metadata group: {path}")
        missing = [column for column in REQUIRED_JOIN_COLUMNS if column not in metadata_group]
        if missing:
            raise ValueError(f"Feature H5 metadata is missing columns: {missing}")
        h5_metadata = pd.DataFrame(
            {column: _decode(metadata_group[column][:]) for column in REQUIRED_JOIN_COLUMNS}
        )
    if h5_metadata["roi_uid"].duplicated().any():
        raise ValueError("Feature H5 contains duplicate roi_uid values")
    split_ids = set(split["roi_uid"].astype(str))
    h5_ids = set(h5_metadata["roi_uid"].astype(str))
    if split_ids != h5_ids:
        missing_h5 = sorted(split_ids - h5_ids)[:5]
        extra_h5 = sorted(h5_ids - split_ids)[:5]
        raise ValueError(
            f"Feature/split roi_uid sets differ; missing={missing_h5}, extra={extra_h5}"
        )
    h5_metadata.insert(0, "_h5_row", np.arange(len(h5_metadata), dtype=np.int64))
    joined = split.merge(
        h5_metadata,
        on="roi_uid",
        how="left",
        validate="one_to_one",
        sort=False,
        suffixes=("_split", "_h5"),
    )
    for column in REQUIRED_JOIN_COLUMNS[1:]:
        left = joined[f"{column}_split"].astype(str)
        right = joined[f"{column}_h5"].astype(str)
        mismatch = left != right
        if mismatch.any():
            example = joined.loc[mismatch, "roi_uid"].iloc[0]
            raise ValueError(f"Feature H5 {column} disagrees with split at roi_uid={example}")
        joined[column] = left
        joined.drop(columns=[f"{column}_split", f"{column}_h5"], inplace=True)
    return FeatureAlignment(
        feature_path=path,
        split=joined,
        h5_rows=joined["_h5_row"].to_numpy(dtype=np.int64),
        provenance=provenance,
    )


def load_feature_matrix(alignment: FeatureAlignment, strategy: str) -> np.ndarray:
    """Read one representation and reorder it through the verified ROI join."""

    selected = normalize_strategy(strategy)
    with h5py.File(alignment.feature_path, "r") as handle:
        stored = np.asarray(handle[FEATURE_DATASETS[selected]][:], dtype=np.float32)
    matrix = np.ascontiguousarray(stored[alignment.h5_rows], dtype=np.float32)
    if matrix.shape[0] != len(alignment.split) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid aligned {selected} feature matrix: {matrix.shape}")
    return matrix


def _linear_probe_config(context: PipelineContext, device: str) -> LinearProbeConfig:
    config = context.protocol["linear_probe"]
    return LinearProbeConfig(
        protocol_id=str(context.protocol["protocol_id"]),
        seeds=tuple(int(value) for value in config["seeds"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        device=device,
        deterministic_algorithms=True,
        primary_metric_unit=str(config["primary_metric_unit"]),
        seed_aggregation=str(config["seed_aggregation"]),
        probability_ensemble_across_seeds=bool(
            config["probability_ensemble_across_seeds"]
        ),
        seed_sd_ddof=int(config["seed_sd_ddof"]),
    )


def run_classification(
    context: PipelineContext,
    *,
    models: Sequence[str] | None = None,
    profile: str | None = None,
    device: str = "cuda",
    feature_directory_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    override_hold: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected = select_models(models)
    chosen_profile = profile or str(context.protocol["feature_protocol"]["primary_transform"])
    if device != "cuda":
        raise ValueError("Benchmark linear probing is frozen to the audited CUDA device")
    features = feature_directory(context, chosen_profile, feature_directory_path)
    output = Path(output_directory).resolve() if output_directory else context.runs_dir / "classification" / _safe_name(chosen_profile)
    tasks = tuple(str(value) for value in context.protocol["tasks"])
    plan = {
        "action": "classify.run",
        "profile": chosen_profile,
        "features": features,
        "output_directory": output,
        "device": device,
        "grid": experiment_grid(selected, tasks),
    }
    if dry_run:
        return {"dry_run": True, **plan}
    permission = require_execution_permission(
        context.hold_path, action="linear-probe training", override_hold=override_hold
    )
    verify_frozen_split(context.split_csv, context.split_lock)
    manifest_sha = _load_manifest_hash(context)
    context_inputs = execution_context_identities(context)
    environment_lock = json.loads(
        (context.locks_dir / "environment.current.json").read_text(encoding="utf-8")
    )
    extraction_manifest_path = features / "extraction_manifest.json"
    extraction_manifest = verify_stage_manifest(
        extraction_manifest_path,
        expected_action="features.extract",
        expected_profile=chosen_profile,
    )
    extraction_outputs = {
        str(Path(identity["path"]).resolve()).casefold(): identity
        for identity in extraction_manifest["output_identities"]
    }
    expected_feature_paths = [feature_file_path(features, model_id) for model_id in selected]
    missing_bound_features = [
        str(path)
        for path in expected_feature_paths
        if str(path.resolve()).casefold() not in extraction_outputs
    ]
    if missing_bound_features:
        raise ValueError(
            f"Extraction manifest does not bind requested feature files: {missing_bound_features}"
        )
    feature_input_identities = [
        extraction_outputs[str(path.resolve()).casefold()] for path in expected_feature_paths
    ]
    protocol = _linear_probe_config(context, device)
    seed_fold_frames: list[pd.DataFrame] = []
    seed_oof_frames: list[pd.DataFrame] = []
    per_class_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    probability_dir = output / "classification_seed_oof_probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    for model_id in selected:
        spec = get_model_spec(model_id)
        alignment = align_feature_file(
            feature_file_path(features, model_id),
            context.split_csv,
            expected_model_name=model_id,
            expected_transform_profile=chosen_profile,
            expected_manifest_sha256=manifest_sha,
            expected_gpu_name=str(environment_lock["gpu"]),
            expected_cuda_version=str(environment_lock["cuda_version"]),
            expected_extraction_batch_size=int(
                context.models_config["models"][model_id][
                    "extraction_batch_size_rtx3090_fp32"
                ]
            ),
        )
        metadata = alignment.split
        for strategy in STRATEGIES:
            matrix = load_feature_matrix(alignment, strategy)
            for task in tasks:
                run_dir = output / model_id / strategy / task
                result = run_outer_cv_linear_probe(
                    matrix,
                    metadata[task].astype(str).to_numpy(),
                    metadata["fold"].to_numpy(dtype=int),
                    metadata["patient_uid"].astype(str).to_numpy(),
                    sample_ids=metadata["roi_uid"].astype(str).to_numpy(),
                    config=protocol,
                    output_dir=run_dir,
                )
                identifiers = {
                    "model_id": model_id,
                    "model": spec.manuscript_label,
                    "strategy": strategy,
                    "task": task,
                }
                seed_fold_frames.append(result.seed_fold_metrics.assign(**identifiers))
                seed_oof_frames.append(result.seed_oof_metrics.assign(**identifiers))
                per_class_frames.append(
                    pd.read_csv(
                        run_dir / "seed_oof_per_class_metrics.csv",
                        keep_default_na=False,
                    ).assign(**identifiers)
                )
                for seed_index, seed in enumerate(result.seeds.tolist()):
                    oof_frames.append(
                        pd.DataFrame(
                            {
                                "protocol_id": context.protocol["protocol_id"],
                                **{key: value for key, value in identifiers.items()},
                                "seed": int(seed),
                                "fold": metadata["fold"].to_numpy(dtype=int),
                                "roi_uid": metadata["roi_uid"].astype(str),
                                "wsi_uid": metadata["wsi_id"].astype(str),
                                "patient_uid": metadata["patient_uid"].astype(str),
                                "diagnosis_stratum": metadata["diagnosis"].astype(str),
                                "true_label": metadata[task].astype(str),
                                "predicted_label": result.seed_oof_predictions[
                                    seed_index
                                ].astype(str),
                            }
                        )
                    )
                _atomic_npz(
                    probability_dir / f"{model_id}__{strategy}__{task}.npz",
                    protocol_id=np.asarray(context.protocol["protocol_id"]),
                    model_id=np.asarray(model_id),
                    model=np.asarray(spec.manuscript_label),
                    strategy=np.asarray(strategy),
                    task=np.asarray(task),
                    seeds=result.seeds,
                    roi_uids=metadata["roi_uid"].astype(str).to_numpy(),
                    wsi_uids=metadata["wsi_id"].astype(str).to_numpy(),
                    patient_uids=metadata["patient_uid"].astype(str).to_numpy(),
                    fold_ids=metadata["fold"].to_numpy(dtype=np.int16),
                    true_labels=metadata[task].astype(str).to_numpy(),
                    class_names=np.asarray(result.classes, dtype=str),
                    probabilities=result.seed_oof_probabilities,
                    predictions=result.seed_oof_predictions_encoded,
                    predicted_labels=np.asarray(result.seed_oof_predictions, dtype=str),
                    probability_ensemble_across_seeds=np.asarray(False),
                )
            del matrix
    seed_fold_metrics = pd.concat(seed_fold_frames, ignore_index=True)
    seed_oof_metrics = pd.concat(seed_oof_frames, ignore_index=True)
    per_class_metrics = pd.concat(per_class_frames, ignore_index=True)
    oof = pd.concat(oof_frames, ignore_index=True)
    expected_systems = len(selected) * len(STRATEGIES) * len(tasks)
    expected_samples = len(pd.read_csv(context.split_csv))
    expected_seeds = len(protocol.seeds)
    if len(seed_oof_metrics) != expected_systems * expected_seeds:
        raise ValueError("Canonical classification seed OOF metric count is incomplete")
    if len(seed_fold_metrics) != expected_systems * expected_seeds * 3:
        raise ValueError("Canonical classification seed-fold audit count is incomplete")
    if len(oof) != expected_systems * expected_seeds * expected_samples:
        raise ValueError("Canonical classification seed OOF prediction count is incomplete")
    summary_keys = [
        "protocol_id", "model_id", "model", "strategy", "task", "class_id", "class_name"
    ]
    per_class_summary = (
        per_class_metrics.groupby(summary_keys, as_index=False, sort=False)
        .agg(
            precision_mean=("precision", "mean"),
            precision_sd=("precision", lambda values: values.std(ddof=1)),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", lambda values: values.std(ddof=1)),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", lambda values: values.std(ddof=1)),
            roi_support=("roi_support", "first"),
            patient_support=("patient_support", "first"),
            n_seeds=("seed", "nunique"),
        )
    )
    paths = {
        "seed_oof_metrics": _atomic_csv(
            output / "classification_seed_oof_metrics.csv", seed_oof_metrics
        ),
        "seed_fold_metrics": _atomic_csv(
            output / "classification_seed_fold_metrics.csv", seed_fold_metrics
        ),
        "seed_oof_predictions": _atomic_csv_gzip(
            output / "classification_seed_oof_predictions.csv.gz", oof
        ),
        "per_class_seed_oof": _atomic_csv(
            output / "classification_per_class_seed_oof.csv", per_class_metrics
        ),
        "per_class_seed_summary": _atomic_csv(
            output / "classification_per_class_seed_summary.csv", per_class_summary
        ),
    }
    manifest = {
        **plan,
        "permission": permission,
        "protocol_id": context.protocol["protocol_id"],
        "estimator_id": "complete_oof_per_seed_metric_mean_sd",
        "primary_metric_unit": protocol.primary_metric_unit,
        "seed_aggregation": protocol.seed_aggregation,
        "probability_ensemble_across_seeds": False,
        "seed_sd_ddof": protocol.seed_sd_ddof,
        "expected_counts": {
            "systems": expected_systems,
            "seed_oof_metrics": len(seed_oof_metrics),
            "seed_fold_metrics": len(seed_fold_metrics),
            "seed_oof_predictions": len(oof),
            "per_class_seed_oof": len(per_class_metrics),
            "per_class_seed_summary": len(per_class_summary),
        },
        "outputs": paths,
        "identity_schema": "pairbst.stage_identity.v1",
        "input_stage_manifests": [file_identity(extraction_manifest_path)],
        "input_identities": [*feature_input_identities, *context_inputs],
        "output_identities": collect_output_identities(
            output, excluded_names=("classification_manifest.json",)
        ),
    }
    _atomic_json(output / "classification_manifest.json", manifest)
    return manifest


def run_retrieval(
    context: PipelineContext,
    *,
    models: Sequence[str] | None = None,
    profile: str | None = None,
    feature_directory_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    query_chunk_size: int = 256,
    override_hold: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected = select_models(models)
    chosen_profile = profile or str(context.protocol["feature_protocol"]["primary_transform"])
    features = feature_directory(context, chosen_profile, feature_directory_path)
    output = Path(output_directory).resolve() if output_directory else context.runs_dir / "retrieval" / _safe_name(chosen_profile)
    tasks = tuple(str(value) for value in context.protocol["tasks"])
    ks = tuple(int(value) for value in context.protocol["retrieval"]["k"])
    plan = {
        "action": "retrieval.run",
        "profile": chosen_profile,
        "features": features,
        "output_directory": output,
        "ks": list(ks),
        "grid": experiment_grid(selected, tasks),
    }
    if dry_run:
        return {"dry_run": True, **plan}
    permission = require_execution_permission(
        context.hold_path, action="patient-disjoint retrieval", override_hold=override_hold
    )
    verify_frozen_split(context.split_csv, context.split_lock)
    manifest_sha = _load_manifest_hash(context)
    context_inputs = execution_context_identities(context)
    environment_lock = json.loads(
        (context.locks_dir / "environment.current.json").read_text(encoding="utf-8")
    )
    extraction_manifest_path = features / "extraction_manifest.json"
    extraction_manifest = verify_stage_manifest(
        extraction_manifest_path,
        expected_action="features.extract",
        expected_profile=chosen_profile,
    )
    extraction_outputs = {
        str(Path(identity["path"]).resolve()).casefold(): identity
        for identity in extraction_manifest["output_identities"]
    }
    expected_feature_paths = [feature_file_path(features, model_id) for model_id in selected]
    missing_bound_features = [
        str(path)
        for path in expected_feature_paths
        if str(path.resolve()).casefold() not in extraction_outputs
    ]
    if missing_bound_features:
        raise ValueError(
            f"Extraction manifest does not bind requested feature files: {missing_bound_features}"
        )
    feature_input_identities = [
        extraction_outputs[str(path.resolve()).casefold()] for path in expected_feature_paths
    ]
    fold_frames: list[pd.DataFrame] = []
    pooled_frames: list[pd.DataFrame] = []
    query_frames: list[pd.DataFrame] = []
    for model_id in selected:
        spec = get_model_spec(model_id)
        alignment = align_feature_file(
            feature_file_path(features, model_id),
            context.split_csv,
            expected_model_name=model_id,
            expected_transform_profile=chosen_profile,
            expected_manifest_sha256=manifest_sha,
            expected_gpu_name=str(environment_lock["gpu"]),
            expected_cuda_version=str(environment_lock["cuda_version"]),
            expected_extraction_batch_size=int(
                context.models_config["models"][model_id][
                    "extraction_batch_size_rtx3090_fp32"
                ]
            ),
        )
        metadata = alignment.split
        for strategy in STRATEGIES:
            matrix = load_feature_matrix(alignment, strategy)
            for task in tasks:
                run_dir = output / model_id / strategy / task
                result = run_patient_disjoint_cv_retrieval(
                    matrix,
                    metadata[task].astype(str).to_numpy(),
                    metadata["fold"].to_numpy(dtype=int),
                    metadata["patient_uid"].astype(str).to_numpy(),
                    sample_ids=metadata["roi_uid"].astype(str).to_numpy(),
                    ks=ks,
                    query_chunk_size=query_chunk_size,
                    output_dir=run_dir,
                )
                identifiers = {
                    "model_id": model_id,
                    "model": spec.manuscript_label,
                    "strategy": strategy,
                    "task": task,
                }
                fold_frames.append(result.fold_metrics.assign(**identifiers))
                pooled_frames.append(result.pooled_metrics.assign(**identifiers))
                query = result.per_query_metrics.assign(**identifiers)
                strata = metadata.set_index("roi_uid")["diagnosis"].astype(str)
                query["diagnosis_stratum"] = query["query_id"].astype(str).map(strata)
                if query["diagnosis_stratum"].isna().any():
                    raise ValueError("Retrieval query could not be joined to diagnosis strata")
                query_frames.append(query)
            del matrix
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    pooled_metrics = pd.concat(pooled_frames, ignore_index=True)
    per_query = pd.concat(query_frames, ignore_index=True)
    paths = {
        "fold_metrics": _atomic_csv(output / "retrieval_fold_metrics.csv", fold_metrics),
        "pooled_metrics": _atomic_csv(output / "retrieval_pooled_metrics.csv", pooled_metrics),
        "per_query_metrics": _atomic_csv(output / "retrieval_per_query_metrics.csv", per_query),
    }
    manifest = {
        **plan,
        "permission": permission,
        "outputs": paths,
        "identity_schema": "pairbst.stage_identity.v1",
        "input_stage_manifests": [file_identity(extraction_manifest_path)],
        "input_identities": [*feature_input_identities, *context_inputs],
        "output_identities": collect_output_identities(
            output, excluded_names=("retrieval_manifest.json",)
        ),
    }
    _atomic_json(output / "retrieval_manifest.json", manifest)
    return manifest


def _validate_patient_stable_strata(frame: pd.DataFrame) -> None:
    counts = frame.groupby("patient_uid", sort=False)["bootstrap_stratum"].nunique()
    if (counts != 1).any():
        examples = counts[counts != 1].index.astype(str).tolist()[:5]
        raise ValueError(f"Bootstrap strata must be patient-stable; examples: {examples}")


def patient_task_label_signatures(
    labels: Sequence[Any] | np.ndarray,
    patient_ids: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Return a patient-stable signature of all unique task labels per patient."""

    label_array = np.asarray(labels).astype(str)
    patient_array = np.asarray(patient_ids).astype(str)
    if label_array.ndim != 1 or patient_array.shape != label_array.shape or not label_array.size:
        raise ValueError("labels and patient_ids must be aligned, non-empty 1D arrays")
    table = pd.DataFrame({"patient": patient_array, "label": label_array})
    signatures = table.groupby("patient", sort=False)["label"].agg(
        lambda values: json.dumps(
            sorted(set(values.astype(str).tolist())), ensure_ascii=False, separators=(",", ":")
        )
    )
    return table["patient"].map(signatures).to_numpy(dtype=object)


def classification_bootstrap_ci_by_seed(
    oof: pd.DataFrame,
    *,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute patient-cluster CIs separately for each complete seed OOF.

    The stratum for a patient is the sorted set of every ROI label that patient
    has for the current task.  This makes multi-label secondary-task patients
    valid clusters while preserving rare task labels in bootstrap replicates.
    Seed-specific intervals are returned without any cross-seed collapsing.
    """

    required = {
        "protocol_id", "model_id", "model", "strategy", "task", "seed",
        "roi_uid", "patient_uid",
        "diagnosis_stratum", "true_label", "predicted_label",
    }
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise ValueError(f"Classification OOF file is missing columns: {missing}")
    rows: list[pd.DataFrame] = []
    distributions: dict[str, np.ndarray] = {}
    protocols = set(oof["protocol_id"].astype(str))
    if protocols != {CANONICAL_PROTOCOL_ID}:
        raise ValueError(
            "Classification OOF input is not the canonical independent-seed protocol."
        )
    group_columns = ["protocol_id", "model_id", "model", "strategy", "task", "seed"]
    for keys, group in oof.groupby(group_columns, sort=False):
        if group["roi_uid"].duplicated().any():
            raise ValueError(f"Duplicate ROI predictions in system {keys}")
        group = group.copy()
        group["bootstrap_stratum"] = patient_task_label_signatures(
            group["true_label"], group["patient_uid"]
        )
        _validate_patient_stable_strata(group)
        labels = sorted(group["true_label"].astype(str).unique().tolist())
        linear_seed = int(keys[-1])
        derived_seed = int(
            np.random.SeedSequence([int(seed), linear_seed])
            .generate_state(1, dtype=np.uint32)[0]
        )
        result = cluster_bootstrap_classification(
            group["true_label"].astype(str).to_numpy(),
            group["predicted_label"].astype(str).to_numpy(),
            group["patient_uid"].astype(str).to_numpy(),
            labels=labels,
            strata=group["bootstrap_stratum"].astype(str).to_numpy(),
            stratified=True,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=derived_seed,
        )
        protocol_id, model_id, model, strategy, task, linear_seed = keys
        summary = result.summary_frame()
        summary.insert(0, "seed", int(linear_seed))
        summary.insert(0, "task", task)
        summary.insert(0, "strategy", strategy)
        summary.insert(0, "model", model)
        summary.insert(0, "model_id", model_id)
        summary.insert(0, "protocol_id", protocol_id)
        summary["n_samples"] = int(len(group))
        summary["bootstrap_seed"] = derived_seed
        summary["strata_definition"] = "patient_unique_task_label_signature"
        rows.append(summary)
        for metric, values in result.distributions.items():
            distributions[
                "__".join(_safe_name(value) for value in (*keys, metric))
            ] = values
    return pd.concat(rows, ignore_index=True), distributions


def _align_prediction_systems(
    left: pd.DataFrame, right: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "roi_uid", "patient_uid", "diagnosis_stratum", "true_label", "predicted_label"
    ]
    merged = left[columns].merge(
        right[columns],
        on="roi_uid",
        how="inner",
        validate="one_to_one",
        suffixes=("_a", "_b"),
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Paired systems do not contain identical roi_uid sets")
    for column in ("patient_uid", "diagnosis_stratum", "true_label"):
        if not (merged[f"{column}_a"].astype(str) == merged[f"{column}_b"].astype(str)).all():
            raise ValueError(f"Paired systems disagree on {column}")
    return merged


def classification_paired_comparisons(
    oof: pd.DataFrame,
    comparisons_config: Mapping[str, Any],
    *,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    families = comparisons_config.get("families", {})
    observed_models = oof["model_id"].drop_duplicates().astype(str).tolist()
    observed_strategies = oof["strategy"].drop_duplicates().astype(str).tolist()
    observed_tasks = oof["task"].drop_duplicates().astype(str).tolist()
    for family, config in families.items():
        metrics = tuple(str(value) for value in config.get("metrics", ("macro_f1", "balanced_accuracy")))
        if family == "sampling_within_model":
            reference = normalize_strategy(str(config["reference"]))
            alternatives = [normalize_strategy(str(value)) for value in config["alternatives"]]
            pairs = [
                (model, alternative, model, reference, task)
                for model in observed_models
                for task in observed_tasks
                for alternative in alternatives
            ]
        elif family == "model_within_sampling":
            reference_model = normalize_model_id(str(config["reference"]))
            alternative_models = [normalize_model_id(str(value)) for value in config["alternatives"]]
            pairs = [
                (alternative, strategy, reference_model, strategy, task)
                for strategy in observed_strategies
                for task in observed_tasks
                for alternative in alternative_models
            ]
        else:
            continue
        for model_a, strategy_a, model_b, strategy_b, task in pairs:
            left = oof[
                (oof["model_id"] == model_a)
                & (oof["strategy"] == strategy_a)
                & (oof["task"] == task)
            ]
            right = oof[
                (oof["model_id"] == model_b)
                & (oof["strategy"] == strategy_b)
                & (oof["task"] == task)
            ]
            if left.empty or right.empty:
                raise ValueError(
                    f"Missing predeclared comparison system: {model_a}/{strategy_a} vs "
                    f"{model_b}/{strategy_b}/{task}"
                )
            paired = _align_prediction_systems(left, right)
            labels = sorted(paired["true_label_a"].astype(str).unique().tolist())
            bootstrap_strata = patient_task_label_signatures(
                paired["true_label_a"], paired["patient_uid_a"]
            )
            comparison = paired_model_comparison(
                paired["true_label_a"].astype(str).to_numpy(),
                paired["predicted_label_a"].astype(str).to_numpy(),
                paired["predicted_label_b"].astype(str).to_numpy(),
                paired["patient_uid_a"].astype(str).to_numpy(),
                model_a=f"{model_a}:{strategy_a}",
                model_b=f"{model_b}:{strategy_b}",
                task=task,
                family=family,
                labels=labels,
                strata=bootstrap_strata,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )
            comparison = comparison[comparison["metric"].isin(metrics)].copy()
            comparison["model_id_a"] = model_a
            comparison["strategy_a"] = strategy_a
            comparison["model_id_b"] = model_b
            comparison["strategy_b"] = strategy_b
            comparison["strata_definition"] = "patient_unique_task_label_signature"
            frames.append(comparison)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return apply_holm_correction(
        combined,
        family_columns=("family", "task", "metric"),
    )


def retrieval_bootstrap_ci(
    per_query: pd.DataFrame,
    *,
    metrics: Sequence[str],
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    required = {
        "model_id", "model", "strategy", "task", "query_id", "patient_id",
        "diagnosis_stratum", "k", *metrics,
    }
    missing = sorted(required.difference(per_query.columns))
    if missing:
        raise ValueError(f"Retrieval per-query file is missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    group_columns = ["model_id", "model", "strategy", "task", "k"]
    for keys, group in per_query.groupby(group_columns, sort=False):
        bootstrap_strata = patient_task_label_signatures(
            group["true_label"], group["patient_id"]
        )
        for metric in metrics:
            result = cluster_bootstrap_mean(
                group[metric].to_numpy(dtype=float),
                group["patient_id"].astype(str).to_numpy(),
                strata=bootstrap_strata,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )
            rows.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "metric": metric,
                    **{key: value for key, value in result.items() if key != "distribution"},
                    "strata_definition": "patient_unique_task_label_signature",
                }
            )
    return pd.DataFrame(rows)


def _align_retrieval_systems(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    columns = ["query_id", "patient_id", "diagnosis_stratum", "true_label"]
    metric_columns = [
        "precision_at_k", "recall_at_k", "hit_at_k", "average_precision_at_k",
        "majority_vote_correct",
    ]
    merged = left[columns + metric_columns].merge(
        right[columns + metric_columns],
        on="query_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_a", "_b"),
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Paired retrieval systems do not contain identical query IDs")
    for column in ("patient_id", "diagnosis_stratum", "true_label"):
        if not (merged[f"{column}_a"].astype(str) == merged[f"{column}_b"].astype(str)).all():
            raise ValueError(f"Paired retrieval systems disagree on {column}")
    return merged


def retrieval_paired_comparisons(
    per_query: pd.DataFrame,
    comparisons_config: Mapping[str, Any],
    *,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Run configured retrieval comparisons.

    A family is included only when it declares ``retrieval_metrics``.  This
    prevents silently multiplying hundreds of unplanned tests.
    """

    frames: list[pd.DataFrame] = []
    families = comparisons_config.get("families", {})
    models = per_query["model_id"].drop_duplicates().astype(str).tolist()
    strategies = per_query["strategy"].drop_duplicates().astype(str).tolist()
    tasks = per_query["task"].drop_duplicates().astype(str).tolist()
    ks = sorted(per_query["k"].astype(int).unique().tolist())
    for family, config in families.items():
        metrics = tuple(str(value) for value in config.get("retrieval_metrics", ()))
        if not metrics:
            continue
        if family == "sampling_within_model":
            reference = normalize_strategy(str(config["reference"]))
            alternatives = [normalize_strategy(str(value)) for value in config["alternatives"]]
            pairs = [(m, a, m, reference) for m in models for a in alternatives]
        elif family == "model_within_sampling":
            reference_model = normalize_model_id(str(config["reference"]))
            alternatives = [normalize_model_id(str(value)) for value in config["alternatives"]]
            pairs = [(a, s, reference_model, s) for s in strategies for a in alternatives]
        else:
            continue
        for model_a, strategy_a, model_b, strategy_b in pairs:
            for task in tasks:
                for k in ks:
                    left = per_query[
                        (per_query["model_id"] == model_a)
                        & (per_query["strategy"] == strategy_a)
                        & (per_query["task"] == task)
                        & (per_query["k"].astype(int) == k)
                    ]
                    right = per_query[
                        (per_query["model_id"] == model_b)
                        & (per_query["strategy"] == strategy_b)
                        & (per_query["task"] == task)
                        & (per_query["k"].astype(int) == k)
                    ]
                    paired = _align_retrieval_systems(left, right)
                    bootstrap_strata = patient_task_label_signatures(
                        paired["true_label_a"], paired["patient_id_a"]
                    )
                    for metric in metrics:
                        comparison = paired_query_metric_comparison(
                            paired[f"{metric}_a"].to_numpy(dtype=float),
                            paired[f"{metric}_b"].to_numpy(dtype=float),
                            paired["patient_id_a"].astype(str).to_numpy(),
                            model_a=f"{model_a}:{strategy_a}",
                            model_b=f"{model_b}:{strategy_b}",
                            task=task,
                            metric=metric,
                            family=family,
                            strata=bootstrap_strata,
                            n_bootstrap=n_bootstrap,
                            confidence_level=confidence_level,
                            seed=seed,
                        )
                        comparison["k"] = k
                        comparison["model_id_a"] = model_a
                        comparison["strategy_a"] = strategy_a
                        comparison["model_id_b"] = model_b
                        comparison["strategy_b"] = strategy_b
                        comparison["strata_definition"] = "patient_unique_task_label_signature"
                        frames.append(comparison)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return apply_holm_correction(
        combined,
        family_columns=("family", "task", "metric", "k"),
    )


def run_statistics(
    context: PipelineContext,
    *,
    profile: str | None = None,
    classification_directory: str | Path | None = None,
    retrieval_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    n_bootstrap: int | None = None,
    retrieval_ci_metrics: Sequence[str] = ("average_precision_at_k", "majority_vote_correct"),
    override_hold: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    chosen_profile = profile or str(context.protocol["feature_protocol"]["primary_transform"])
    classification_dir = Path(classification_directory).resolve() if classification_directory else context.runs_dir / "classification" / _safe_name(chosen_profile)
    retrieval_dir = Path(retrieval_directory).resolve() if retrieval_directory else context.runs_dir / "retrieval" / _safe_name(chosen_profile)
    output = Path(output_directory).resolve() if output_directory else context.runs_dir / "statistics" / _safe_name(chosen_profile)
    statistics_config = context.protocol["statistics"]
    frozen_iterations = int(statistics_config["bootstrap_iterations"])
    if n_bootstrap is not None and int(n_bootstrap) != frozen_iterations:
        raise ValueError(
            f"Primary statistics requires exactly {frozen_iterations} bootstrap iterations"
        )
    iterations = frozen_iterations
    confidence = float(statistics_config["confidence_level"])
    seed = int(context.protocol["outer_cv"]["assignment_seed"])
    plan = {
        "action": "statistics.run",
        "profile": chosen_profile,
        "classification_directory": classification_dir,
        "retrieval_directory": retrieval_dir,
        "output_directory": output,
        "n_bootstrap": iterations,
        "confidence_level": confidence,
        "bootstrap_unit": "patient",
        "classification_bootstrap_scope": "separately_by_seed",
        "classification_paired_significance": "disabled",
        "strata_definition": "patient_unique_task_label_signature",
        "retrieval_ci_metrics": list(retrieval_ci_metrics),
    }
    if dry_run:
        return {"dry_run": True, **plan}
    permission = require_execution_permission(
        context.hold_path, action="bootstrap statistics", override_hold=override_hold
    )
    verify_frozen_split(context.split_csv, context.split_lock)
    classification_manifest_path = classification_dir / "classification_manifest.json"
    retrieval_manifest_path = retrieval_dir / "retrieval_manifest.json"
    classification_manifest = verify_stage_manifest(
        classification_manifest_path,
        expected_action="classify.run",
        expected_profile=chosen_profile,
    )
    retrieval_manifest = verify_stage_manifest(
        retrieval_manifest_path,
        expected_action="retrieval.run",
        expected_profile=chosen_profile,
    )
    verify_compatible_feature_lineage(classification_manifest, retrieval_manifest)
    if classification_manifest.get("protocol_id") != CANONICAL_PROTOCOL_ID or bool(
        classification_manifest.get("probability_ensemble_across_seeds", True)
    ):
        raise ValueError("Statistics requires canonical independent-seed classification output")
    oof_path = classification_dir / "classification_seed_oof_predictions.csv.gz"
    per_query_path = retrieval_dir / "retrieval_per_query_metrics.csv"
    oof_identity = require_manifest_bound_paths(
        classification_manifest,
        [oof_path],
        identity_field="output_identities",
        label="classification manifest",
    )[0]
    per_query_identity = require_manifest_bound_paths(
        retrieval_manifest,
        [per_query_path],
        identity_field="output_identities",
        label="retrieval manifest",
    )[0]
    context_inputs = execution_context_identities(context)
    oof = pd.read_csv(oof_path, keep_default_na=False)
    per_query = pd.read_csv(per_query_path, keep_default_na=False)
    classification_ci, distributions = classification_bootstrap_ci_by_seed(
        oof,
        n_bootstrap=iterations,
        confidence_level=confidence,
        seed=seed,
    )
    retrieval_ci = retrieval_bootstrap_ci(
        per_query,
        metrics=retrieval_ci_metrics,
        n_bootstrap=iterations,
        confidence_level=confidence,
        seed=seed,
    )
    paired_retrieval = retrieval_paired_comparisons(
        per_query,
        context.comparisons,
        n_bootstrap=iterations,
        confidence_level=confidence,
        seed=seed,
    )
    output.mkdir(parents=True, exist_ok=True)
    distribution_path = output / "classification_bootstrap_distributions.npz"
    temporary_distribution = output / ".classification_bootstrap_distributions.npz.tmp"
    with temporary_distribution.open("wb") as handle:
        np.savez_compressed(handle, **distributions)
    os.replace(temporary_distribution, distribution_path)
    paths = {
        "classification_ci_by_seed": _atomic_csv(
            output / "classification_patient_cluster_ci_by_seed.csv", classification_ci
        ),
        "classification_distributions": distribution_path,
        "retrieval_ci": _atomic_csv(output / "retrieval_patient_cluster_ci.csv", retrieval_ci),
        "retrieval_comparisons": _atomic_csv(output / "retrieval_paired_comparisons_holm.csv", paired_retrieval),
    }
    manifest = {
        **plan,
        "permission": permission,
        "outputs": paths,
        "identity_schema": "pairbst.stage_identity.v1",
        "input_stage_manifests": [
            file_identity(classification_manifest_path),
            file_identity(retrieval_manifest_path),
        ],
        "input_identities": [oof_identity, per_query_identity, *context_inputs],
        "output_identities": collect_output_identities(
            output, excluded_names=("statistics_manifest.json",)
        ),
    }
    _atomic_json(output / "statistics_manifest.json", manifest)
    return manifest


def _require_complete_grid(
    frame: pd.DataFrame,
    *,
    name: str,
    expected_k: Sequence[int] | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> None:
    unit_column = "seed" if expected_seeds is not None else "held_fold"
    required = {"model_id", "strategy", "task", unit_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    observed = set(
        frame[["model_id", "strategy", "task", unit_column]]
        .astype({unit_column: int})
        .itertuples(index=False, name=None)
    )
    expected = {
        (model, strategy, task, fold)
        for model in MODEL_IDS
        for strategy in STRATEGIES
        for task in ("diagnosis", "differentiation", "growth_pattern")
        for fold in (
            tuple(int(value) for value in expected_seeds)
            if expected_seeds is not None
            else (0, 1, 2)
        )
    }
    if observed != expected:
        missing_cells = sorted(expected - observed)[:8]
        extra_cells = sorted(observed - expected)[:8]
        raise ValueError(f"{name} grid is incomplete; missing={missing_cells}, extra={extra_cells}")
    if expected_seeds is not None:
        keys = ["model_id", "strategy", "task", "seed"]
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} contains duplicate system/seed rows")
        expected_rows = len(MODEL_IDS) * len(STRATEGIES) * 3 * len(expected_seeds)
        if len(frame) != expected_rows:
            raise ValueError(f"{name} has {len(frame)} rows; expected {expected_rows}")
    if expected_k is not None:
        if "k" not in frame.columns:
            raise ValueError(f"{name} is missing column: k")
        required_k = {int(value) for value in expected_k}
        observed_k = set(pd.to_numeric(frame["k"], errors="raise").astype(int))
        if observed_k != required_k:
            raise ValueError(
                f"{name} K values differ; observed={sorted(observed_k)}, "
                f"expected={sorted(required_k)}"
            )
        keys = ["model_id", "strategy", "task", "held_fold", "k"]
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} contains duplicate fold/K rows")
        expected_rows = len(MODEL_IDS) * len(STRATEGIES) * 3 * 3 * len(required_k)
        if len(frame) != expected_rows:
            raise ValueError(f"{name} has {len(frame)} rows; expected {expected_rows}")


def build_report(
    context: PipelineContext,
    *,
    profile: str | None = None,
    classification_directory: str | Path | None = None,
    retrieval_directory: str | Path | None = None,
    statistics_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    override_hold: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    chosen_profile = profile or str(context.protocol["feature_protocol"]["primary_transform"])
    classification_dir = Path(classification_directory).resolve() if classification_directory else context.runs_dir / "classification" / _safe_name(chosen_profile)
    retrieval_dir = Path(retrieval_directory).resolve() if retrieval_directory else context.runs_dir / "retrieval" / _safe_name(chosen_profile)
    statistics_dir = Path(statistics_directory).resolve() if statistics_directory else context.runs_dir / "statistics" / _safe_name(chosen_profile)
    if output_directory is not None:
        output = Path(output_directory).resolve()
    elif chosen_profile == str(context.protocol["feature_protocol"]["primary_transform"]):
        output = context.final_dir
    else:
        output = context.final_dir / "sensitivity" / _safe_name(chosen_profile)
    plan = {
        "action": "report.build",
        "profile": chosen_profile,
        "classification_directory": classification_dir,
        "retrieval_directory": retrieval_dir,
        "statistics_directory": statistics_dir,
        "output_directory": output,
        "expected_files": [
            output / "table5_manuscript.csv",
            output / "table5_manuscript.md",
            output / "table5_manuscript.tex",
            output / "retrieval_primary.csv",
            output / "per_class_f1.csv",
            output / "confusion_matrices",
            output / "provenance_table.csv",
            output / "final_results_manifest.json",
        ],
    }
    if dry_run:
        return {"dry_run": True, **plan}
    permission = require_execution_permission(
        context.hold_path, action="final result generation", override_hold=override_hold
    )
    verify_frozen_split(context.split_csv, context.split_lock)
    classification_manifest_path = classification_dir / "classification_manifest.json"
    retrieval_manifest_path = retrieval_dir / "retrieval_manifest.json"
    statistics_manifest_path = statistics_dir / "statistics_manifest.json"
    classification_manifest = verify_stage_manifest(
        classification_manifest_path,
        expected_action="classify.run",
        expected_profile=chosen_profile,
    )
    retrieval_manifest = verify_stage_manifest(
        retrieval_manifest_path,
        expected_action="retrieval.run",
        expected_profile=chosen_profile,
    )
    statistics_manifest = verify_stage_manifest(
        statistics_manifest_path,
        expected_action="statistics.run",
        expected_profile=chosen_profile,
    )
    if classification_manifest.get("protocol_id") != CANONICAL_PROTOCOL_ID or bool(
        classification_manifest.get("probability_ensemble_across_seeds", True)
    ):
        raise ValueError("Report requires canonical independent-seed classification output")
    if (
        int(statistics_manifest.get("n_bootstrap", -1))
        != int(context.protocol["statistics"]["bootstrap_iterations"])
        or float(statistics_manifest.get("confidence_level", -1))
        != float(context.protocol["statistics"]["confidence_level"])
        or statistics_manifest.get("bootstrap_unit") != "patient"
        or statistics_manifest.get("strata_definition")
        != "patient_unique_task_label_signature"
        or statistics_manifest.get("classification_bootstrap_scope")
        != "separately_by_seed"
    ):
        raise ValueError("Statistics manifest does not match the frozen primary protocol")
    verify_compatible_feature_lineage(classification_manifest, retrieval_manifest)
    require_exact_stage_manifest_inputs(
        statistics_manifest,
        [classification_manifest_path, retrieval_manifest_path],
        label="statistics manifest",
    )
    classification_seed_path = classification_dir / "classification_seed_oof_metrics.csv"
    per_class_path = classification_dir / "classification_per_class_seed_summary.csv"
    retrieval_fold_path = retrieval_dir / "retrieval_fold_metrics.csv"
    classification_ci_path = (
        statistics_dir / "classification_patient_cluster_ci_by_seed.csv"
    )
    consumed_input_identities = [
        *require_manifest_bound_paths(
            classification_manifest,
            [classification_seed_path, per_class_path],
            identity_field="output_identities",
            label="classification manifest",
        ),
        *require_manifest_bound_paths(
            retrieval_manifest,
            [retrieval_fold_path],
            identity_field="output_identities",
            label="retrieval manifest",
        ),
        *require_manifest_bound_paths(
            statistics_manifest,
            [classification_ci_path],
            identity_field="output_identities",
            label="statistics manifest",
        ),
    ]
    context_inputs = execution_context_identities(context)
    classification_seed = pd.read_csv(classification_seed_path)
    per_class = pd.read_csv(
        per_class_path,
        keep_default_na=False,
    )
    retrieval_fold = pd.read_csv(retrieval_fold_path)
    classification_ci = pd.read_csv(classification_ci_path)
    if (
        len(classification_ci) != len(MODEL_IDS) * len(STRATEGIES) * 3 * 5 * 4
        or set(classification_ci.get("protocol_id", pd.Series(dtype=str)).astype(str))
        != {CANONICAL_PROTOCOL_ID}
    ):
        raise ValueError("Seed-specific classification CI output is incomplete or incompatible")
    _require_complete_grid(
        classification_seed,
        name="classification seed OOF metrics",
        expected_seeds=tuple(int(value) for value in context.protocol["linear_probe"]["seeds"]),
    )
    _require_complete_grid(
        retrieval_fold,
        name="retrieval fold metrics",
        expected_k=tuple(int(value) for value in context.protocol["retrieval"]["k"]),
    )
    outputs = write_final_results_bundle(
        classification_seed,
        retrieval_fold,
        output,
        retrieval_k_order=tuple(int(value) for value in context.protocol["retrieval"]["k"]),
    )
    per_class_columns = [
        "protocol_id", "model_id", "model", "strategy", "task", "class_id",
        "class_name", "precision_mean", "precision_sd", "recall_mean", "recall_sd",
        "f1_mean", "f1_sd", "roi_support", "patient_support", "n_seeds",
    ]
    missing_per_class = sorted(set(per_class_columns).difference(per_class.columns))
    if missing_per_class:
        raise ValueError(f"Per-class metrics are missing columns: {missing_per_class}")
    outputs["per_class"] = {
        "csv": _atomic_csv(output / "per_class_f1.csv", per_class[per_class_columns])
    }
    outputs["classification_ci_by_seed"] = {
        "csv": _atomic_csv(
            output / "classification_patient_cluster_ci_by_seed.csv", classification_ci
        )
    }

    confusion_output = output / "confusion_matrices"
    confusion_output.mkdir(parents=True, exist_ok=True)
    confusion_paths: list[Path] = []
    for model_id in MODEL_IDS:
        for strategy in STRATEGIES:
            for task in ("diagnosis", "differentiation", "growth_pattern"):
                source_dir = classification_dir / model_id / strategy / task
                stem = f"{model_id}__{strategy}__{task}"
                source = source_dir / "seed_oof_confusion_matrices.npz"
                consumed_input_identities.extend(
                    require_manifest_bound_paths(
                        classification_manifest,
                        [source],
                        identity_field="output_identities",
                        label="classification manifest",
                    )
                )
                destination = confusion_output / f"{stem}.npz"
                shutil.copy2(source, destination)
                confusion_paths.append(destination)
    outputs["confusion_matrices"] = {"npz_files": confusion_paths}

    feature_root = Path(str(classification_manifest["features"]))
    provenance_rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        feature_path = feature_file_path(feature_root, model_id)
        consumed_input_identities.extend(
            require_manifest_bound_paths(
                classification_manifest,
                [feature_path],
                identity_field="input_identities",
                label="classification manifest",
            )
        )
        validation = validate_feature_file(feature_path, require_complete=True)
        feature_provenance = validation["provenance"]
        provenance_rows.append(
            {
                "protocol_id": context.protocol["protocol_id"],
                "split_sha256": sha256_file(context.split_csv),
                "model_id": model_id,
                "model": get_model_spec(model_id).manuscript_label,
                "architecture": feature_provenance.get("architecture"),
                "checkpoint_sha256": feature_provenance.get("checkpoint_sha256"),
                "transform_profile": feature_provenance.get("transform_profile"),
                "transform_json": json.dumps(
                    feature_provenance.get("transform", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "feature_dim": validation["feature_dim"],
                "feature_h5": str(feature_path),
            }
        )
    outputs["provenance"] = {
        "csv": _atomic_csv(output / "provenance_table.csv", pd.DataFrame(provenance_rows))
    }
    manifest = {
        **plan,
        "permission": permission,
        "outputs": outputs,
        "identity_schema": "pairbst.stage_identity.v1",
        "input_stage_manifests": [
            file_identity(classification_manifest_path),
            file_identity(retrieval_manifest_path),
            file_identity(statistics_manifest_path),
        ],
        "input_identities": [*consumed_input_identities, *context_inputs],
        "output_identities": collect_output_identities(
            output,
            excluded_names=("README.md", "final_results_manifest.json"),
        ),
    }
    _atomic_json(output / "final_results_manifest.json", manifest)
    return manifest
