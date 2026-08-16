"""Patient-grouped split creation and historical split recovery for PAIR-BST."""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .hashing import sha256_dataframe, sha256_file, write_json_atomic
from .manifest import (
    ManifestValidationError,
    _canonical_index,
    _write_csv_atomic,
    make_patient_uid,
    validate_manifest,
)


DEFAULT_N_SPLITS = 3
DEFAULT_SPLIT_SEED = 20260814
HISTORICAL_PATIENT_COUNTS = {"train": 133, "val": 27, "test": 108}
HISTORICAL_ROI_COUNTS = {"train": 1161, "val": 199, "test": 892}
HISTORICAL_WSI_COUNTS = {"train": 243, "val": 41, "test": 186}
LEGACY_SPLIT_ORDER = ("train", "val", "test")


class SplitValidationError(ValueError):
    """Raised when a split violates patient grouping or class coverage."""


@dataclass(frozen=True)
class FoldSummary:
    fold: int
    n_patient: int
    n_wsi: int
    n_roi: int
    n_diagnosis: int
    n_differentiation: int
    n_growth_pattern: int


def _patient_table(manifest: pd.DataFrame) -> pd.DataFrame:
    validate_manifest(manifest)
    diagnosis_counts = manifest.groupby("patient_uid", sort=False)["diagnosis"].nunique()
    if (diagnosis_counts != 1).any():
        raise SplitValidationError("each patient_uid must have exactly one diagnosis")
    return (
        manifest.groupby("patient_uid", sort=False)
        .agg(
            diagnosis=("diagnosis", "first"),
            n_roi=("roi_uid", "size"),
            n_wsi=("wsi_id", "nunique"),
        )
        .reset_index()
        .sort_values("patient_uid", kind="stable")
        .reset_index(drop=True)
    )


def _safe_feature_name(prefix: str, value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_")
    return f"{prefix}__{text}"


def _balance_features(
    manifest: pd.DataFrame, patients: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Create patient-level feature totals used only to improve fold balance."""

    patient_index = pd.Index(patients["patient_uid"], name="patient_uid")
    pieces: list[pd.DataFrame] = []
    weights: list[float] = []
    names: list[str] = []

    basics = patients.set_index("patient_uid")[["n_roi", "n_wsi"]].astype(float)
    basics.columns = ["total_roi", "total_wsi"]
    pieces.append(basics)
    names.extend(basics.columns)
    weights.extend((1.0, 0.5))

    # ROI incidence by diagnosis balances tissue volume while the initial
    # stratification separately guarantees patient-level diagnosis coverage.
    diagnosis_roi = pd.crosstab(manifest["patient_uid"], manifest["diagnosis"]).reindex(
        patient_index, fill_value=0
    )
    diagnosis_roi.columns = [
        _safe_feature_name("diagnosis_roi", value) for value in diagnosis_roi.columns
    ]
    pieces.append(diagnosis_roi.astype(float))
    names.extend(diagnosis_roi.columns)
    weights.extend([1.5] * diagnosis_roi.shape[1])

    for column, prefix in (
        ("differentiation", "differentiation_roi"),
        ("growth_pattern", "growth_pattern_roi"),
    ):
        counts = pd.crosstab(manifest["patient_uid"], manifest[column]).reindex(
            patient_index, fill_value=0
        )
        counts.columns = [_safe_feature_name(prefix, value) for value in counts.columns]
        pieces.append(counts.astype(float))
        names.extend(counts.columns)
        weights.extend([1.0] * counts.shape[1])

        presence = (counts > 0).astype(float)
        presence.columns = [name.replace("_roi__", "_patient__") for name in counts.columns]
        pieces.append(presence)
        names.extend(presence.columns)
        weights.extend([1.0] * presence.shape[1])

    matrix = pd.concat(pieces, axis=1).reindex(patient_index).to_numpy(dtype=np.float64)
    return matrix, np.asarray(weights, dtype=np.float64), tuple(names)


def _objective_rows(
    totals: np.ndarray,
    targets: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    folds: Iterable[int],
) -> float:
    selected = np.asarray(list(folds), dtype=np.int64)
    deviation = (totals[selected] - targets[selected]) / scales
    return float(np.sum(np.square(deviation) * weights[None, :]))


def _optimize_same_diagnosis_swaps(
    patient_table: pd.DataFrame,
    assignments: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    *,
    n_splits: int,
    max_passes: int,
) -> np.ndarray:
    """Improve ROI/task balance without changing diagnosis strata or fold sizes."""

    if max_passes <= 0:
        return assignments.copy()
    optimized = assignments.copy()
    totals = np.vstack([features[optimized == fold].sum(axis=0) for fold in range(n_splits)])
    # Feature targets account for the one-patient fold-size difference (90/89/89)
    # rather than forcing exactly one third of every ROI-level feature.
    fold_sizes = np.bincount(optimized, minlength=n_splits).astype(np.float64)
    targets = fold_sizes[:, None] / fold_sizes.sum() * features.sum(axis=0)[None, :]
    scales = np.maximum(features.sum(axis=0) / n_splits, 1.0)
    diagnoses = patient_table["diagnosis"].astype(str).to_numpy()

    diagnosis_groups = [
        np.flatnonzero(diagnoses == diagnosis)
        for diagnosis in sorted(pd.unique(diagnoses).tolist())
    ]
    tolerance = 1e-12
    for _ in range(max_passes):
        changed = False
        for group in diagnosis_groups:
            for left, right in itertools.combinations(group.tolist(), 2):
                left_fold = int(optimized[left])
                right_fold = int(optimized[right])
                if left_fold == right_fold:
                    continue
                old_cost = _objective_rows(
                    totals, targets, scales, weights, (left_fold, right_fold)
                )
                proposed_left = totals[left_fold] - features[left] + features[right]
                proposed_right = totals[right_fold] - features[right] + features[left]
                old_left = totals[left_fold].copy()
                old_right = totals[right_fold].copy()
                totals[left_fold] = proposed_left
                totals[right_fold] = proposed_right
                new_cost = _objective_rows(
                    totals, targets, scales, weights, (left_fold, right_fold)
                )
                if new_cost < old_cost - tolerance:
                    optimized[left], optimized[right] = right_fold, left_fold
                    changed = True
                else:
                    totals[left_fold] = old_left
                    totals[right_fold] = old_right
        if not changed:
            break
    return optimized


def make_patient_folds(
    manifest: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SPLIT_SEED,
    optimize_balance: bool = True,
    max_balance_passes: int = 30,
) -> pd.DataFrame:
    """Create deterministic patient-grouped folds stratified by diagnosis.

    The initial split uses diagnosis-stratified K-fold assignment.  Optional
    deterministic same-diagnosis swaps then improve ROI volume, WSI volume,
    differentiation, and growth-pattern balance without changing patient counts or
    diagnosis coverage.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    patients = _patient_table(manifest)
    diagnosis_support = patients.groupby("diagnosis", sort=False)["patient_uid"].size()
    insufficient = diagnosis_support[diagnosis_support < n_splits]
    if not insufficient.empty:
        details = ", ".join(f"{name}={count}" for name, count in insufficient.items())
        raise SplitValidationError(
            f"every diagnosis needs at least {n_splits} patients; insufficient: {details}"
        )
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as error:
        raise RuntimeError("scikit-learn is required to construct CV folds") from error

    assignments = np.full(len(patients), -1, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    labels = patients["diagnosis"].astype(str).to_numpy()
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros(len(patients)), labels)):
        assignments[held_out] = fold
    if (assignments < 0).any():
        raise RuntimeError("internal error: at least one patient was not assigned a fold")

    if optimize_balance:
        features, weights, _ = _balance_features(manifest, patients)
        assignments = _optimize_same_diagnosis_swaps(
            patients,
            assignments,
            features,
            weights,
            n_splits=n_splits,
            max_passes=max_balance_passes,
        )

    result = patients.copy()
    result.insert(2, "fold", assignments.astype(int))
    result = result[["patient_uid", "diagnosis", "fold", "n_roi", "n_wsi"]]
    validate_patient_folds(result, n_splits=n_splits)
    return result.sort_values("patient_uid", kind="stable").reset_index(drop=True)


def validate_patient_folds(
    patient_folds: pd.DataFrame, *, n_splits: int = DEFAULT_N_SPLITS
) -> None:
    """Validate a row-per-patient fold table."""

    required = {"patient_uid", "diagnosis", "fold"}
    missing = sorted(required.difference(patient_folds.columns))
    if missing:
        raise SplitValidationError(
            f"patient fold table is missing columns: {', '.join(missing)}"
        )
    if patient_folds.empty:
        raise SplitValidationError("patient fold table is empty")
    if patient_folds["patient_uid"].duplicated().any():
        raise SplitValidationError("patient_uid appears more than once in patient fold table")
    try:
        observed_folds = set(patient_folds["fold"].astype(int).unique().tolist())
    except (TypeError, ValueError) as error:
        raise SplitValidationError("fold values must be integers") from error
    expected_folds = set(range(n_splits))
    if observed_folds != expected_folds:
        raise SplitValidationError(
            f"expected folds {sorted(expected_folds)}, observed {sorted(observed_folds)}"
        )
    coverage = pd.crosstab(patient_folds["diagnosis"], patient_folds["fold"])
    coverage = coverage.reindex(columns=range(n_splits), fill_value=0)
    absent = coverage.index[(coverage == 0).any(axis=1)].tolist()
    if absent:
        raise SplitValidationError(
            "diagnoses absent from at least one fold: " + ", ".join(map(str, absent))
        )


def apply_patient_folds(
    manifest: pd.DataFrame, patient_folds: pd.DataFrame
) -> pd.DataFrame:
    """Attach held-out fold indices to every ROI without changing row order."""

    validate_manifest(manifest)
    n_splits = int(patient_folds["fold"].nunique())
    validate_patient_folds(patient_folds, n_splits=n_splits)
    mapped = manifest.merge(
        patient_folds[["patient_uid", "fold"]],
        on="patient_uid",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if mapped["fold"].isna().any():
        missing = mapped.loc[mapped["fold"].isna(), "patient_uid"].unique()[:5]
        raise SplitValidationError(
            "patients missing from fold table: " + ", ".join(map(str, missing))
        )
    mapped["fold"] = mapped["fold"].astype(int)
    if set(patient_folds["patient_uid"]) != set(manifest["patient_uid"]):
        extras = sorted(set(patient_folds["patient_uid"]).difference(manifest["patient_uid"]))
        raise SplitValidationError(
            f"patient fold table contains {len(extras)} patient(s) absent from manifest"
        )
    return mapped


def summarize_cv_folds(
    split_manifest: pd.DataFrame, *, n_splits: int | None = None
) -> list[FoldSummary]:
    """Validate and summarize a row-per-ROI cross-validation manifest."""

    if "fold" not in split_manifest.columns:
        raise SplitValidationError("split manifest has no fold column")
    patient_fold_counts = split_manifest.groupby("patient_uid", sort=False)["fold"].nunique()
    if (patient_fold_counts != 1).any():
        raise SplitValidationError("at least one patient_uid appears in multiple folds")
    observed = sorted(split_manifest["fold"].astype(int).unique().tolist())
    resolved_n_splits = n_splits if n_splits is not None else len(observed)
    patient_table = split_manifest[
        ["patient_uid", "diagnosis", "fold"]
    ].drop_duplicates("patient_uid")
    validate_patient_folds(patient_table, n_splits=resolved_n_splits)

    summaries: list[FoldSummary] = []
    for fold in range(resolved_n_splits):
        selected = split_manifest[split_manifest["fold"].astype(int) == fold]
        summaries.append(
            FoldSummary(
                fold=fold,
                n_patient=int(selected["patient_uid"].nunique()),
                n_wsi=int(selected["wsi_id"].nunique()),
                n_roi=len(selected),
                n_diagnosis=int(selected["diagnosis"].nunique()),
                n_differentiation=int(selected["differentiation"].nunique()),
                n_growth_pattern=int(selected["growth_pattern"].nunique()),
            )
        )
    return summaries


def cv_train_test_indices(
    split_manifest: pd.DataFrame, held_out_fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer row indices for two-fold train / one-fold held-out evaluation."""

    if "fold" not in split_manifest.columns:
        raise SplitValidationError("split manifest has no fold column")
    folds = split_manifest["fold"].to_numpy(dtype=np.int64)
    if held_out_fold not in set(folds.tolist()):
        raise SplitValidationError(f"held_out_fold {held_out_fold} is not present")
    test = np.flatnonzero(folds == held_out_fold)
    train = np.flatnonzero(folds != held_out_fold)
    return train, test


def _normalize_wsi_id(value: Any) -> str:
    name = Path(str(value)).name
    name = re.sub(r"_roi_[0-9]+\.png$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.svs$", "", name, flags=re.IGNORECASE)
    if name.startswith("SMC_"):
        name = name[4:]
    return name


def _legacy_key_frame(frame: pd.DataFrame, *, public_manifest: bool) -> pd.DataFrame:
    required = {"diagnosis", "patient_idx", "roi_idx"}
    slide_column = "wsi_id" if public_manifest else "slide_name"
    required.add(slide_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SplitValidationError(f"split input is missing columns: {', '.join(missing)}")
    keyed = pd.DataFrame(index=frame.index)
    keyed["diagnosis_key"] = frame["diagnosis"].astype(str)
    keyed["patient_idx_key"] = frame["patient_idx"].map(
        lambda value: _canonical_index(value, "patient_idx")
    )
    keyed["roi_idx_key"] = frame["roi_idx"].map(
        lambda value: _canonical_index(value, "roi_idx")
    )
    keyed["wsi_key"] = frame[slide_column].map(_normalize_wsi_id)
    return keyed


def import_legacy_split(
    manifest: pd.DataFrame,
    legacy_csv: str | Path,
    *,
    verify_historical_counts: bool = True,
    verify_shared_metadata: bool = True,
) -> pd.DataFrame:
    """Map the recovered legacy train/val/test split onto the public manifest.

    The recovered file uses ``SMC_<WSI>.svs`` whereas the Figshare metadata stores
    ``<WSI>.svs_roi_<N>.png``.  The four-part key diagnosis, diagnosis-scoped
    patient index, WSI, and ROI index makes the mapping one-to-one.
    """

    validate_manifest(manifest)
    legacy_path = Path(legacy_csv)
    if not legacy_path.is_file():
        raise FileNotFoundError(f"legacy split CSV does not exist: {legacy_path}")
    legacy = pd.read_csv(
        legacy_path,
        dtype={"patient_idx": "string", "roi_idx": "string"},
        keep_default_na=False,
    )
    if "split" not in legacy.columns:
        raise SplitValidationError("legacy CSV has no split column")
    public_keys = _legacy_key_frame(manifest, public_manifest=True)
    legacy_keys = _legacy_key_frame(legacy, public_manifest=False)
    key_columns = ["diagnosis_key", "patient_idx_key", "roi_idx_key", "wsi_key"]
    if public_keys.duplicated(key_columns).any():
        raise SplitValidationError("public manifest has duplicate legacy join keys")
    if legacy_keys.duplicated(key_columns).any():
        raise SplitValidationError("legacy split has duplicate join keys")

    public_side = public_keys.copy()
    public_side["__public_row"] = np.arange(len(manifest), dtype=np.int64)
    legacy_side = legacy_keys.copy()
    legacy_side["legacy_split"] = legacy["split"].astype(str).str.strip().str.lower()
    legacy_side["__legacy_row"] = np.arange(len(legacy), dtype=np.int64)
    joined = public_side.merge(
        legacy_side,
        on=key_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched_public = int((joined["_merge"] == "left_only").sum())
    unmatched_legacy = int((joined["_merge"] == "right_only").sum())
    if unmatched_public or unmatched_legacy:
        raise SplitValidationError(
            "legacy/public ROI mapping is not one-to-one: "
            f"public_without_legacy={unmatched_public}, legacy_without_public={unmatched_legacy}"
        )
    joined = joined.sort_values("__public_row", kind="stable")
    legacy_order = joined["__legacy_row"].astype(int).to_numpy()
    splits = joined["legacy_split"].to_numpy()
    allowed = set(LEGACY_SPLIT_ORDER)
    unexpected = sorted(set(splits.tolist()).difference(allowed))
    if unexpected:
        raise SplitValidationError(f"unexpected legacy split labels: {unexpected}")

    if verify_shared_metadata:
        comparable = [
            column
            for column in (
                "diagnosis_raw",
                "diagnosis",
                "differentiation",
                "growth_pattern",
                "top_left_x",
                "top_left_y",
                "top_right_x",
                "top_right_y",
                "bottom_left_x",
                "bottom_left_y",
                "bottom_right_x",
                "bottom_right_y",
            )
            if column in manifest.columns and column in legacy.columns
        ]
        for column in comparable:
            public_values = manifest[column].astype(str).to_numpy()
            legacy_values = legacy.iloc[legacy_order][column].astype(str).to_numpy()
            mismatch = public_values != legacy_values
            if mismatch.any():
                raise SplitValidationError(
                    f"legacy/public metadata mismatch in {column}: {int(mismatch.sum())} row(s)"
                )

    mapped = manifest.copy()
    mapped["legacy_split"] = splits
    summarize_legacy_split(mapped, verify_historical_counts=verify_historical_counts)
    return mapped


def summarize_legacy_split(
    split_manifest: pd.DataFrame, *, verify_historical_counts: bool = True
) -> dict[str, Any]:
    """Validate patient isolation and summarize the recovered ~50/10/40 split."""

    if "legacy_split" not in split_manifest.columns:
        raise SplitValidationError("legacy split manifest has no legacy_split column")
    labels = split_manifest["legacy_split"].astype(str).str.lower()
    unexpected = sorted(set(labels).difference(LEGACY_SPLIT_ORDER))
    if unexpected:
        raise SplitValidationError(f"unexpected legacy split labels: {unexpected}")
    patient_membership = (
        split_manifest.assign(legacy_split=labels)
        .groupby("patient_uid", sort=False)["legacy_split"]
        .nunique()
    )
    if (patient_membership != 1).any():
        raise SplitValidationError("at least one operational patient appears in multiple legacy splits")
    counts: dict[str, dict[str, int]] = {}
    for split in LEGACY_SPLIT_ORDER:
        selected = split_manifest[labels == split]
        counts[split] = {
            "n_patient": int(selected["patient_uid"].nunique()),
            "n_wsi": int(selected["wsi_id"].nunique()),
            "n_roi": len(selected),
        }
    if verify_historical_counts:
        observed_patient = {name: values["n_patient"] for name, values in counts.items()}
        observed_wsi = {name: values["n_wsi"] for name, values in counts.items()}
        observed_roi = {name: values["n_roi"] for name, values in counts.items()}
        if observed_patient != HISTORICAL_PATIENT_COUNTS:
            raise SplitValidationError(
                f"historical patient counts changed: {observed_patient}"
            )
        if observed_wsi != HISTORICAL_WSI_COUNTS:
            raise SplitValidationError(f"historical WSI counts changed: {observed_wsi}")
        if observed_roi != HISTORICAL_ROI_COUNTS:
            raise SplitValidationError(f"historical ROI counts changed: {observed_roi}")
    total_patients = sum(values["n_patient"] for values in counts.values())
    return {
        "protocol": "legacy_50_10_40",
        "counts": counts,
        "patient_percent": {
            name: 100.0 * values["n_patient"] / total_patients
            for name, values in counts.items()
        },
    }


def write_cv_split_artifacts(
    manifest: pd.DataFrame,
    patient_folds: pd.DataFrame,
    *,
    roi_split_csv: str | Path,
    patient_folds_csv: str | Path,
    lock_json: str | Path,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Write immutable-looking CV split inputs plus a machine-verifiable lock."""

    split_manifest = apply_patient_folds(manifest, patient_folds)
    summaries = summarize_cv_folds(split_manifest)
    roi_output = _write_csv_atomic(split_manifest, roi_split_csv)
    patient_output = _write_csv_atomic(patient_folds, patient_folds_csv)
    lock = {
        "schema_version": 1,
        "protocol": "patient_grouped_diagnosis_stratified_3fold",
        "seed": int(seed),
        "patient_uid_definition": "exact diagnosis + '::patient_idx=' + patient_idx",
        "fold_semantics": "fold is held out once; the other two folds are training data",
        "roi_split_csv": str(roi_output),
        "roi_split_sha256": sha256_file(roi_output),
        "roi_split_content_sha256": sha256_dataframe(
            split_manifest, sort_by=("roi_uid",)
        ),
        "patient_folds_csv": str(patient_output),
        "patient_folds_sha256": sha256_file(patient_output),
        "patient_folds_content_sha256": sha256_dataframe(
            patient_folds, sort_by=("patient_uid",)
        ),
        "folds": [asdict(summary) for summary in summaries],
    }
    write_json_atomic(lock_json, lock)
    return lock


def write_legacy_split_artifacts(
    manifest: pd.DataFrame,
    legacy_csv: str | Path,
    *,
    output_csv: str | Path,
    lock_json: str | Path,
) -> dict[str, Any]:
    """Import, validate, write, and hash the recovered historical split."""

    mapped = import_legacy_split(manifest, legacy_csv, verify_historical_counts=True)
    summary = summarize_legacy_split(mapped, verify_historical_counts=True)
    destination = _write_csv_atomic(mapped, output_csv)
    lock = {
        "schema_version": 1,
        "protocol": "legacy_50_10_40",
        "source_csv": str(Path(legacy_csv).resolve()),
        "source_sha256": sha256_file(legacy_csv),
        "output_csv": str(destination),
        "output_sha256": sha256_file(destination),
        "output_content_sha256": sha256_dataframe(mapped, sort_by=("roi_uid",)),
        "summary": summary,
    }
    write_json_atomic(lock_json, lock)
    return lock
