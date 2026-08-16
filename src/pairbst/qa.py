"""Leakage and class-coverage quality assurance for frozen split manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .provenance import write_json_atomic


DEFAULT_TASKS = ("diagnosis", "differentiation", "growth_pattern")
PAIRBST_EXPECTED_FOLDS = (0, 1, 2)
PAIRBST_EXPECTED_FOLD_COUNTS: Mapping[int, Mapping[str, int]] = {
    0: {"patients": 90, "wsi": 162, "roi": 771},
    1: {"patients": 89, "wsi": 157, "roi": 757},
    2: {"patients": 89, "wsi": 151, "roi": 724},
}


def _coverage_table(
    frame: pd.DataFrame,
    task: str,
    *,
    unit: str,
    fold_column: str,
    fold_values: Iterable[int],
) -> pd.DataFrame:
    if unit == "patient":
        source = frame[["patient_uid", task, fold_column]].drop_duplicates()
        values = pd.crosstab(source[task], source[fold_column])
    elif unit == "roi":
        values = pd.crosstab(frame[task], frame[fold_column])
    else:
        raise ValueError(f"unsupported coverage unit: {unit}")
    values = values.reindex(columns=list(fold_values), fill_value=0)
    values.columns = [f"fold_{value}" for value in values.columns]
    values.insert(0, "class", values.index.astype(str))
    values.insert(0, "task", task)
    return values.reset_index(drop=True)


def audit_cv_split(
    frame: pd.DataFrame,
    *,
    tasks: Iterable[str] = DEFAULT_TASKS,
    fold_column: str = "fold",
    expected_folds: Iterable[int] = PAIRBST_EXPECTED_FOLDS,
    expected_fold_counts: Mapping[int, Mapping[str, int]] | None = PAIRBST_EXPECTED_FOLD_COUNTS,
) -> dict[str, Any]:
    tasks = tuple(tasks)
    expected_fold_ids = tuple(int(value) for value in expected_folds)
    if len(expected_fold_ids) != len(set(expected_fold_ids)):
        raise ValueError("expected_folds contains duplicates")
    if not expected_fold_ids:
        raise ValueError("expected_folds must not be empty")
    required = {"patient_uid", "wsi_id", "roi_uid", fold_column, *tasks}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"split manifest is missing columns: {missing}")
    if frame["roi_uid"].duplicated().any():
        raise ValueError("roi_uid is not unique")

    numeric_folds = pd.to_numeric(frame[fold_column], errors="coerce")
    if numeric_folds.isna().any() or not (numeric_folds == numeric_folds.astype(int)).all():
        raise ValueError(f"{fold_column} must contain integer fold IDs")
    frame = frame.copy()
    frame[fold_column] = numeric_folds.astype(int)

    patient_membership = frame.groupby("patient_uid")[fold_column].nunique()
    wsi_membership = frame.groupby("wsi_id")[fold_column].nunique()
    folds = sorted(frame[fold_column].unique().tolist())
    fold_ids_match = folds == sorted(expected_fold_ids)
    patient_coverage = pd.concat(
        [
            _coverage_table(
                frame,
                task,
                unit="patient",
                fold_column=fold_column,
                fold_values=expected_fold_ids,
            )
            for task in tasks
        ],
        ignore_index=True,
    )
    roi_coverage = pd.concat(
        [
            _coverage_table(
                frame,
                task,
                unit="roi",
                fold_column=fold_column,
                fold_values=expected_fold_ids,
            )
            for task in tasks
        ],
        ignore_index=True,
    )
    coverage_columns = [f"fold_{value}" for value in expected_fold_ids]
    missing_classes = patient_coverage.loc[
        (patient_coverage[coverage_columns] == 0).any(axis=1), ["task", "class"]
    ]

    fold_rows: list[dict[str, int]] = []
    for fold in folds:
        selected = frame[frame[fold_column] == fold]
        fold_rows.append(
            {
                "fold": int(fold),
                "patients": int(selected["patient_uid"].nunique()),
                "wsi": int(selected["wsi_id"].nunique()),
                "roi": int(len(selected)),
            }
        )
    observed_count_map = {int(row["fold"]): row for row in fold_rows}
    fold_counts_match = True
    count_mismatches: list[dict[str, Any]] = []
    if expected_fold_counts is not None:
        for fold in expected_fold_ids:
            expected = expected_fold_counts.get(fold)
            observed = observed_count_map.get(fold)
            for unit in ("patients", "wsi", "roi"):
                expected_value = None if expected is None else int(expected.get(unit, -1))
                observed_value = None if observed is None else int(observed[unit])
                if expected_value is None or expected_value < 0 or observed_value != expected_value:
                    fold_counts_match = False
                    count_mismatches.append(
                        {
                            "fold": int(fold),
                            "unit": unit,
                            "observed": observed_value,
                            "expected": expected_value,
                        }
                    )
        if set(int(value) for value in expected_fold_counts) != set(expected_fold_ids):
            fold_counts_match = False
    roi_duplicates = int(frame["roi_uid"].duplicated().sum())
    semantic_pass = (
        fold_ids_match
        and fold_counts_match
        and patient_membership.max() == 1
        and wsi_membership.max() == 1
        and roi_duplicates == 0
        and missing_classes.empty
    )
    result: dict[str, Any] = {
        "status": "PASS" if semantic_pass else "FAIL",
        "folds": fold_rows,
        "expected_fold_ids": list(expected_fold_ids),
        "observed_fold_ids": folds,
        "fold_ids_match": fold_ids_match,
        "fold_counts_match": fold_counts_match,
        "fold_count_mismatches": count_mismatches,
        "unique_patients": int(frame["patient_uid"].nunique()),
        "unique_wsi": int(frame["wsi_id"].nunique()),
        "unique_roi": int(frame["roi_uid"].nunique()),
        "patients_in_multiple_folds": int((patient_membership > 1).sum()),
        "wsi_in_multiple_folds": int((wsi_membership > 1).sum()),
        "roi_duplicates": roi_duplicates,
        "missing_class_fold_pairs": missing_classes.to_dict(orient="records"),
    }
    result["_patient_coverage"] = patient_coverage
    result["_roi_coverage"] = roi_coverage
    return result


def write_cv_split_qa(
    split_csv: str | Path,
    output_directory: str | Path,
    *,
    tasks: Iterable[str] = DEFAULT_TASKS,
) -> dict[str, Any]:
    frame = pd.read_csv(split_csv)
    result = audit_cv_split(frame, tasks=tasks)
    patient_coverage = result.pop("_patient_coverage")
    roi_coverage = result.pop("_roi_coverage")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    patient_coverage.to_csv(output / "class_coverage_patients.csv", index=False, lineterminator="\n")
    roi_coverage.to_csv(output / "class_coverage_rois.csv", index=False, lineterminator="\n")
    write_json_atomic(result, output / "split_qa.json")
    return result
