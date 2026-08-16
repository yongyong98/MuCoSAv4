"""Build and validate a portable manifest for the public PAIR-BST ROI files.

The public metadata reuses ``patient_idx`` within diagnoses.  The operational
patient identity is therefore the exact published diagnosis string plus
``patient_idx``.  Labels are never corrected or otherwise rewritten here.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .hashing import sha256_dataframe, sha256_file, write_json_atomic


PUBLIC_METADATA_RELATIVE_PATH = Path(
    "data/figshare/PAIR_BST_8223469_v1/metadata/Sarcoma_WSI_and_ROI_Metadata.csv"
)
PUBLIC_ROI_RELATIVE_PATH = Path("data/figshare/PAIR_BST_8223469_v1/roi_4096")
PATIENT_UID_SEPARATOR = "::patient_idx="

REQUIRED_METADATA_COLUMNS = (
    "slide_name",
    "patient_idx",
    "roi_idx",
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

MANIFEST_ID_COLUMNS = (
    "patient_uid",
    "wsi_id",
    "roi_uid",
    "roi_file",
    "roi_path",
    "roi_part",
)

_PUBLIC_ROI_NAME = re.compile(
    r"^(?P<wsi>.+)\.svs_roi_(?P<roi>[0-9]+)\.png$", re.IGNORECASE
)
_FILE_ROI_NAME = re.compile(r"^(?P<wsi>.+)_roi_(?P<roi>[0-9]+)\.png$", re.IGNORECASE)


class ManifestValidationError(ValueError):
    """Raised when metadata cannot be mapped one-to-one onto PAIR-BST ROI files."""


@dataclass(frozen=True)
class ManifestSummary:
    n_roi: int
    n_wsi: int
    n_patient: int
    n_diagnosis: int
    n_differentiation: int
    n_growth_pattern: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _canonical_index(value: Any, field: str) -> str:
    if pd.isna(value):
        raise ManifestValidationError(f"{field} contains a missing value")
    text = str(value).strip()
    if not text:
        raise ManifestValidationError(f"{field} contains an empty value")
    if re.fullmatch(r"[+-]?[0-9]+\.0+", text):
        text = text.split(".", maxsplit=1)[0]
    return text


def make_patient_uid(diagnosis: Any, patient_idx: Any) -> str:
    """Return the diagnosis-scoped operational patient identifier.

    The diagnosis is retained byte-for-byte apart from rejecting null/empty values;
    in particular, published spellings such as ``Solitaty Fibrous Tumor`` remain
    unchanged.
    """

    if pd.isna(diagnosis):
        raise ManifestValidationError("diagnosis contains a missing value")
    diagnosis_text = str(diagnosis)
    if not diagnosis_text:
        raise ManifestValidationError("diagnosis contains an empty value")
    if PATIENT_UID_SEPARATOR in diagnosis_text:
        raise ManifestValidationError(
            f"diagnosis contains reserved patient UID separator {PATIENT_UID_SEPARATOR!r}"
        )
    return f"{diagnosis_text}{PATIENT_UID_SEPARATOR}{_canonical_index(patient_idx, 'patient_idx')}"


def parse_public_roi_name(slide_name: Any, roi_idx: Any) -> tuple[str, str]:
    """Map a public metadata ``slide_name`` to ``(wsi_id, roi_file)``."""

    name = str(slide_name)
    match = _PUBLIC_ROI_NAME.fullmatch(name)
    if match is None:
        raise ManifestValidationError(
            f"unexpected public slide_name {name!r}; expected '<WSI>.svs_roi_<N>.png'"
        )
    metadata_roi = _canonical_index(roi_idx, "roi_idx")
    filename_roi = str(int(match.group("roi")))
    if metadata_roi != filename_roi:
        raise ManifestValidationError(
            f"roi_idx mismatch for {name!r}: metadata={metadata_roi!r}, filename={filename_roi!r}"
        )
    wsi_stem = match.group("wsi")
    return f"{wsi_stem}.svs", f"{wsi_stem}_roi_{filename_roi}.png"


def _scan_roi_files(roi_root: Path) -> dict[str, Path]:
    if not roi_root.is_dir():
        raise FileNotFoundError(f"ROI root does not exist: {roi_root}")
    index: dict[str, Path] = {}
    duplicates: list[str] = []
    for part in sorted(roi_root.glob("part_*")):
        if not part.is_dir():
            continue
        for path in sorted(part.glob("*.png")):
            if path.name in index:
                duplicates.append(path.name)
            else:
                index[path.name] = path
    if duplicates:
        examples = ", ".join(sorted(set(duplicates))[:5])
        raise ManifestValidationError(f"duplicate ROI filenames across parts: {examples}")
    if not index:
        raise ManifestValidationError(f"no PNG ROI files found under {roi_root}")
    return index


def _relative_or_absolute(path: Path, dataset_root: Path) -> str:
    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_manifest(
    metadata_csv: str | Path,
    roi_root: str | Path,
    *,
    verify_files: bool = True,
    strict_file_set: bool = True,
    verify_dimensions: bool = False,
    expected_roi_size: tuple[int, int] = (4096, 4096),
) -> pd.DataFrame:
    """Build a row-per-ROI manifest while preserving all public metadata columns.

    ``roi_path`` is stored relative to the dataset directory when possible, so the
    resulting CSV can move with the downloaded Figshare dataset.
    """

    metadata_path = Path(metadata_csv)
    roi_directory = Path(roi_root)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata CSV does not exist: {metadata_path}")

    frame = pd.read_csv(
        metadata_path,
        dtype={"patient_idx": "string", "roi_idx": "string"},
        keep_default_na=False,
    )
    missing_columns = sorted(set(REQUIRED_METADATA_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise ManifestValidationError(
            f"metadata is missing required columns: {', '.join(missing_columns)}"
        )
    if frame.empty:
        raise ManifestValidationError("metadata contains no ROI rows")

    roi_index = _scan_roi_files(roi_directory) if verify_files else {}
    dataset_root = metadata_path.parent.parent
    rows: list[dict[str, str]] = []
    expected_files: set[str] = set()
    for row_number, row in frame.iterrows():
        patient_idx = _canonical_index(row["patient_idx"], "patient_idx")
        roi_idx = _canonical_index(row["roi_idx"], "roi_idx")
        wsi_id, roi_file = parse_public_roi_name(row["slide_name"], roi_idx)
        patient_uid = make_patient_uid(row["diagnosis"], patient_idx)
        if verify_files:
            path = roi_index.get(roi_file)
            if path is None:
                raise ManifestValidationError(
                    f"metadata row {row_number + 2} has no matching ROI PNG: {roi_file}"
                )
            roi_path = _relative_or_absolute(path, dataset_root)
            roi_part = path.parent.name
        else:
            roi_path = ""
            roi_part = ""
        expected_files.add(roi_file)
        rows.append(
            {
                "patient_uid": patient_uid,
                "wsi_id": wsi_id,
                "roi_uid": f"{wsi_id}::roi_idx={roi_idx}",
                "roi_file": roi_file,
                "roi_path": roi_path,
                "roi_part": roi_part,
            }
        )

    if verify_files and strict_file_set:
        extras = sorted(set(roi_index).difference(expected_files))
        if extras:
            raise ManifestValidationError(
                f"ROI directory contains {len(extras)} PNG(s) absent from metadata; "
                f"examples: {', '.join(extras[:5])}"
            )

    ids = pd.DataFrame(rows, index=frame.index)
    manifest = pd.concat([frame, ids], axis=1)
    manifest["patient_idx"] = manifest["patient_idx"].map(
        lambda value: _canonical_index(value, "patient_idx")
    )
    manifest["roi_idx"] = manifest["roi_idx"].map(
        lambda value: _canonical_index(value, "roi_idx")
    )
    validate_manifest(manifest)

    if verify_dimensions:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Pillow is required for verify_dimensions=True") from error
        bad_dimensions: list[str] = []
        for roi_file in manifest["roi_file"]:
            with Image.open(roi_index[roi_file]) as image:
                if image.size != expected_roi_size:
                    bad_dimensions.append(f"{roi_file}={image.size}")
        if bad_dimensions:
            raise ManifestValidationError(
                f"{len(bad_dimensions)} ROI image(s) are not {expected_roi_size}: "
                + ", ".join(bad_dimensions[:5])
            )
    return manifest


def validate_manifest(manifest: pd.DataFrame) -> ManifestSummary:
    """Validate manifest identities and return cohort counts."""

    required = set(REQUIRED_METADATA_COLUMNS).union(MANIFEST_ID_COLUMNS)
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ManifestValidationError(f"manifest is missing columns: {', '.join(missing)}")
    if manifest.empty:
        raise ManifestValidationError("manifest contains no rows")
    null_columns = [column for column in required if manifest[column].isna().any()]
    if null_columns:
        raise ManifestValidationError(
            f"manifest contains nulls in required columns: {', '.join(sorted(null_columns))}"
        )
    for unique_column in ("roi_uid", "roi_file", "slide_name"):
        duplicated = manifest[unique_column].duplicated(keep=False)
        if duplicated.any():
            examples = manifest.loc[duplicated, unique_column].astype(str).unique()[:5]
            raise ManifestValidationError(
                f"{unique_column} is not unique; examples: {', '.join(examples)}"
            )

    expected_patient_uid = [
        make_patient_uid(diagnosis, patient_idx)
        for diagnosis, patient_idx in zip(
            manifest["diagnosis"], manifest["patient_idx"], strict=True
        )
    ]
    if expected_patient_uid != manifest["patient_uid"].astype(str).tolist():
        raise ManifestValidationError(
            "patient_uid must equal the exact diagnosis plus diagnosis-scoped patient_idx"
        )

    # Validate every row-wise identity derivation, not just column uniqueness.
    # Otherwise two rows can exchange roi_path values while retaining a
    # superficially valid set of filenames and hashes.
    identity_errors: list[str] = []
    for row_index, row in manifest.iterrows():
        roi_idx = _canonical_index(row["roi_idx"], "roi_idx")
        try:
            expected_wsi, expected_file = parse_public_roi_name(row["slide_name"], roi_idx)
        except ManifestValidationError as exc:
            identity_errors.append(f"row {row_index}: {exc}")
            continue
        expected_uid = f"{expected_wsi}::roi_idx={roi_idx}"
        raw_path = str(row["roi_path"]).replace("\\", "/")
        path_name = raw_path.rsplit("/", maxsplit=1)[-1] if raw_path else ""
        expected = {
            "wsi_id": expected_wsi,
            "roi_file": expected_file,
            "roi_uid": expected_uid,
        }
        for field, value in expected.items():
            if str(row[field]) != value:
                identity_errors.append(
                    f"row {row_index}: {field}={row[field]!r}, expected {value!r}"
                )
        if path_name != expected_file:
            identity_errors.append(
                f"row {row_index}: roi_path basename={path_name!r}, expected {expected_file!r}"
            )
        if len(identity_errors) >= 10:
            break
    if identity_errors:
        raise ManifestValidationError(
            "manifest row identities are inconsistent: " + "; ".join(identity_errors)
        )

    diagnoses_per_patient = manifest.groupby("patient_uid", sort=False)["diagnosis"].nunique()
    if (diagnoses_per_patient != 1).any():
        raise ManifestValidationError("a patient_uid maps to more than one diagnosis")

    return ManifestSummary(
        n_roi=len(manifest),
        n_wsi=int(manifest["wsi_id"].nunique()),
        n_patient=int(manifest["patient_uid"].nunique()),
        n_diagnosis=int(manifest["diagnosis"].nunique()),
        n_differentiation=int(manifest["differentiation"].nunique()),
        n_growth_pattern=int(manifest["growth_pattern"].nunique()),
    )


def _write_csv_atomic(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False, lineterminator="\n")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination.resolve()


def create_manifest_artifacts(
    metadata_csv: str | Path,
    roi_root: str | Path,
    manifest_csv: str | Path,
    lock_json: str | Path,
    **build_options: Any,
) -> dict[str, Any]:
    """Build, validate, write, and hash the public ROI manifest."""

    manifest = build_manifest(metadata_csv, roi_root, **build_options)
    summary = validate_manifest(manifest)
    output_path = _write_csv_atomic(manifest, manifest_csv)
    lock: dict[str, Any] = {
        "schema_version": 1,
        "patient_uid_definition": "exact diagnosis + '::patient_idx=' + patient_idx",
        "metadata_csv": str(Path(metadata_csv).resolve()),
        "metadata_sha256": sha256_file(metadata_csv),
        "manifest_csv": str(output_path),
        "manifest_sha256": sha256_file(output_path),
        "manifest_content_sha256": sha256_dataframe(
            manifest, sort_by=("roi_uid",)
        ),
        "summary": summary.to_dict(),
    }
    write_json_atomic(lock_json, lock)
    return lock


def resolve_manifest_roi_path(
    roi_path: str | Path, dataset_root: str | Path
) -> Path:
    """Resolve a manifest ROI path that may be dataset-relative or absolute."""

    path = Path(roi_path)
    return path if path.is_absolute() else Path(dataset_root) / path
