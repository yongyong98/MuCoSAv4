"""Metadata-to-image binding for the public PAIR-BST 4096-pixel ROI archive."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image

from .manifest import make_patient_uid, parse_public_roi_name


REQUIRED_METADATA_COLUMNS = (
    "slide_name",
    "patient_idx",
    "roi_idx",
    "diagnosis",
    "differentiation",
    "growth_pattern",
)


def _clean_index(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


@dataclass(frozen=True)
class ROIRecord:
    """One public 4096 x 4096 ROI and its three benchmark labels."""

    roi_path: Path
    relative_path: str
    slide_name: str
    wsi_name: str
    patient_idx: str
    roi_idx: str
    diagnosis: str
    differentiation: str
    growth_pattern: str
    extra: Mapping[str, str] = field(default_factory=dict, compare=False, repr=False)

    @property
    def patient_uid(self) -> str:
        # patient_idx is diagnosis-scoped in the released metadata.
        return make_patient_uid(self.diagnosis, self.patient_idx)

    @property
    def roi_uid(self) -> str:
        return f"{self.wsi_name}::roi_idx={self.roi_idx}"

    def store_metadata(self) -> dict[str, str]:
        return {
            "roi_uid": self.roi_uid,
            "roi_relpath": self.relative_path,
            "patient_uid": self.patient_uid,
            "patient_idx": self.patient_idx,
            "slide_name": self.slide_name,
            "wsi_name": self.wsi_name,
            "roi_idx": self.roi_idx,
            "diagnosis": self.diagnosis,
            "differentiation": self.differentiation,
            "growth_pattern": self.growth_pattern,
        }


def discover_roi_files(roi_root: str | Path) -> dict[str, Path]:
    """Index released PNGs by basename and reject ambiguous duplicates."""

    root = Path(roi_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ROI root does not exist: {root}")
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        if path.name in index:
            raise ValueError(f"Duplicate ROI filename {path.name!r}: {index[path.name]} and {path}")
        index[path.name] = path.resolve()
    if not index:
        raise FileNotFoundError(f"No PNG ROI files found under {root}")
    return index


def load_roi_records(
    metadata_csv: str | Path,
    roi_root: str | Path,
    *,
    strict: bool = True,
) -> list[ROIRecord]:
    """Bind metadata rows to ROI PNGs while preserving the published row order."""

    metadata_path = Path(metadata_csv).expanduser().resolve()
    root = Path(roi_root).expanduser().resolve()
    file_index = discover_roi_files(root)
    records: list[ROIRecord] = []
    missing: list[str] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        absent_columns = [name for name in REQUIRED_METADATA_COLUMNS if name not in (reader.fieldnames or ())]
        if absent_columns:
            raise ValueError(f"Metadata is missing required columns: {absent_columns}")
        for row_number, raw_row in enumerate(reader, start=2):
            row = {str(key): "" if value is None else str(value) for key, value in raw_row.items()}
            public_slide_name = Path(row["slide_name"]).name
            wsi_name, roi_filename = parse_public_roi_name(public_slide_name, row["roi_idx"])
            roi_path = file_index.get(roi_filename)
            if roi_path is None:
                missing.append(roi_filename)
                if strict:
                    continue
                else:
                    continue
            records.append(
                ROIRecord(
                    roi_path=roi_path,
                    relative_path=roi_path.relative_to(root).as_posix(),
                    slide_name=public_slide_name,
                    wsi_name=wsi_name,
                    patient_idx=_clean_index(row["patient_idx"]),
                    roi_idx=_clean_index(row["roi_idx"]),
                    diagnosis=row["diagnosis"],
                    differentiation=row["differentiation"],
                    growth_pattern=row["growth_pattern"],
                    extra={**row, "metadata_row": str(row_number)},
                )
            )
    if strict and missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} metadata ROI files were not found; first: {preview}")
    roi_uids = [record.roi_uid for record in records]
    if len(set(roi_uids)) != len(roi_uids):
        raise ValueError("Metadata produced duplicate ROI identities")
    return records


def open_roi_rgb(record: ROIRecord) -> Image.Image:
    """Decode one ROI once and detach it from the underlying file handle."""

    with Image.open(record.roi_path) as source:
        source.load()
        return source.convert("RGB")


def records_fingerprint(records: Iterable[ROIRecord]) -> str:
    """Hash ordered metadata/path identities (not the large PNG byte streams)."""

    digest = hashlib.sha256()
    for record in records:
        payload = {
            **record.store_metadata(),
            "file_size": record.roi_path.stat().st_size,
        }
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
