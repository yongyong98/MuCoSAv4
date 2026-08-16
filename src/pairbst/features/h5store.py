"""Crash-resumable canonical HDF5 storage for ROI embeddings."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SCHEMA_NAME = "pairbst_roi_features"
SCHEMA_VERSION = "1.1"
FEATURE_DATASETS = {
    "center": "features/center",
    "mean": "features/grid_mean",
    "max": "features/grid_max",
}
METADATA_FIELDS = (
    "roi_uid",
    "roi_relpath",
    "patient_uid",
    "patient_idx",
    "slide_name",
    "wsi_name",
    "roi_idx",
    "diagnosis",
    "differentiation",
    "growth_pattern",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def partial_path_for(final_path: str | Path) -> Path:
    final = Path(final_path)
    if final.suffix.lower() == ".h5":
        return final.with_name(f"{final.stem}.partial.h5")
    return final.with_name(f"{final.name}.partial.h5")


def _normalize_metadata(
    rows: Sequence[Mapping[str, Any] | Any],
) -> dict[str, list[str]]:
    normalized = {field: [] for field in METADATA_FIELDS}
    for row in rows:
        mapping = row.store_metadata() if hasattr(row, "store_metadata") else row
        absent = [field for field in METADATA_FIELDS if field not in mapping]
        if absent:
            raise ValueError(f"Feature metadata row is missing fields: {absent}")
        for field in METADATA_FIELDS:
            normalized[field].append(str(mapping[field]))
    if len(set(normalized["roi_uid"])) != len(rows):
        raise ValueError("roi_uid values must be unique")
    return normalized


class ResumableFeatureStore:
    """Write center/mean/max arrays with an atomic completion transition."""

    def __init__(
        self,
        final_path: str | Path,
        metadata_rows: Sequence[Mapping[str, Any] | Any],
        feature_dim: int,
        provenance: Mapping[str, Any],
    ) -> None:
        self.final_path = Path(final_path).expanduser().resolve()
        self.partial_path = partial_path_for(self.final_path)
        self.feature_dim = int(feature_dim)
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.metadata = _normalize_metadata(metadata_rows)
        self.num_rows = len(metadata_rows)
        if self.num_rows < 1:
            raise ValueError("At least one ROI metadata row is required")
        self.provenance = dict(provenance)
        self._file: h5py.File | None = None

    def __enter__(self) -> "ResumableFeatureStore":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("Feature store is not open")
        return self._file

    def open(self) -> None:
        if self._file is not None:
            return
        if self.final_path.exists():
            raise FileExistsError(
                f"Final feature file already exists: {self.final_path}; use validate_feature_file to inspect it"
            )
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.partial_path.exists():
            handle = h5py.File(self.partial_path, "r+")
            try:
                self._validate_open_file(handle)
            except Exception:
                handle.close()
                raise
            self._file = handle
        else:
            self._file = h5py.File(self.partial_path, "w")
            self._initialize_file()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def _initialize_file(self) -> None:
        handle = self.file
        handle.attrs.update(
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "status": "partial",
                "created_utc": utc_now(),
                "feature_dim": self.feature_dim,
                "num_rois": self.num_rows,
                "pooling_dtype": "float32",
                "grid_order": "row-major",
                "provenance_json": json.dumps(
                    self.provenance, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
        feature_group = handle.create_group("features")
        row_chunk = min(64, self.num_rows)
        for name in ("center", "grid_mean", "grid_max"):
            feature_group.create_dataset(
                name,
                shape=(self.num_rows, self.feature_dim),
                dtype=np.float32,
                chunks=(row_chunk, self.feature_dim),
                compression="lzf",
                fillvalue=np.nan,
            )
        handle.create_dataset(
            "completed",
            shape=(self.num_rows,),
            dtype=np.bool_,
            chunks=(min(1024, self.num_rows),),
            fillvalue=False,
        )
        metadata_group = handle.create_group("metadata")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for field, values in self.metadata.items():
            metadata_group.create_dataset(field, data=np.asarray(values, dtype=object), dtype=string_dtype)
        handle.flush()

    def _validate_open_file(self, handle: h5py.File) -> None:
        if handle.attrs.get("schema_name") != SCHEMA_NAME:
            raise ValueError(f"Not a {SCHEMA_NAME} file: {self.partial_path}")
        if str(handle.attrs.get("schema_version")) != SCHEMA_VERSION:
            raise ValueError("Feature schema version changed; refuse unsafe resume")
        if int(handle.attrs.get("feature_dim", -1)) != self.feature_dim:
            raise ValueError("Feature dimension changed; refuse unsafe resume")
        if int(handle.attrs.get("num_rois", -1)) != self.num_rows:
            raise ValueError("ROI count changed; refuse unsafe resume")
        if str(handle.attrs.get("status", "")) != "partial":
            raise ValueError("Partial feature path has a non-partial status")
        stored_provenance = str(handle.attrs.get("provenance_json", ""))
        expected_provenance = json.dumps(
            self.provenance, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        if stored_provenance != expected_provenance:
            raise ValueError("Extraction provenance changed; refuse unsafe resume")
        for dataset_path in FEATURE_DATASETS.values():
            dataset = handle.get(dataset_path)
            if dataset is None or dataset.shape != (self.num_rows, self.feature_dim):
                raise ValueError(f"Invalid resumable feature dataset: {dataset_path}")
            if dataset.dtype != np.dtype("float32"):
                raise ValueError(f"Invalid resumable feature dtype: {dataset_path}")
        for field, expected in self.metadata.items():
            actual = _decode_strings(handle[f"metadata/{field}"][:])
            if actual != expected:
                raise ValueError(f"ROI metadata field {field!r} changed; refuse unsafe resume")
        completed = handle["completed"]
        if completed.shape != (self.num_rows,):
            raise ValueError("Invalid completion-mask shape")

    def incomplete_indices(self) -> np.ndarray:
        return np.flatnonzero(~np.asarray(self.file["completed"][:], dtype=bool))

    def is_complete(self) -> bool:
        return bool(np.asarray(self.file["completed"][:], dtype=bool).all())

    def write_row(
        self,
        index: int,
        *,
        center: np.ndarray,
        mean: np.ndarray,
        max_: np.ndarray,
    ) -> None:
        if not 0 <= index < self.num_rows:
            raise IndexError(index)
        if bool(self.file["completed"][index]):
            return
        arrays = {
            "center": np.asarray(center, dtype=np.float32),
            "mean": np.asarray(mean, dtype=np.float32),
            "max": np.asarray(max_, dtype=np.float32),
        }
        for name, array in arrays.items():
            if array.shape != (self.feature_dim,):
                raise ValueError(
                    f"{name} feature for row {index} has shape {array.shape}; "
                    f"expected {(self.feature_dim,)}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} feature for row {index} contains NaN or Inf")
            self.file[FEATURE_DATASETS[name]][index] = array
        # Flush feature payload before marking the row complete.  This makes every
        # completed row safe to trust after an interrupted process.
        self.file.flush()
        self.file["completed"][index] = True
        self.file.flush()

    def finalize(self) -> Path:
        """Close, validate, then atomically rename ``*.partial.h5`` to ``*.h5``."""

        self.close()
        if self.final_path.exists():
            raise FileExistsError(f"Refusing to replace existing final file: {self.final_path}")
        with h5py.File(self.partial_path, "r+") as handle:
            handle.attrs["status"] = "complete"
            handle.attrs["completed_utc"] = utc_now()
            handle.flush()
        try:
            validate_feature_file(
                self.partial_path,
                expected_rows=self.num_rows,
                expected_dim=self.feature_dim,
                expected_metadata_rows=self.metadata,
                expected_provenance=self.provenance,
                require_complete=True,
            )
        except Exception:
            # Keep a failed validation recoverable as a partial file.
            with h5py.File(self.partial_path, "r+") as handle:
                handle.attrs["status"] = "partial"
                handle.flush()
            raise
        os.replace(self.partial_path, self.final_path)
        return self.final_path


def _decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def validate_feature_file(
    path: str | Path,
    *,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
    expected_metadata_rows: Sequence[Mapping[str, Any] | Any] | Mapping[str, Sequence[Any]] | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate schema, identities, provenance, values, and completion state."""

    feature_path = Path(path).expanduser().resolve()
    with h5py.File(feature_path, "r") as handle:
        if handle.attrs.get("schema_name") != SCHEMA_NAME:
            raise ValueError(f"Not a canonical PAIR-BST feature file: {feature_path}")
        if str(handle.attrs.get("schema_version", "")) != SCHEMA_VERSION:
            raise ValueError(
                f"Feature schema version is {handle.attrs.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        status = str(handle.attrs.get("status", "unknown"))
        if status not in {"partial", "complete"}:
            raise ValueError(f"Invalid feature-file status: {status!r}")
        if require_complete and status != "complete":
            raise ValueError(
                f"A completed feature file was required, but status is {status!r}"
            )
        rows = int(handle.attrs["num_rois"])
        dimension = int(handle.attrs["feature_dim"])
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows, got {rows}")
        if expected_dim is not None and dimension != expected_dim:
            raise ValueError(f"Expected dimension {expected_dim}, got {dimension}")
        for dataset_path in FEATURE_DATASETS.values():
            dataset = handle[dataset_path]
            if dataset.shape != (rows, dimension) or dataset.dtype != np.dtype("float32"):
                raise ValueError(f"Invalid dataset schema for {dataset_path}: {dataset.shape}/{dataset.dtype}")
        completed = np.asarray(handle["completed"][:], dtype=bool)
        if completed.shape != (rows,):
            raise ValueError("Invalid completion mask")
        if require_complete and not completed.all():
            raise ValueError(f"Feature file has {int((~completed).sum())} incomplete ROI rows")
        metadata: dict[str, list[str]] = {}
        for field in METADATA_FIELDS:
            dataset_path = f"metadata/{field}"
            if dataset_path not in handle:
                raise ValueError(f"Feature file is missing metadata dataset {dataset_path}")
            values = _decode_strings(handle[dataset_path][:])
            if len(values) != rows:
                raise ValueError(f"Metadata field {field!r} has {len(values)} rows; expected {rows}")
            metadata[field] = values
        if len(set(metadata["roi_uid"])) != rows:
            raise ValueError("Feature-file roi_uid metadata is not unique")
        if expected_metadata_rows is not None:
            if isinstance(expected_metadata_rows, Mapping):
                expected_metadata = {
                    field: [str(value) for value in expected_metadata_rows[field]]
                    for field in METADATA_FIELDS
                }
            else:
                expected_metadata = _normalize_metadata(expected_metadata_rows)
            for field in METADATA_FIELDS:
                if metadata[field] != expected_metadata[field]:
                    raise ValueError(
                        f"Feature-file metadata field {field!r} does not match the requested ordered ROIs"
                    )
        try:
            provenance = json.loads(str(handle.attrs.get("provenance_json", "")))
        except json.JSONDecodeError as exc:
            raise ValueError("Feature-file provenance_json is invalid") from exc
        if not isinstance(provenance, dict):
            raise ValueError("Feature-file provenance_json must contain an object")
        if expected_provenance is not None:
            observed_json = json.dumps(
                provenance, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            expected_json = json.dumps(
                dict(expected_provenance),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if observed_json != expected_json:
                raise ValueError("Feature-file provenance does not match the requested run")
        completed_indices = np.flatnonzero(completed)
        for dataset_path in FEATURE_DATASETS.values():
            for start in range(0, len(completed_indices), 128):
                batch_indices = completed_indices[start : start + 128]
                if len(batch_indices) and not np.isfinite(handle[dataset_path][batch_indices]).all():
                    raise ValueError(f"Completed rows contain NaN/Inf in {dataset_path}")
        return {
            "path": str(feature_path),
            "rows": rows,
            "feature_dim": dimension,
            "completed_rows": int(completed.sum()),
            "status": status,
            "provenance": provenance,
            "roi_uids": metadata["roi_uid"],
        }
