"""Read-only adapter for preserved recovery-package feature HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import h5py
import numpy as np

from ..datasets import ROIRecord
from ..hashing import sha256_file


BANNED_RECOVERY_DIRECTORIES = frozenset(
    {"center_repooling_not_genuine_grid", "alternate_center_features"}
)
SUPPORTED_STRATEGIES = ("center", "mean", "max")
SUPPORTED_USES = ("inspection", "legacy_sensitivity", "primary_after_anchor")


@dataclass(frozen=True)
class RecoverySourceContract:
    """External immutable identity required for every recovered HDF5 source."""

    model_name: str
    strategy: str
    feature_dim: int
    source_sha256: str
    checkpoint_sha256: str | None
    transform_profile: str
    allowed_use: str = "inspection"

    def validate(self) -> None:
        if self.strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported recovery strategy in contract: {self.strategy!r}")
        if self.feature_dim < 1:
            raise ValueError("Recovery feature_dim must be positive")
        digest = self.source_sha256.upper()
        if len(digest) != 64 or any(character not in "0123456789ABCDEF" for character in digest):
            raise ValueError("Recovery source_sha256 must be a 64-character hexadecimal digest")
        if self.allowed_use not in SUPPORTED_USES:
            raise ValueError(f"Unsupported recovered-feature use: {self.allowed_use!r}")
        if not self.model_name or not self.transform_profile:
            raise ValueError("Recovery model_name and transform_profile are required")
        if self.allowed_use == "primary_after_anchor":
            checkpoint_digest = (self.checkpoint_sha256 or "").upper()
            if len(checkpoint_digest) != 64 or any(
                character not in "0123456789ABCDEF" for character in checkpoint_digest
            ):
                raise ValueError(
                    "Primary recovered-feature reuse requires a frozen checkpoint SHA-256"
                )
            if self.transform_profile != "official_model_specific":
                raise ValueError(
                    "Primary recovered-feature reuse requires the official_model_specific transform"
                )


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _canonical_index(value: str) -> str:
    return value[:-2] if value.endswith(".0") and value[:-2].isdigit() else value


def _record_key(record: ROIRecord) -> tuple[str, str, str, str]:
    return (record.diagnosis, record.patient_idx, _canonical_wsi(record.wsi_name), record.roi_idx)


def _canonical_wsi(value: str) -> str:
    name = Path(value).name
    if name.startswith("SMC_"):
        name = name[4:]
    marker = ".svs_roi_"
    location = name.lower().find(marker)
    if location >= 0:
        name = f"{name[:location]}.svs"
    return name


def _assert_safe_recovery_path(path: Path) -> None:
    path_parts = {part.casefold() for part in path.resolve().parts}
    blocked = path_parts.intersection(BANNED_RECOVERY_DIRECTORIES)
    if blocked:
        raise ValueError(
            f"Refusing invalid recovery feature source under {sorted(blocked)[0]!r}: {path}"
        )


class RecoveryFeatureAdapter:
    """Expose preserved arrays in canonical metadata order without copying them.

    ``center`` accepts the validated ``center_features`` HDF5 layout. ``mean``
    and ``max`` accept only files that identify their matching pooling method;
    pseudo-grid and alternate-center directories are rejected unconditionally.
    """

    def __init__(
        self,
        records: Sequence[ROIRecord],
        source_paths: Mapping[str, str | Path],
        contracts: Mapping[str, RecoverySourceContract],
        *,
        intended_use: str = "inspection",
        anchor_verified: bool = False,
    ) -> None:
        if not records:
            raise ValueError("At least one canonical ROI record is required")
        unknown = set(source_paths).difference(SUPPORTED_STRATEGIES)
        if unknown:
            raise ValueError(f"Unsupported recovery strategies: {sorted(unknown)}")
        if set(contracts) != set(source_paths):
            raise ValueError("Every recovery source must have exactly one immutable contract")
        if intended_use not in SUPPORTED_USES:
            raise ValueError(f"Unsupported intended recovery use: {intended_use!r}")
        if intended_use == "primary_after_anchor" and not anchor_verified:
            raise ValueError("Primary recovered-feature reuse requires a verified anchor pilot")
        self.contracts = dict(contracts)
        for strategy, contract in self.contracts.items():
            contract.validate()
            if contract.strategy != strategy:
                raise ValueError(
                    f"Recovery contract strategy {contract.strategy!r} does not match key {strategy!r}"
                )
            allowed_rank = SUPPORTED_USES.index(contract.allowed_use)
            intended_rank = SUPPORTED_USES.index(intended_use)
            if intended_rank > allowed_rank:
                raise ValueError(
                    f"Recovery source {strategy!r} permits {contract.allowed_use!r}, "
                    f"not requested use {intended_use!r}"
                )
        self.records = records
        self.source_paths = {
            strategy: Path(path).expanduser().resolve() for strategy, path in source_paths.items()
        }
        for path in self.source_paths.values():
            _assert_safe_recovery_path(path)
        self._files: dict[str, h5py.File] = {}
        self._orders: dict[str, np.ndarray] = {}

    def __enter__(self) -> "RecoveryFeatureAdapter":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def open(self) -> None:
        if self._files:
            return
        try:
            for strategy, path in self.source_paths.items():
                handle = h5py.File(path, "r")
                try:
                    self._validate_source(strategy, path, handle)
                    self._files[strategy] = handle
                    self._orders[strategy] = self._join_order(handle)
                except Exception:
                    if strategy not in self._files:
                        handle.close()
                    raise
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._orders.clear()

    def _validate_source(self, strategy: str, path: Path, handle: h5py.File) -> None:
        contract = self.contracts[strategy]
        observed_sha256 = sha256_file(path)
        if observed_sha256 != contract.source_sha256.upper():
            raise ValueError(
                f"Recovery source SHA-256 mismatch for {path}: {observed_sha256} "
                f"!= {contract.source_sha256.upper()}"
            )
        required = {"features", "diagnosis", "patient_idx", "slide_name", "roi_idx"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise ValueError(f"Recovery feature file {path} is missing datasets: {missing}")
        features = handle["features"]
        if features.ndim != 2 or features.dtype != np.dtype("float32"):
            raise ValueError(f"Expected float32 [ROI, feature] dataset in {path}")
        if int(features.shape[1]) != contract.feature_dim:
            raise ValueError(
                f"Recovery feature dimension {features.shape[1]} != contract {contract.feature_dim}"
            )
        if strategy in {"mean", "max"}:
            raw_pooling = handle.attrs.get("pooling_method", "")
            pooling_method = (
                raw_pooling.decode("utf-8") if isinstance(raw_pooling, bytes) else str(raw_pooling)
            ).lower()
            if pooling_method != strategy:
                raise ValueError(
                    f"Grid source {path} pooling_method={pooling_method!r}; expected {strategy!r}"
                )
        elif str(handle.attrs.get("pooling_method", "")).lower() in {"mean", "max"}:
            raise ValueError(f"Center source unexpectedly declares grid pooling: {path}")

    def _join_order(self, handle: h5py.File) -> np.ndarray:
        fields = {
            name: _decode(handle[name][:])
            for name in ("diagnosis", "patient_idx", "slide_name", "roi_idx")
        }
        row_count = handle["features"].shape[0]
        if any(len(values) != row_count for values in fields.values()):
            raise ValueError("Recovery metadata lengths do not match feature rows")
        source_index: dict[tuple[str, str, str, str], int] = {}
        for index in range(row_count):
            key = (
                fields["diagnosis"][index],
                _canonical_index(fields["patient_idx"][index]),
                _canonical_wsi(fields["slide_name"][index]),
                _canonical_index(fields["roi_idx"][index]),
            )
            if key in source_index:
                raise ValueError(f"Duplicate recovery ROI identity: {key}")
            source_index[key] = index
        canonical_keys = [_record_key(record) for record in self.records]
        missing = [key for key in canonical_keys if key not in source_index]
        if missing:
            raise ValueError(f"Recovery features are missing {len(missing)} canonical ROIs; first={missing[0]}")
        if len(source_index) != len(canonical_keys):
            raise ValueError(
                f"Recovery source has {len(source_index)} rows but canonical manifest has "
                f"{len(canonical_keys)}"
            )
        return np.asarray([source_index[key] for key in canonical_keys], dtype=np.int64)

    def feature_dim(self, strategy: str) -> int:
        return int(self._files[strategy]["features"].shape[1])

    def read(self, strategy: str, canonical_indices: Sequence[int] | None = None) -> np.ndarray:
        """Read selected rows as float32 in canonical manifest order."""

        if strategy not in self._files:
            raise KeyError(f"Strategy {strategy!r} was not opened")
        if canonical_indices is None:
            canonical = np.arange(len(self.records), dtype=np.int64)
        else:
            canonical = np.asarray(canonical_indices, dtype=np.int64)
        if canonical.ndim != 1:
            raise ValueError("canonical_indices must be one-dimensional")
        if len(canonical) == 0:
            return np.empty((0, self.feature_dim(strategy)), dtype=np.float32)
        if canonical.min() < 0 or canonical.max() >= len(self.records):
            raise IndexError("canonical feature index is out of range")
        source_rows = self._orders[strategy][canonical]
        # h5py requires strictly increasing fancy indices. Unique rows cover
        # arbitrary order and repeated caller indices, then inverse restores it.
        unique_rows, inverse = np.unique(source_rows, return_inverse=True)
        unique_features = np.asarray(
            self._files[strategy]["features"][unique_rows], dtype=np.float32
        )
        return unique_features[inverse]

    def iter_batches(
        self, strategy: str, batch_size: int = 256
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        for start in range(0, len(self.records), batch_size):
            indices = np.arange(start, min(start + batch_size, len(self.records)))
            yield indices, self.read(strategy, indices)
