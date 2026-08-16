"""Small, dependency-free helpers for reproducibility hashes and JSON locks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the uppercase SHA-256 digest of *path* without loading it in memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    """Return the uppercase SHA-256 digest of a byte string."""

    return hashlib.sha256(value).hexdigest().upper()


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON deterministically for lock files and content hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    """Hash a value after canonical JSON serialization."""

    return sha256_bytes(canonical_json_dumps(value).encode("utf-8"))


def sha256_dataframe(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
    *,
    sort_by: Iterable[str] | None = None,
) -> str:
    """Hash a DataFrame using a stable UTF-8 CSV representation.

    Callers should explicitly supply ``sort_by`` when row order is not meaningful.
    Column order is preserved unless ``columns`` is provided.
    """

    selected = frame.loc[:, list(columns)].copy() if columns is not None else frame.copy()
    if sort_by is not None:
        selected = selected.sort_values(list(sort_by), kind="stable").reset_index(drop=True)
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256_bytes(payload)


def write_json_atomic(path: str | Path, value: Any, *, indent: int = 2) -> Path:
    """Atomically write a UTF-8 JSON file and return its resolved path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination.resolve()
