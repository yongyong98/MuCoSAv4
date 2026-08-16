"""Pixel-level verification gates for the public ROI benchmark inputs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .datasets import ROIRecord
from .hashing import sha256_file
from .provenance import write_json_atomic


CENTER_BOX_XYXY = (1936, 1936, 2160, 2160)


@dataclass(frozen=True)
class ImageCheckRow:
    roi_uid: str
    roi_path: str
    decode_ok: bool
    rgb_4096_ok: bool
    comparison_performed: bool
    legacy_center_found: bool
    legacy_center_pixel_exact: bool
    error: str


@dataclass(frozen=True)
class ReleaseFileCheckRow:
    filename: str
    path: str
    expected_bytes: int
    observed_bytes: int
    expected_md5: str
    observed_md5: str
    mtime_ns: int
    matches: bool
    error: str


def expected_legacy_center_name(record: ROIRecord) -> str:
    return f"SMC_{record.slide_name}"


def index_legacy_centers(root: str | Path) -> dict[str, Path]:
    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Legacy diagnosis-center root does not exist: {directory}")
    index: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.png")):
        if path.name in index:
            raise ValueError(f"Duplicate legacy center filename: {path.name}")
        index[path.name] = path
    return index


def _verify_one_record(
    record: ROIRecord,
    centers: dict[str, Path],
    *,
    comparison_performed: bool,
) -> ImageCheckRow:
    decode_ok = False
    shape_ok = False
    center_found = False
    center_equal = False
    error = ""
    try:
        with Image.open(record.roi_path) as source:
            source.load()
            decode_ok = True
            shape_ok = source.mode == "RGB" and source.size == (4096, 4096)
            if not shape_ok:
                raise ValueError(f"expected RGB 4096x4096, got {source.mode} {source.size}")
            if comparison_performed:
                expected_name = expected_legacy_center_name(record)
                center_path = centers.get(expected_name)
                center_found = center_path is not None
                if center_path is None:
                    raise FileNotFoundError(f"legacy center not found: {expected_name}")
                with Image.open(center_path) as legacy:
                    legacy.load()
                    if legacy.mode != "RGB" or legacy.size != (224, 224):
                        raise ValueError(
                            f"invalid legacy center {center_path}: {legacy.mode} {legacy.size}"
                        )
                    public_center = source.crop(CENTER_BOX_XYXY)
                    try:
                        center_equal = bool(
                            np.array_equal(
                                np.asarray(public_center, dtype=np.uint8),
                                np.asarray(legacy, dtype=np.uint8),
                            )
                        )
                    finally:
                        public_center.close()
                    if not center_equal:
                        raise ValueError("public and legacy center pixels differ")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return ImageCheckRow(
        roi_uid=record.roi_uid,
        roi_path=str(record.roi_path),
        decode_ok=decode_ok,
        rgb_4096_ok=shape_ok,
        comparison_performed=comparison_performed,
        legacy_center_found=center_found,
        legacy_center_pixel_exact=center_equal,
        error=error,
    )


def verify_images_and_centers(
    records: Sequence[ROIRecord],
    *,
    legacy_center_root: str | Path | None = None,
    output_directory: str | Path | None = None,
    workers: int = 1,
    progress: Callable[[int, int, ROIRecord], None] | None = None,
) -> dict[str, object]:
    """Fully decode each ROI and perform an exact center-pixel comparison.

    This function performs no model inference. It is the required image and
    center-compatibility gate before any cached center feature is reused.
    """

    if not records:
        raise ValueError("No ROI records supplied")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    comparison_performed = legacy_center_root is not None
    centers = index_legacy_centers(legacy_center_root) if comparison_performed else {}
    rows: list[ImageCheckRow] = []
    checker = partial(
        _verify_one_record,
        centers=centers,
        comparison_performed=comparison_performed,
    )
    if workers == 1:
        iterator = map(checker, records)
        for index, (record, row) in enumerate(zip(records, iterator, strict=True), start=1):
            rows.append(row)
            if progress is not None:
                progress(index, len(records), record)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pairbst-image-qa") as pool:
            iterator = pool.map(checker, records)
            for index, (record, row) in enumerate(zip(records, iterator, strict=True), start=1):
                rows.append(row)
                if progress is not None:
                    progress(index, len(records), record)

    frame = pd.DataFrame([asdict(row) for row in rows])
    failures = frame[frame["error"] != ""]
    if comparison_performed:
        center_equivalence_status = "PASS" if failures.empty else "FAIL"
    else:
        center_equivalence_status = "NOT_PERFORMED"
    result: dict[str, object] = {
        "status": "PASS" if failures.empty else "FAIL",
        "comparison_performed": comparison_performed,
        "center_equivalence_status": center_equivalence_status,
        "roi_checked": int(len(frame)),
        "fully_decoded": int(frame["decode_ok"].sum()),
        "rgb_4096": int(frame["rgb_4096_ok"].sum()),
        "legacy_centers_found": int(frame["legacy_center_found"].sum()),
        "legacy_centers_pixel_exact": int(frame["legacy_center_pixel_exact"].sum()),
        "failure_count": int(len(failures)),
        "failure_examples": failures[["roi_uid", "error"]].head(20).to_dict(orient="records"),
    }
    if output_directory is not None:
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output / "image_center_checks.csv", index=False, lineterminator="\n")
        write_json_atomic(result, output / "image_center_summary.json")
    return result


def _md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _verify_release_file(
    item: tuple[str, Path, int, str],
) -> ReleaseFileCheckRow:
    filename, path, expected_bytes, expected_md5 = item
    observed_bytes = -1
    observed_md5 = ""
    mtime_ns = -1
    error = ""
    try:
        stat = path.stat()
        observed_bytes = stat.st_size
        mtime_ns = stat.st_mtime_ns
        observed_md5 = _md5_file(path)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    matches = (
        not error
        and observed_bytes == expected_bytes
        and observed_md5 == expected_md5.lower()
    )
    if not matches and not error:
        error = (
            f"identity mismatch: bytes {observed_bytes} != {expected_bytes} or "
            f"md5 {observed_md5} != {expected_md5.lower()}"
        )
    return ReleaseFileCheckRow(
        filename=filename,
        path=str(path),
        expected_bytes=expected_bytes,
        observed_bytes=observed_bytes,
        expected_md5=expected_md5.lower(),
        observed_md5=observed_md5,
        mtime_ns=mtime_ns,
        matches=matches,
        error=error,
    )


def verify_release_files_against_manifest(
    records: Sequence[ROIRecord],
    *,
    metadata_csv: str | Path,
    required_manifest: str | Path,
    output_directory: str | Path | None = None,
    workers: int = 4,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Hash every released PNG and metadata CSV against Figshare MD5/size."""

    if not records:
        raise ValueError("No ROI records supplied")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    metadata_path = Path(metadata_csv).expanduser().resolve()
    manifest_path = Path(required_manifest).expanduser().resolve()
    released = pd.read_csv(
        manifest_path,
        sep="\t",
        header=None,
        names=("article_id", "file_id", "filename", "bytes", "md5", "url"),
        dtype={"filename": "string", "md5": "string"},
    )
    if released["filename"].duplicated().any():
        raise ValueError("Figshare required manifest contains duplicate filenames")
    local_paths: dict[str, Path] = {record.roi_path.name: record.roi_path for record in records}
    if len(local_paths) != len(records):
        raise ValueError("Local ROI basenames are not unique")
    local_paths[metadata_path.name] = metadata_path
    expected_names = set(released["filename"].astype(str))
    if set(local_paths) != expected_names:
        raise ValueError(
            "Local/released filename sets differ; "
            f"missing={sorted(expected_names - set(local_paths))[:5]}, "
            f"extra={sorted(set(local_paths) - expected_names)[:5]}"
        )
    items = [
        (
            str(row.filename),
            local_paths[str(row.filename)],
            int(row.bytes),
            str(row.md5),
        )
        for row in released.itertuples(index=False)
    ]
    rows: list[ReleaseFileCheckRow] = []
    if workers == 1:
        iterator = map(_verify_release_file, items)
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            if progress is not None:
                progress(index, len(items), row.filename)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pairbst-md5") as pool:
            for index, row in enumerate(pool.map(_verify_release_file, items), start=1):
                rows.append(row)
                if progress is not None:
                    progress(index, len(items), row.filename)
    frame = pd.DataFrame([asdict(row) for row in rows])
    failures = frame[~frame["matches"]]
    result: dict[str, Any] = {
        "status": "PASS" if failures.empty else "FAIL",
        "algorithm": "MD5_as_published_by_Figshare_plus_exact_byte_size",
        "required_manifest_path": str(manifest_path),
        "required_manifest_sha256": sha256_file(manifest_path),
        "file_count": len(frame),
        "matched_files": int(frame["matches"].sum()),
        "failure_count": len(failures),
        "failure_examples": failures[["filename", "error"]].head(20).to_dict("records"),
    }
    if output_directory is not None:
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output / "release_file_integrity.csv", index=False, lineterminator="\n")
        write_json_atomic(result, output / "release_file_integrity_summary.json")
    return result
