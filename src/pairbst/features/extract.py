"""Streaming, one-model-at-a-time feature extraction from public 4096 ROIs."""

from __future__ import annotations

import gc
import importlib.metadata
import os
import platform
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..datasets import ROIRecord, open_roi_rgb, records_fingerprint
from ..geometry import GRID_PATCH_COUNT, batched, iter_center_and_grid
from ..hashing import sha256_file
from ..models.adapters import (
    build_model,
    build_transform,
    forward_embeddings,
    verify_checkpoint,
)
from ..models.registry import ModelSpec, get_model_spec, get_transform_spec
from .h5store import ResumableFeatureStore, validate_feature_file
from .pooling import OnlineMeanMax
from ..provenance import configure_deterministic_cuda_environment


@dataclass(frozen=True)
class ExtractionConfig:
    batch_size: int = 64
    device: str = "cuda"
    transform_profile: str = "official_model_specific"
    autocast_dtype: str | None = None
    deterministic_algorithms: bool = True

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.autocast_dtype not in (None, "float16", "bfloat16"):
            raise ValueError("autocast_dtype must be null, 'float16', or 'bfloat16'")


@dataclass(frozen=True)
class ModelExtractionRun:
    model_name: str
    checkpoint_path: Path
    output_path: Path
    retccl_source_path: Path | None = None


def _autocast_context(torch: Any, device: Any, dtype_name: str | None) -> Any:
    if dtype_name is None:
        return nullcontext()
    if device.type == "cpu" and dtype_name == "float16":
        raise ValueError("CPU float16 autocast is unsupported; use bfloat16 or disable autocast")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    return torch.autocast(device_type=device.type, dtype=dtype)


def extract_roi_features(
    image: Any,
    model: Any,
    transform: Callable[[Any], Any],
    *,
    device: str | Any,
    feature_dim: int,
    batch_size: int,
    autocast_dtype: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract center, grid-mean and grid-max vectors from one decoded ROI."""

    import torch

    device_object = torch.device(device)
    accumulator = OnlineMeanMax(feature_dim)
    center_feature: np.ndarray | None = None
    for sampled_batch in batched(iter_center_and_grid(image), batch_size):
        tensors = []
        for sampled in sampled_batch:
            tensors.append(transform(sampled.image))
            sampled.image.close()
        tensor_batch = torch.stack(tensors, dim=0)
        if device_object.type == "cuda":
            tensor_batch = tensor_batch.pin_memory()
        tensor_batch = tensor_batch.to(device_object, non_blocking=device_object.type == "cuda")
        with torch.inference_mode(), _autocast_context(torch, device_object, autocast_dtype):
            embedded = forward_embeddings(model, tensor_batch)
        if embedded.shape != (len(sampled_batch), feature_dim):
            raise ValueError(
                f"Model returned {tuple(embedded.shape)}; expected "
                f"{(len(sampled_batch), feature_dim)}"
            )
        # Pooling is always explicitly float32, independent of encoder autocast.
        features = embedded.detach().to(device="cpu", dtype=torch.float32).numpy()
        for sampled, feature in zip(sampled_batch, features, strict=True):
            if sampled.sampling == "center":
                if center_feature is not None:
                    raise RuntimeError("Center patch was emitted more than once")
                center_feature = np.asarray(feature, dtype=np.float32).copy()
            else:
                accumulator.update(feature)
        del tensor_batch, embedded, features
    if center_feature is None:
        raise RuntimeError("Center patch was not emitted")
    mean_feature, max_feature = accumulator.finalize(expected_count=GRID_PATCH_COUNT)
    return center_feature, mean_feature, max_feature


def _configure_torch(config: ExtractionConfig) -> Any:
    config.validate()
    if config.deterministic_algorithms:
        configure_deterministic_cuda_environment()
    import torch

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
    return device


def _provenance(
    spec: ModelSpec,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    records: Sequence[ROIRecord],
    dataset_manifest_sha256: str,
    config: ExtractionConfig,
    device: Any,
    architecture_source_path: Path | None,
    architecture_source_sha256: str | None,
) -> dict[str, Any]:
    import torch

    transform_spec = get_transform_spec(spec, config.transform_profile)
    software: dict[str, str] = {}
    for name in ("torch", "torchvision", "timm", "numpy", "pillow", "h5py"):
        try:
            software[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            software[name] = "not-installed"
    hardware: dict[str, Any] = {
        "requested_device": config.device,
        "resolved_device": str(device),
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "gpu_capability": None,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        hardware["gpu_name"] = torch.cuda.get_device_name(index)
        hardware["gpu_capability"] = list(torch.cuda.get_device_capability(index))
    provenance = {
        "model_name": spec.name,
        "manuscript_label": spec.manuscript_label,
        "architecture": spec.architecture,
        "embedding_dim": spec.embedding_dim,
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_source": spec.checkpoint.source,
        "architecture_source_filename": (
            architecture_source_path.name if architecture_source_path is not None else None
        ),
        "architecture_source_sha256": architecture_source_sha256,
        "transform_profile": config.transform_profile,
        "transform": transform_spec.as_dict(),
        "sampling": {
            "roi_size": 4096,
            "center_box_xyxy": [1936, 1936, 2160, 2160],
            "grid": "16x16 non-overlapping 256px row-major",
            "patch_png_materialization": False,
        },
        "pooling_dtype": "float32",
        "encoder_autocast_dtype": config.autocast_dtype,
        "extraction_batch_size": config.batch_size,
        "deterministic_algorithms": config.deterministic_algorithms,
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if config.deterministic_algorithms
            else None
        ),
        "tf32_allowed": False if config.deterministic_algorithms else None,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "ordered_records_fingerprint": records_fingerprint(records),
        "num_rois": len(records),
        "software": software,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": hardware,
    }
    # Preserve byte-for-byte provenance compatibility for the original five
    # encoders while binding the two newly added gated models to their pinned
    # upstream repository configuration.
    if spec.upstream_revision is not None:
        provenance["upstream_revision"] = spec.upstream_revision
    if spec.upstream_config_sha256 is not None:
        provenance["upstream_config_sha256"] = spec.upstream_config_sha256
    return provenance


def extract_model_features(
    records: Sequence[ROIRecord],
    run: ModelExtractionRun,
    *,
    dataset_manifest_sha256: str,
    config: ExtractionConfig = ExtractionConfig(),
    progress: Callable[[int, int, ROIRecord], None] | None = None,
) -> Path:
    """Run one encoder and resume safely at ROI boundaries.

    This is the execution boundary used by the CLI. Merely importing this
    module cannot load a model, touch a checkpoint, or start an experiment.
    """

    if not records:
        raise ValueError("No ROI records supplied")
    spec = get_model_spec(run.model_name)
    output_path = Path(run.output_path).expanduser().resolve()
    checkpoint_path = Path(run.checkpoint_path).expanduser().resolve()
    architecture_source_path: Path | None = None
    architecture_source_sha256: str | None = None
    if spec.name == "retccl":
        if run.retccl_source_path is None:
            raise ValueError("RetCCL requires an explicit hash-locked architecture source path")
        architecture_source_path = Path(run.retccl_source_path).expanduser().resolve()
        if not architecture_source_path.is_file():
            raise FileNotFoundError(architecture_source_path)
        architecture_source_sha256 = sha256_file(architecture_source_path)
        if architecture_source_sha256 != spec.architecture_source_sha256:
            raise ValueError(
                "RetCCL architecture source SHA-256 mismatch: "
                f"{architecture_source_sha256} != {spec.architecture_source_sha256}"
            )
    device = _configure_torch(config)
    if spec.checkpoint.sha256 is None:
        raise ValueError(f"Model {spec.name} has no frozen checkpoint SHA-256")
    requested_provenance = _provenance(
        spec,
        checkpoint_path,
        spec.checkpoint.sha256,
        records,
        dataset_manifest_sha256,
        config,
        device,
        architecture_source_path,
        architecture_source_sha256,
    )
    if output_path.exists():
        validate_feature_file(
            output_path,
            expected_rows=len(records),
            expected_dim=spec.embedding_dim,
            expected_metadata_rows=records,
            expected_provenance=requested_provenance,
            require_complete=True,
        )
        return output_path

    checkpoint = verify_checkpoint(spec, checkpoint_path, calculate_hash=True)
    if not checkpoint.exists:
        raise FileNotFoundError(checkpoint.path)
    if checkpoint.size_matches is False:
        raise ValueError(
            f"Checkpoint size mismatch for {spec.name}: {checkpoint.size_bytes} "
            f"!= {checkpoint.expected_size_bytes}"
        )
    if checkpoint.hash_matches is False:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {spec.name}: {checkpoint.sha256} "
            f"!= {checkpoint.expected_sha256}"
        )
    assert checkpoint.sha256 is not None
    transform = build_transform(spec, config.transform_profile)
    model = build_model(
        spec,
        checkpoint.path,
        device=device,
        # The path was fully hashed immediately above; avoid hashing multi-GB
        # UNI checkpoints a second time before constructing the architecture.
        verify_hash=False,
        retccl_source_path=architecture_source_path,
    )
    if checkpoint.sha256 != requested_provenance["checkpoint_sha256"]:
        raise RuntimeError("Verified checkpoint identity changed during extraction setup")
    store = ResumableFeatureStore(
        output_path, records, spec.embedding_dim, requested_provenance
    )
    with store:
        pending = store.incomplete_indices().tolist()
        for completed_count, index in enumerate(pending, start=1):
            record = records[index]
            image = open_roi_rgb(record)
            try:
                center, mean, max_feature = extract_roi_features(
                    image,
                    model,
                    transform,
                    device=device,
                    feature_dim=spec.embedding_dim,
                    batch_size=config.batch_size,
                    autocast_dtype=config.autocast_dtype,
                )
            except Exception as exc:
                raise RuntimeError(f"Feature extraction failed for ROI {record.roi_uid}") from exc
            finally:
                image.close()
            store.write_row(index, center=center, mean=mean, max_=max_feature)
            if progress is not None:
                progress(completed_count, len(pending), record)
        complete = store.is_complete()
    if not complete:
        raise RuntimeError("Extraction ended with incomplete ROI rows")
    return store.finalize()


def extract_models_sequentially(
    records: Sequence[ROIRecord],
    runs: Iterable[ModelExtractionRun],
    *,
    dataset_manifest_sha256: str,
    config: ExtractionConfig = ExtractionConfig(),
    progress: Callable[[str, int, int, ROIRecord], None] | None = None,
) -> list[Path]:
    """Process models serially so only one encoder occupies accelerator memory."""

    outputs: list[Path] = []
    for run in runs:
        callback = None
        if progress is not None:
            callback = lambda done, total, record, name=run.model_name: progress(
                name, done, total, record
            )
        outputs.append(
            extract_model_features(
                records,
                run,
                dataset_manifest_sha256=dataset_manifest_sha256,
                config=config,
                progress=callback,
            )
        )
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
    return outputs
