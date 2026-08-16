"""Lazy model construction, checkpoint verification, and input transforms."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .registry import ModelSpec, TransformSpec, get_model_spec, get_transform_spec


@dataclass(frozen=True)
class CheckpointVerification:
    path: Path
    exists: bool
    size_bytes: int | None
    expected_size_bytes: int | None
    size_matches: bool | None
    sha256: str | None
    expected_sha256: str | None
    hash_matches: bool | None

    @property
    def ok(self) -> bool:
        return self.exists and self.size_matches is not False and self.hash_matches is not False


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_checkpoint(
    model: str | ModelSpec,
    checkpoint_path: str | Path,
    *,
    calculate_hash: bool = True,
) -> CheckpointVerification:
    """Verify local checkpoint identity without loading or downloading it."""

    spec = get_model_spec(model) if isinstance(model, str) else model
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        return CheckpointVerification(
            path=path,
            exists=False,
            size_bytes=None,
            expected_size_bytes=spec.checkpoint.size_bytes,
            size_matches=None,
            sha256=None,
            expected_sha256=spec.checkpoint.sha256,
            hash_matches=None,
        )
    size = path.stat().st_size
    size_match = None if spec.checkpoint.size_bytes is None else size == spec.checkpoint.size_bytes
    digest = _sha256(path) if calculate_hash else None
    match = None
    if digest is not None and spec.checkpoint.sha256 is not None:
        match = digest == spec.checkpoint.sha256.upper()
    return CheckpointVerification(
        path=path,
        exists=True,
        size_bytes=size,
        expected_size_bytes=spec.checkpoint.size_bytes,
        size_matches=size_match,
        sha256=digest,
        expected_sha256=spec.checkpoint.sha256,
        hash_matches=match,
    )


def build_transform(model: str | ModelSpec, profile: str = "official_model_specific") -> Any:
    """Build a deterministic PIL-to-tensor transform for one frozen encoder."""

    transform_spec = get_transform_spec(model, profile)
    return _build_transform_from_spec(transform_spec)


def _build_transform_from_spec(spec: TransformSpec) -> Any:
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except ImportError as exc:  # pragma: no cover - depends on target environment
        raise RuntimeError("torchvision is required to construct model transforms") from exc

    interpolation = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }[spec.interpolation]
    operations: list[Any] = [
        transforms.Resize(spec.resize, interpolation=interpolation, antialias=spec.antialias)
    ]
    if spec.crop is not None:
        operations.append(transforms.CenterCrop(spec.crop))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=spec.mean, std=spec.std),
        ]
    )
    return transforms.Compose(operations)


def _torch_load_state_dict(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - target environment dependency
            raise RuntimeError("safetensors is required to load this checkpoint") from exc
        loaded = load_file(str(path), device="cpu")
        if not isinstance(loaded, dict):  # pragma: no cover - library contract
            raise TypeError(f"Checkpoint {path} did not contain a state dictionary")
        return loaded

    import torch

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Checkpoint {path} did not contain a state dictionary")
    for key in ("state_dict", "model"):
        candidate = loaded.get(key)
        if isinstance(candidate, dict):
            loaded = candidate
            break
    if loaded and all(str(key).startswith("module.") for key in loaded):
        loaded = {str(key)[7:]: value for key, value in loaded.items()}
    return loaded


def _import_module_from_file(path: Path) -> ModuleType:
    module_spec = importlib.util.spec_from_file_location("pairbst_external_retccl_resnet", path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Unable to import RetCCL architecture from {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def pool_virchow2_tokens(tokens: Any) -> Any:
    """Create Paige's 2,560-D tile embedding from Virchow2 token output."""

    import torch

    if getattr(tokens, "ndim", None) != 3 or tuple(tokens.shape[1:]) != (261, 1280):
        shape = tuple(tokens.shape) if getattr(tokens, "shape", None) is not None else None
        raise ValueError(
            "Virchow2 must return [batch, 261, 1280] tokens; "
            f"got {shape}"
        )
    class_token = tokens[:, 0]
    patch_tokens = tokens[:, 5:]  # tokens 1..4 are register tokens
    return torch.cat((class_token, patch_tokens.mean(dim=1)), dim=-1)


def build_model(
    model: str | ModelSpec,
    checkpoint_path: str | Path,
    *,
    device: str | Any = "cpu",
    verify_hash: bool = True,
    retccl_source_path: str | Path | None = None,
) -> Any:
    """Construct one frozen encoder strictly from a local checkpoint.

    No builder uses ``pretrained=True`` and no network access is attempted.
    ``verify_hash=False`` is intended only when the caller has verified the same
    path immediately beforehand and retained that digest in provenance.
    """

    spec = get_model_spec(model) if isinstance(model, str) else model
    verification = verify_checkpoint(spec, checkpoint_path, calculate_hash=verify_hash)
    if not verification.exists:
        raise FileNotFoundError(
            f"Checkpoint for {spec.name} is missing: {verification.path}. "
            "Acquire it under its upstream terms; the benchmark never downloads weights."
        )
    if verification.size_matches is False:
        raise ValueError(
            f"Checkpoint size mismatch for {spec.name}: got {verification.size_bytes}, "
            f"expected {verification.expected_size_bytes}"
        )
    if verification.hash_matches is False:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {spec.name}: got {verification.sha256}, "
            f"expected {verification.expected_sha256}"
        )

    if spec.upstream_config_sha256 is not None:
        config_path = verification.path.parent / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Pinned upstream config for {spec.name} is missing: {config_path}"
            )
        observed_config_sha256 = _sha256(config_path)
        if observed_config_sha256 != spec.upstream_config_sha256:
            raise ValueError(
                f"Upstream config SHA-256 mismatch for {spec.name}: "
                f"got {observed_config_sha256}, expected {spec.upstream_config_sha256}"
            )

    import torch
    from torch import nn

    state_dict = _torch_load_state_dict(verification.path)
    if spec.name == "resnet50_v2":
        from torchvision.models import resnet50

        encoder = resnet50(weights=None)
        encoder.load_state_dict(state_dict, strict=True)
        encoder.fc = nn.Identity()
    elif spec.name == "swin_t":
        from torchvision.models import swin_t

        encoder = swin_t(weights=None)
        encoder.load_state_dict(state_dict, strict=True)
        encoder.head = nn.Identity()
    elif spec.name == "retccl":
        if retccl_source_path is None:
            raise RuntimeError(
                "RetCCL requires an explicit hash-locked retccl_source_path; "
                "ambient Python imports are forbidden."
            )
        source_path = Path(retccl_source_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"RetCCL architecture source is missing: {source_path}")
        observed_source_sha256 = _sha256(source_path)
        expected_source_sha256 = spec.architecture_source_sha256
        if expected_source_sha256 is None or observed_source_sha256 != expected_source_sha256:
            raise ValueError(
                "RetCCL architecture source SHA-256 mismatch: "
                f"got {observed_source_sha256}, expected {expected_source_sha256}"
            )
        retccl_resnet = _import_module_from_file(source_path)
        encoder = retccl_resnet.resnet50(
            num_classes=128, mlp=False, two_branch=False, normlinear=True
        )
        encoder.fc = nn.Identity()
        encoder.load_state_dict(state_dict, strict=True)
    elif spec.name == "uni":
        import timm

        encoder = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
            pretrained=False,
        )
        encoder.load_state_dict(state_dict, strict=True)
    elif spec.name == "uni2_h":
        import timm

        encoder = timm.create_model(
            "vit_giant_patch14_224",
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
            pretrained=False,
        )
        encoder.load_state_dict(state_dict, strict=True)
    elif spec.name == "prov_gigapath":
        import timm

        # The official GigaPath tile encoder is a DINOv2 ViT-giant whose
        # released timm configuration overrides the nominal patch14 backbone
        # to a 16-pixel patch.  The slide encoder is intentionally not used:
        # PAIR-BST compares one representation per 224px image patch.
        encoder = timm.create_model(
            "vit_giant_patch14_dinov2",
            img_size=224,
            in_chans=3,
            patch_size=16,
            embed_dim=1536,
            depth=40,
            num_heads=24,
            init_values=1e-5,
            mlp_ratio=5.33334,
            num_classes=0,
            global_pool="token",
            pretrained=False,
        )
        encoder.load_state_dict(state_dict, strict=True)
    elif spec.name == "virchow2":
        import timm

        backbone = timm.create_model(
            "vit_huge_patch14_224",
            img_size=224,
            init_values=1e-5,
            num_classes=0,
            reg_tokens=4,
            mlp_ratio=5.3375,
            global_pool="",
            dynamic_img_size=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            pretrained=False,
        )
        backbone.load_state_dict(state_dict, strict=True)

        class Virchow2TileEmbedding(nn.Module):
            """Apply Paige's official CLS + mean-patch representation rule."""

            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, inputs: Any) -> Any:
                return pool_virchow2_tokens(self.model(inputs))

        encoder = Virchow2TileEmbedding(backbone)
    else:  # pragma: no cover - protected by the registry
        raise AssertionError(f"No adapter implemented for {spec.name}")

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return encoder.to(device)


def forward_embeddings(model: Any, batch: Any) -> Any:
    """Normalize common model return types to a two-dimensional embedding tensor."""

    output = model(batch)
    if isinstance(output, (tuple, list)):
        output = output[0]
    elif isinstance(output, dict):
        for key in ("features", "embeddings", "x", "last_hidden_state"):
            if key in output:
                output = output[key]
                break
        else:
            raise TypeError(f"Unsupported model output mapping keys: {tuple(output)}")
    if getattr(output, "ndim", None) is None:
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    if output.ndim > 2:
        raise ValueError(
            "Encoder returned a token/spatial tensor without an explicit, "
            f"model-specific pooling rule: {tuple(output.shape)}"
        )
    if output.ndim != 2:
        raise ValueError(f"Expected [batch, embedding] output; got shape {tuple(output.shape)}")
    return output
