"""Declarative registry for the encoders in the manuscript benchmark.

The registry deliberately contains no eager PyTorch imports.  Preflight and
reporting commands can therefore inspect model provenance on machines that do
not have the GPU environment installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransformSpec:
    """A deterministic inference transform and its provenance."""

    resize: int | tuple[int, int]
    crop: int | None
    interpolation: Literal["bilinear", "bicubic"]
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    antialias: bool = True
    source: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "resize": self.resize,
            "crop": self.crop,
            "interpolation": self.interpolation,
            "mean": self.mean,
            "std": self.std,
            "antialias": self.antialias,
            "source": self.source,
        }


@dataclass(frozen=True)
class CheckpointSpec:
    """Expected checkpoint identity without redistributing the binary."""

    filename: str
    sha256: str | None
    size_bytes: int | None
    source: str
    license_note: str


@dataclass(frozen=True)
class ModelSpec:
    """Frozen encoder definition."""

    name: str
    manuscript_label: str
    architecture: str
    embedding_dim: int
    checkpoint: CheckpointSpec
    official_transform: TransformSpec
    architecture_source_sha256: str | None = None
    upstream_revision: str | None = None
    upstream_config_sha256: str | None = None
    aliases: tuple[str, ...] = ()


_LEGACY_COMMON = TransformSpec(
    resize=(224, 224),
    crop=None,
    interpolation="bilinear",
    source="PAIR-BST legacy common square Resize((224, 224)) sensitivity analysis",
)


MODEL_REGISTRY: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "resnet50_v2": ModelSpec(
            name="resnet50_v2",
            manuscript_label="ResNet",
            architecture="torchvision.models.resnet50",
            embedding_dim=2048,
            checkpoint=CheckpointSpec(
                filename="resnet50-11ad3fa6.pth",
                sha256="11AD3FA62CA79E40ADDFD354A8EC4B7C75143B3038B8D2A807FBC68DEAB379CA",
                size_bytes=102_540_417,
                source="torchvision ResNet50_Weights.IMAGENET1K_V2",
                license_note="Torchvision model weights; preserve upstream attribution.",
            ),
            official_transform=TransformSpec(
                resize=232,
                crop=224,
                interpolation="bilinear",
                source="torchvision ResNet50_Weights.IMAGENET1K_V2.transforms",
            ),
            aliases=("resnet", "resnet50", "resnet-50"),
        ),
        "swin_t": ModelSpec(
            name="swin_t",
            manuscript_label="Swin-T",
            architecture="torchvision.models.swin_t",
            embedding_dim=768,
            checkpoint=CheckpointSpec(
                filename="swin_t-704ceda3.pth",
                sha256="704CEDA373461B0A224FCDDDD75CD2A5E9F8064512ED47ADBDDEF7F343FD147B",
                size_bytes=113_445_839,
                source="torchvision Swin_T_Weights.IMAGENET1K_V1",
                license_note="Torchvision model weights; preserve upstream attribution.",
            ),
            official_transform=TransformSpec(
                resize=232,
                crop=224,
                interpolation="bicubic",
                source="torchvision Swin_T_Weights.IMAGENET1K_V1.transforms",
            ),
            aliases=("swin", "swin-t", "swin_tiny"),
        ),
        "retccl": ModelSpec(
            name="retccl",
            manuscript_label="RetCCL",
            architecture="RetCCL.ResNet.resnet50",
            embedding_dim=2048,
            checkpoint=CheckpointSpec(
                filename="best_ckpt.pth",
                sha256="931956F31D3F1A3F6047F3172B9E59EE3460D29F7C0C2BB219CBC8E9207795FF",
                size_bytes=94_367_005,
                source="Xiyue-Wang/RetCCL",
                license_note="GPL-3.0/non-commercial academic use; checkpoint redistribution unverified.",
            ),
            official_transform=TransformSpec(
                resize=256,
                crop=None,
                interpolation="bilinear",
                source="Xiyue-Wang/RetCCL get_feature.py (Resize(256), ImageNet normalization)",
            ),
            architecture_source_sha256=(
                "FF1AB31BF7A9E475A51EE8CB50068B82C9FCE6E2616DC6AFADD6D3884F865A06"
            ),
            aliases=("ret-ccl",),
        ),
        "uni": ModelSpec(
            name="uni",
            manuscript_label="UNI",
            architecture="timm.vit_large_patch16_224",
            embedding_dim=1024,
            checkpoint=CheckpointSpec(
                filename="pytorch_model.bin",
                sha256="56EF09B44A25DC5C7EEDC55551B3D47BCD17659A7A33837CF9ABC9EC4E2FFB40",
                size_bytes=1_213_527_781,
                source="MahmoodLab/UNI",
                license_note="Gated CC BY-NC-ND 4.0; do not redistribute the checkpoint.",
            ),
            official_transform=TransformSpec(
                resize=224,
                crop=None,
                interpolation="bilinear",
                source="MahmoodLab/UNI model card local-checkpoint transform",
            ),
            aliases=("uni-v1",),
        ),
        "uni2_h": ModelSpec(
            name="uni2_h",
            manuscript_label="UNI-2",
            architecture="timm.vit_giant_patch14_224 custom UNI2-H",
            embedding_dim=1536,
            checkpoint=CheckpointSpec(
                filename="pytorch_model.bin",
                sha256="6E077EDA234BEBC595868D918D3458D9DD32A050199B0FF04443B2F46A0A3B1E",
                size_bytes=2_725_669_217,
                source="MahmoodLab/UNI2-h",
                license_note="Gated CC BY-NC-ND 4.0; do not redistribute the checkpoint.",
            ),
            official_transform=TransformSpec(
                resize=224,
                crop=None,
                interpolation="bilinear",
                source="MahmoodLab/UNI2-h model card local-checkpoint transform",
            ),
            aliases=("uni2-h", "uni2h", "uni-2", "uni2"),
        ),
        "prov_gigapath": ModelSpec(
            name="prov_gigapath",
            manuscript_label="Prov-GigaPath",
            architecture="timm.vit_giant_patch14_dinov2 custom patch16 tile encoder",
            embedding_dim=1536,
            checkpoint=CheckpointSpec(
                filename="pytorch_model.bin",
                sha256="877947214318AFA9E011754B74BBC3894A1F480A253AFC7BC8045B8321DEDD63",
                size_bytes=4_540_023_137,
                source=(
                    "prov-gigapath/prov-gigapath tile encoder "
                    "revision 64f9e26c15019f2d4f6d9113c6822f88bb16b01b"
                ),
                license_note=(
                    "Gated Apache-2.0 research checkpoint; accept the upstream access "
                    "terms and do not redistribute the checkpoint from this benchmark."
                ),
            ),
            official_transform=TransformSpec(
                resize=256,
                crop=224,
                interpolation="bicubic",
                source="Prov-GigaPath official tile-encoder inference example",
            ),
            upstream_revision="64f9e26c15019f2d4f6d9113c6822f88bb16b01b",
            upstream_config_sha256=(
                "6937EB16679F8DE311D6EDD824930C27A2B1A1029A24907BE59CED33A908AE55"
            ),
            aliases=("prov-gigapath", "gigapath", "prov_giga_path"),
        ),
        "virchow2": ModelSpec(
            name="virchow2",
            manuscript_label="Virchow2",
            architecture="timm.vit_huge_patch14_224 custom Virchow2",
            # Official representation: CLS (1280) concatenated with the mean
            # of 256 patch tokens (1280), excluding four register tokens.
            embedding_dim=2560,
            checkpoint=CheckpointSpec(
                filename="model.safetensors",
                sha256="8D6CEA947EB2418C3B0DFF48CFB9B238E47744AB0DFCA21B2B0637B140769B4B",
                size_bytes=2_525_001_112,
                source=(
                    "paige-ai/Virchow2 revision "
                    "3158645804b69e3f3bc4439d4116edddf0840a72"
                ),
                license_note=(
                    "Gated CC BY-NC-ND 4.0, non-commercial academic research only; "
                    "do not redistribute the checkpoint."
                ),
            ),
            official_transform=TransformSpec(
                resize=224,
                crop=224,
                interpolation="bicubic",
                source="paige-ai/Virchow2 config.json (crop_pct=1.0, ImageNet mean/std)",
            ),
            upstream_revision="3158645804b69e3f3bc4439d4116edddf0840a72",
            upstream_config_sha256=(
                "7DB445B996BB165E88FE70E826C2EBB530539A2B1D136AA16EEB847DF5F1E3DB"
            ),
            aliases=("virchow-2", "virchow_v2", "virchow-v2"),
        ),
    }
)


def get_model_spec(name: str) -> ModelSpec:
    """Resolve a canonical name or a case-insensitive manuscript alias."""

    normalized = name.strip().lower().replace(" ", "_")
    if normalized in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized]
    for spec in MODEL_REGISTRY.values():
        candidates = (spec.manuscript_label, *spec.aliases)
        if any(normalized == candidate.lower().replace(" ", "_") for candidate in candidates):
            return spec
    choices = ", ".join(MODEL_REGISTRY)
    raise KeyError(f"Unknown model {name!r}; expected one of: {choices}")


def get_transform_spec(model: str | ModelSpec, profile: str = "official") -> TransformSpec:
    """Return the primary official transform or the frozen legacy sensitivity transform."""

    spec = get_model_spec(model) if isinstance(model, str) else model
    normalized = profile.strip().lower().replace("-", "_")
    if normalized in {"official", "official_model_specific", "primary"}:
        return spec.official_transform
    if normalized in {
        "legacy",
        "legacy_common_224",
        "common_square_resize_224",
        "common_square_resize_224_imagenet",
    }:
        return _LEGACY_COMMON
    raise KeyError(
        f"Unknown transform profile {profile!r}; use 'official_model_specific' "
        "or 'legacy_common_224'."
    )


def list_model_names() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)
