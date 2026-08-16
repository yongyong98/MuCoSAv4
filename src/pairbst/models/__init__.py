"""Frozen-encoder model definitions used by the PAIR-BST benchmark."""

from .adapters import (
    CheckpointVerification,
    build_model,
    build_transform,
    forward_embeddings,
    verify_checkpoint,
)
from .registry import (
    MODEL_REGISTRY,
    CheckpointSpec,
    ModelSpec,
    TransformSpec,
    get_model_spec,
    get_transform_spec,
    list_model_names,
)

__all__ = [
    "MODEL_REGISTRY",
    "CheckpointSpec",
    "CheckpointVerification",
    "ModelSpec",
    "TransformSpec",
    "build_model",
    "build_transform",
    "forward_embeddings",
    "get_model_spec",
    "get_transform_spec",
    "list_model_names",
    "verify_checkpoint",
]
