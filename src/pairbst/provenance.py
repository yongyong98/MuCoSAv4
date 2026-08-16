"""Small, dependency-light provenance helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_VALID_CUBLAS_WORKSPACE_CONFIGS = {":4096:8", ":16:8"}


def configure_deterministic_cuda_environment() -> str:
    """Set and validate PyTorch's deterministic cuBLAS workspace contract.

    This function must be called before CUDA is initialized.  It deliberately
    refuses an incompatible pre-existing value instead of silently producing a
    run whose determinism differs from the frozen protocol.
    """

    current = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if current not in _VALID_CUBLAS_WORKSPACE_CONFIGS:
        raise RuntimeError(
            "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG to be "
            f"':4096:8' or ':16:8'; got {current!r}."
        )
    return current


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and Torch when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_torch:
        configure_deterministic_cuda_environment()
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def environment_snapshot() -> dict[str, Any]:
    data: dict[str, Any] = {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for name in ("numpy", "pandas", "sklearn", "scipy", "h5py", "PIL", "torch", "torchvision", "timm"):
        try:
            module = __import__(name)
            data[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # provenance must not make a run crash
            data[name] = f"unavailable: {type(exc).__name__}"
    try:
        import torch

        data["cuda_available"] = bool(torch.cuda.is_available())
        data["cuda_version"] = torch.version.cuda
        data["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return data


def write_json_atomic(payload: Any, output: str | Path) -> Path:
    """Write JSON through a sibling temporary file and atomically replace."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)
    return destination
