"""Configuration loading and path resolution for PAIR-BST."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and annotate it with its source directory."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {source}")
    value = deepcopy(value)
    value["_source_path"] = str(source)
    value["_source_dir"] = str(source.parent)
    return value


def resolve_path(value: str | Path, *, base: str | Path) -> Path:
    """Resolve a possibly relative path against a configuration directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def get_path(config: Mapping[str, Any], *keys: str) -> Path:
    """Read a nested path field and resolve it relative to the YAML file."""
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            dotted = ".".join(keys)
            raise KeyError(f"Missing configuration path: {dotted}")
        current = current[key]
    return resolve_path(current, base=config.get("_source_dir", "."))

