"""YAML configuration loading with deep-merge and environment-variable overrides.

Override convention: ``TRAJECTORYOS__section__key=value`` sets ``cfg["section"]["key"]``.
Path segments are split on double underscores and lowercased; values are parsed with
``yaml.safe_load`` so ``"8"`` becomes an int, ``"true"`` a bool, ``"[1, 2]"`` a list.
Validated configs use Pydantic models with ``extra="forbid"``, so unknown keys —
including mistyped override paths — fail loudly instead of being ignored.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

DEFAULT_ENV_PREFIX = "TRAJECTORYOS"

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict. An empty file yields ``{}``."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top-level YAML must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (non-mutating; override wins)."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_env_overrides(
    config: Mapping[str, Any],
    *,
    prefix: str = DEFAULT_ENV_PREFIX,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Overlay ``PREFIX__a__b=value`` environment variables onto ``config``."""
    if env is None:
        env = os.environ
    marker = f"{prefix}__"
    result: dict[str, Any] = dict(config)
    for name, raw in env.items():
        if not name.upper().startswith(marker.upper()):
            continue
        path = [seg.lower() for seg in name[len(marker) :].split("__") if seg]
        if not path:
            continue
        override: dict[str, Any] = {path[-1]: yaml.safe_load(raw)}
        for seg in reversed(path[:-1]):
            override = {seg: override}
        result = deep_merge(result, override)
    return result


def load_config(
    path: str | Path,
    model: type[ModelT],
    *,
    env_prefix: str = DEFAULT_ENV_PREFIX,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ModelT:
    """Load a YAML file, apply env + explicit overrides, validate into ``model``.

    Precedence (lowest to highest): file < environment < ``overrides``.
    """
    data = load_yaml(path)
    data = apply_env_overrides(data, prefix=env_prefix, env=env)
    if overrides:
        data = deep_merge(data, overrides)
    return model.model_validate(data)
