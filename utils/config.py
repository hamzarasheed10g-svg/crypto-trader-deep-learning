"""Load and validate the YAML configuration.

Usage
-----
>>> from utils.config import load_config
>>> cfg = load_config("configs/default.yaml")
>>> cfg.lstm.hidden_size
128

Implementation notes
--------------------
We use ``types.SimpleNamespace`` recursively for attribute-style access (``cfg.lstm.hidden_size``)
while keeping the original ``dict`` available at ``cfg._raw`` for serialisation.
This avoids a heavyweight dependency on pydantic at the config layer.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for attribute access."""
    if isinstance(obj, Mapping):
        ns = SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
        # Keep the underlying dict accessible for round-tripping
        ns._raw = dict(obj)  # type: ignore[attr-defined]
        return ns
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | os.PathLike | None = None) -> SimpleNamespace:
    """Load the YAML config file at ``path`` (defaults to configs/default.yaml)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _to_namespace(raw)
    cfg._path = str(cfg_path)  # type: ignore[attr-defined]
    cfg._project_root = str(PROJECT_ROOT)  # type: ignore[attr-defined]
    return cfg


def resolve_path(cfg: SimpleNamespace, relative: str) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return Path(cfg._project_root) / p  # type: ignore[attr-defined]
