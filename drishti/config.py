"""Configuration loading: packaged defaults <- user YAML <- environment overrides."""

from __future__ import annotations

import copy
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict with attribute access for nested sections (cfg.video.max_age)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(dict(self), memo))

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict) and not isinstance(obj, Config):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = Config.wrap(v)
            return obj
        return obj


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _env_overrides(prefix: str = "DRISHTI__") -> dict:
    """DRISHTI__VIDEO__MAX_AGE=45 -> {"video": {"max_age": 45}}."""
    result: dict = {}
    for name, raw in os.environ.items():
        if not name.startswith(prefix):
            continue
        parts = [p.lower() for p in name[len(prefix):].split("__") if p]
        if not parts:
            continue
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(raw)
    return result


def default_config_dict() -> dict:
    text = resources.files("drishti").joinpath("default.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_config(path: str | os.PathLike | list | tuple | None = None, overrides: dict | None = None) -> Config:
    """Defaults <- each config file in order (later wins) <- DRISHTI__* env <- overrides."""
    data = default_config_dict()
    for p in ([path] if isinstance(path, (str, os.PathLike)) else list(path or [])):
        with open(p, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        data = _deep_merge(data, user)
    data = _deep_merge(data, _env_overrides())
    if overrides:
        data = _deep_merge(data, overrides)
    return Config.wrap(data)


def expand_path(p: str | os.PathLike) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))
