"""Configuration loading and validation."""

from __future__ import annotations

import json
import os

DEFAULT_TIMEOUT = 30.0


class ConfigError(ValueError):
    """Raised when a configuration value is missing or out of range."""


def load_config(path: str, env_prefix: str = "TASKQ_") -> dict:
    """Read a JSON config file, then let environment variables starting with ``env_prefix`` override it."""
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            cfg[key[len(env_prefix):].lower()] = value
    return cfg


def validate_config(cfg: dict) -> None:
    """Raise ConfigError when worker_count or visibility_timeout is out of range."""
    if int(cfg.get("worker_count", 1)) < 1:
        raise ConfigError("worker_count must be at least 1")
    if float(cfg.get("visibility_timeout", DEFAULT_TIMEOUT)) <= 0:
        raise ConfigError("visibility_timeout must be positive")
