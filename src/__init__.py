"""IMDB Movie Sentiment Analysis Federated ML."""

from __future__ import annotations

from configparser import ConfigParser
from typing import Any


def parse_run_config_bool(value: Any, default: bool = False) -> bool:
    """Parse Flower run-config booleans from native bools or strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ConfigParser.BOOLEAN_STATES:
            return ConfigParser.BOOLEAN_STATES[normalized]
    raise ValueError(f"Expected a boolean-compatible value, got {value!r}")
