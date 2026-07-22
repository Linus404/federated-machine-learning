"""Validate the frozen protocol runtime before model or vectorizer use."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from typing import Any, Mapping

_DETERMINISM_REGISTERED = False


def _runtime_protocol(protocol: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the supplied or repository-frozen scientific protocol.

    Parameters
    ----------
    protocol : mapping of str to Any or None
        Optional parsed protocol supplied by deterministic tests.

    Returns
    -------
    mapping of str to Any
        Protocol containing the runtime contract.
    """
    if protocol is not None:
        return protocol
    from src.evaluation_artifact import load_scientific_protocol

    return load_scientific_protocol()


def validate_protocol_startup_environment(
    protocol: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require frozen environment values that must exist before Python starts.

    Parameters
    ----------
    protocol : mapping of str to Any or None, optional
        Parsed frozen protocol, or the repository protocol when omitted.

    Returns
    -------
    mapping of str to Any
        Protocol used for validation.

    Raises
    ------
    ValueError
        If a required value is missing, conflicting, or was applied after startup.
    """
    frozen = _runtime_protocol(protocol)
    required = frozen["framework"]["execution_environment_before_import"]
    mismatches: list[str] = []
    for name, expected in required.items():
        actual = os.environ.get(name)
        if actual != expected:
            mismatches.append(
                f"{name}: expected {expected}, got {actual if actual is not None else '<missing>'}"
            )
    if required.get("PYTHONHASHSEED") == "0" and sys.flags.hash_randomization != 0:
        mismatches.append("PYTHONHASHSEED was not applied before interpreter startup")
    if mismatches:
        raise ValueError(
            "startup environment differs from the frozen protocol: "
            + "; ".join(mismatches)
        )
    return frozen


def _register_tensorflow_determinism() -> Any:
    """Enable deterministic TensorFlow operations once before application imports.

    Returns
    -------
    Any
        Imported TensorFlow module.
    """
    global _DETERMINISM_REGISTERED
    import tensorflow as tf

    if not _DETERMINISM_REGISTERED:
        tf.config.experimental.enable_op_determinism()
        _DETERMINISM_REGISTERED = True
    return tf


def validate_protocol_runtime(
    protocol: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate exact environment and framework versions before consumption.

    Parameters
    ----------
    protocol : mapping of str to Any or None, optional
        Parsed frozen protocol, or the repository protocol when omitted.

    Returns
    -------
    mapping of str to Any
        Protocol used for validation.

    Raises
    ------
    ValueError
        If required environment values or runtime versions differ.
    """
    frozen = validate_protocol_startup_environment(protocol)

    import keras
    import numpy as np

    tf = _register_tensorflow_determinism()

    framework = frozen["framework"]
    installed = {
        "flower": importlib.metadata.version("flwr"),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
    }
    expected = {
        "flower": frozen["aggregation"]["flower_version"],
        "tensorflow": framework["tensorflow_version"],
        "keras": framework["keras_version"],
        "numpy": framework["numpy_version"],
    }
    mismatches = [
        f"{name}: expected {expected[name]}, got {installed[name]}"
        for name in expected
        if installed[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "framework versions differ from the frozen protocol: "
            + "; ".join(mismatches)
        )
    return frozen


validate_protocol_startup_environment()
_register_tensorflow_determinism()
