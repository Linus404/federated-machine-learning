"""Validate the frozen protocol runtime before model or vectorizer use."""

from __future__ import annotations

import os
from typing import Any, Mapping


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


def configure_protocol_environment(
    protocol: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Set absent pre-import values and reject conflicting environment values.

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
        If a required environment value was explicitly set to another value.
    """
    frozen = _runtime_protocol(protocol)
    required = frozen["framework"]["execution_environment_before_import"]
    mismatches = []
    for name, expected in required.items():
        actual = os.environ.setdefault(name, expected)
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, got {actual}")
    if mismatches:
        raise ValueError(
            "runtime environment differs from the frozen protocol: "
            + "; ".join(mismatches)
        )
    return frozen


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
    frozen = configure_protocol_environment(protocol)

    import keras
    import numpy as np
    import tensorflow as tf

    framework = frozen["framework"]
    installed = {
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
    }
    expected = {
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


configure_protocol_environment()
