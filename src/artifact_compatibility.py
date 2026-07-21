"""Artifact schema compatibility checks shared by producers and consumers."""

from __future__ import annotations

from typing import Any, Mapping

ARTIFACT_SCHEMA_VERSION = 1


def validate_artifact_schema(payload: object, artifact_name: str) -> Mapping[str, Any]:
    """Validate an artifact manifest against the supported schema.

    Parameters
    ----------
    payload : object
        Decoded JSON payload to validate.
    artifact_name : str
        Human-readable artifact name used in error messages.

    Returns
    -------
    collections.abc.Mapping
        The validated mapping.

    Raises
    ------
    ValueError
        If the payload is not a mapping or its schema is not supported.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"{artifact_name} must be a JSON object")

    version = payload.get("schema_version")
    if type(version) is not int:
        raise ValueError(
            f"{artifact_name} has no valid schema_version; regenerate its artifacts"
        )
    if version != ARTIFACT_SCHEMA_VERSION:
        direction = "older" if version < ARTIFACT_SCHEMA_VERSION else "newer"
        raise ValueError(
            f"{artifact_name} schema_version {version} is {direction} than supported "
            f"version {ARTIFACT_SCHEMA_VERSION}; regenerate with this project version"
        )

    return payload
