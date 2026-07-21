"""Artifact schema compatibility checks shared by producers and consumers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

ARTIFACT_SCHEMA_VERSION = 1
SERVER_ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
SERVER_ARTIFACTS = {
    "model": {"filename": "global_model.keras", "format": "keras-v3"},
    "metrics": {
        "filename": "metrics.csv",
        "columns": ["round", "loss", "accuracy"],
    },
    "client_metrics": {
        "filename": "client_metrics.csv",
        "columns": ["round", "client_id", "loss", "accuracy", "samples"],
    },
}


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


def write_server_artifact_manifest(artifact_dir: Path) -> Path:
    """Write the compatibility contract for one server artifact directory.

    Parameters
    ----------
    artifact_dir : pathlib.Path
        Server output directory.

    Returns
    -------
    pathlib.Path
        Path to the written artifact manifest.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / SERVER_ARTIFACT_MANIFEST_FILENAME
    path.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifacts": SERVER_ARTIFACTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_server_artifact_manifest(artifact_dir: Path) -> Mapping[str, Any]:
    """Load and validate one server artifact directory contract.

    Parameters
    ----------
    artifact_dir : pathlib.Path
        Directory containing ``artifact_manifest.json``.

    Returns
    -------
    collections.abc.Mapping
        Validated server artifact manifest.

    Raises
    ------
    ValueError
        If the schema or declared artifact layouts are incompatible.
    """
    path = artifact_dir / SERVER_ARTIFACT_MANIFEST_FILENAME
    if not path.exists():
        raise ValueError(
            "server artifact manifest has no valid schema_version; regenerate its "
            "artifacts"
        )
    payload = validate_artifact_schema(
        json.loads(path.read_text(encoding="utf-8")), "server artifact manifest"
    )
    if payload.get("artifacts") != SERVER_ARTIFACTS:
        raise ValueError(
            "server artifact manifest does not match schema_version "
            f"{ARTIFACT_SCHEMA_VERSION}; regenerate its artifacts"
        )
    return payload
