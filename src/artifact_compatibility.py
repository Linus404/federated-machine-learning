"""Artifact schema compatibility checks shared by producers and consumers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

ARTIFACT_SCHEMA_VERSION = 1
SERVER_ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
SERVER_ARTIFACTS: dict[str, dict[str, Any]] = {
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


def write_json_atomically(
    path: Path, payload: Mapping[str, Any], *, overwrite: bool = True
) -> Path:
    """Persist a JSON object atomically.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    payload : mapping of str to Any
        JSON-compatible object to persist.
    overwrite : bool, optional
        Replace an existing destination when ``True``. When ``False``, publish by
        an atomic hard link so an existing immutable file cannot be replaced.

    Returns
    -------
    pathlib.Path
        Destination path.

    Raises
    ------
    FileExistsError
        If ``overwrite`` is ``False`` and the destination already exists.
    ValueError
        If the payload contains values outside the JSON data model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.chmod(0o644)
        if overwrite:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
            temporary_path.unlink()
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


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
    return write_json_atomically(
        path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifacts": SERVER_ARTIFACTS,
        },
    )


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
    artifacts = payload.get("artifacts")
    mismatch_message = (
        "server artifact manifest does not match schema_version "
        f"{ARTIFACT_SCHEMA_VERSION}; regenerate its artifacts"
    )
    if not isinstance(artifacts, Mapping):
        raise ValueError(mismatch_message)
    for name, layout in SERVER_ARTIFACTS.items():
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping) or any(
            artifact.get(key) != value for key, value in layout.items()
        ):
            raise ValueError(mismatch_message)
    return payload
