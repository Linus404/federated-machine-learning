"""Create and validate immutable training-run provenance manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.app_manifest import load_app_manifest
from src.artifact_compatibility import (
    ARTIFACT_SCHEMA_VERSION,
    validate_artifact_schema,
    write_json_atomically,
)
from src.contracts import DEFAULT_SPLIT_SEED
from src.local_training import DEFAULT_VALIDATION_SEED
from src.paths import resolve_dir, run_manifest_path

CODE_REVISION_ENV = "FML_CODE_REVISION"
RUNTIME_PACKAGES = ("datasets", "flwr", "keras", "numpy", "tensorflow")
REQUIRED_FIELDS = {
    "run_id",
    "flower_run_id",
    "created_at",
    "run_config",
    "environment",
    "code_revision",
    "seeds",
    "dataset",
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Algorithm-prefixed hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_run_config(run_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize Flower run configuration.

    Parameters
    ----------
    run_config : mapping of str to Any
        Configuration supplied at the server run boundary.

    Returns
    -------
    dict of str to Any
        Sorted JSON-compatible configuration.

    Raises
    ------
    ValueError
        If a key or value is outside Flower's scalar configuration contract.
    """
    if not isinstance(run_config, Mapping):
        raise ValueError("run configuration must be a mapping")
    canonical: dict[str, Any] = {}
    for key, value in run_config.items():
        if not isinstance(key, str) or not key:
            raise ValueError("run configuration keys must be non-empty strings")
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if type(value) not in (bool, float, int, str):
            raise ValueError(f"run configuration value for {key!r} must be a scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"run configuration value for {key!r} must be finite")
        canonical[key] = value
    return dict(sorted(canonical.items()))


def _environment_metadata() -> dict[str, Any]:
    """Describe the runtime relevant to model reproduction.

    Returns
    -------
    dict of str to Any
        Python, operating-system, architecture, and package versions.
    """
    packages: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _code_revision() -> dict[str, Any]:
    """Resolve the deployed source revision without requiring Git at runtime.

    Returns
    -------
    dict of str to Any
        Commit identity, worktree state when observable, and evidence source.

    Raises
    ------
    ValueError
        If ``FML_CODE_REVISION`` is not a full Git object ID.
    """
    configured_revision = os.environ.get(CODE_REVISION_ENV)
    if configured_revision:
        normalized = configured_revision.lower()
        if len(normalized) not in (40, 64) or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError(
                f"{CODE_REVISION_ENV} must be a full hexadecimal Git object ID"
            )
        return {"commit": normalized, "dirty": None, "source": CODE_REVISION_ENV}

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "source": "unavailable"}
    return {"commit": commit, "dirty": bool(status), "source": "git"}


def _dataset_metadata(public_artifact_dir: str | Path | None) -> dict[str, Any]:
    """Capture public dataset identity and checksums available to the server.

    Parameters
    ----------
    public_artifact_dir : str or pathlib.Path or None
        Directory containing the public model and vocabulary manifest.

    Returns
    -------
    dict of str to Any
        Available public checksums and an explicit private-data boundary.

    Raises
    ------
    ValueError
        If a declared public artifact escapes its configured directory.
    """
    public_dir = resolve_dir(public_artifact_dir) if public_artifact_dir else None
    private_status = {
        "status": "not_collected",
        "reason": "private client data is outside the server trust boundary",
    }
    if public_dir is None or not (public_dir / "manifest.json").exists():
        return {
            "identity": None,
            "checksums": {},
            "status": "unavailable",
            "private_client_shards": private_status,
        }

    manifest = load_app_manifest(public_artifact_dir=public_dir)
    vocabulary_path = manifest.vocabulary_path.resolve()
    if not vocabulary_path.is_relative_to(public_dir.resolve()):
        raise ValueError("public vocabulary must remain inside its artifact directory")
    identity = manifest.payload.get("dataset") or manifest.payload.get("provenance")
    return {
        "identity": identity,
        "checksums": {
            "manifest.json": _sha256(public_dir / "manifest.json"),
            manifest.vocabulary_path.name: _sha256(vocabulary_path),
        },
        "status": "available",
        "private_client_shards": private_status,
    }


def write_run_provenance_manifest(
    artifact_dir: str | Path,
    run_config: Mapping[str, Any],
    *,
    public_artifact_dir: str | Path | None = None,
    flower_run_id: int | None = None,
    created_at: str | None = None,
) -> Path:
    """Create the immutable provenance manifest for one training run.

    Parameters
    ----------
    artifact_dir : str or pathlib.Path
        Server artifact directory owned by this run.
    run_config : mapping of str to Any
        Complete Flower run configuration.
    public_artifact_dir : str or pathlib.Path or None, optional
        Public artifacts used by the run.
    flower_run_id : int or None, optional
        Flower's infrastructure-level run identifier when available.
    created_at : str or None, optional
        UTC timestamp override used by deterministic tests.

    Returns
    -------
    pathlib.Path
        Path to the newly published manifest.

    Raises
    ------
    FileExistsError
        If this artifact directory already owns a run manifest.
    ValueError
        If boundary data is invalid.
    """
    if flower_run_id is not None and (
        type(flower_run_id) is not int or flower_run_id < 0
    ):
        raise ValueError("Flower run ID must be a non-negative integer")
    config = _canonical_run_config(run_config)
    timestamp = created_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    seed_config = {key: value for key, value in config.items() if "seed" in key.lower()}
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "flower_run_id": flower_run_id,
        "created_at": timestamp,
        "run_config": config,
        "environment": _environment_metadata(),
        "code_revision": _code_revision(),
        "seeds": {
            "run_config": seed_config,
            "code_defaults": {
                "client_validation_split": DEFAULT_VALIDATION_SEED,
                "data_partition": DEFAULT_SPLIT_SEED,
            },
        },
        "dataset": _dataset_metadata(public_artifact_dir),
    }
    return write_json_atomically(
        run_manifest_path(artifact_dir),
        payload,
        overwrite=False,
    )


def load_run_provenance_manifest(path: str | Path) -> Mapping[str, Any]:
    """Load and validate one training-run provenance manifest.

    Parameters
    ----------
    path : str or pathlib.Path
        Manifest path.

    Returns
    -------
    collections.abc.Mapping
        Validated manifest payload.

    Raises
    ------
    ValueError
        If the schema or required provenance fields are invalid.
    """
    manifest_path = Path(path)
    try:
        payload = validate_artifact_schema(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            "run provenance manifest",
        )
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid run provenance manifest: {manifest_path}") from error
    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        raise ValueError(
            "run provenance manifest is missing: " + ", ".join(sorted(missing))
        )
    if not isinstance(payload["run_id"], str):
        raise ValueError("run provenance manifest has an invalid run_id")
    try:
        parsed_run_id = uuid.UUID(payload["run_id"])
    except ValueError as error:
        raise ValueError("run provenance manifest has an invalid run_id") from error
    if parsed_run_id.version != 4 or str(parsed_run_id) != payload["run_id"]:
        raise ValueError("run provenance manifest has an invalid run_id")
    flower_run_id = payload["flower_run_id"]
    if flower_run_id is not None and (
        type(flower_run_id) is not int or flower_run_id < 0
    ):
        raise ValueError("run provenance manifest has an invalid flower_run_id")
    created_at = payload["created_at"]
    if not isinstance(created_at, str):
        raise ValueError("run provenance manifest has an invalid created_at")
    try:
        parsed_creation_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("run provenance manifest has an invalid created_at") from error
    if parsed_creation_time.tzinfo is None:
        raise ValueError("run provenance manifest has an invalid created_at")
    for field in ("run_config", "environment", "code_revision", "seeds", "dataset"):
        if not isinstance(payload[field], Mapping):
            raise ValueError(f"run provenance manifest has an invalid {field}")
    _canonical_run_config(payload["run_config"])
    return payload
