"""Create and validate immutable training-run provenance manifests."""

from __future__ import annotations

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

from src.app_manifest import (
    AppManifest,
    PUBLIC_ARTIFACT_FILENAMES,
    expected_train_dataset,
    load_app_manifest,
    resolve_public_artifact_dir,
)
from src.artifact_compatibility import (
    ARTIFACT_SCHEMA_VERSION,
    sha256_bytes,
    strict_json_loads,
    validate_artifact_schema,
    write_json_atomically,
)
from src.contracts import DEFAULT_SPLIT_SEED, DEFAULT_VALIDATION_SEED
from src.paths import run_manifest_path

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
ENVIRONMENT_FIELDS = {
    "python_version",
    "python_implementation",
    "operating_system",
    "operating_system_release",
    "machine",
    "packages",
}


def _required_nested_fields(
    value: Mapping[str, Any], required: set[str], field: str
) -> None:
    """Require nested manifest fields.

    Parameters
    ----------
    value : mapping of str to Any
        Nested manifest object.
    required : set of str
        Required keys.
    field : str
        Dotted field name used in validation errors.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a required key is absent.
    """
    missing = required - value.keys()
    if missing:
        raise ValueError(
            f"run provenance manifest is missing {field}: " + ", ".join(sorted(missing))
        )


def _validate_environment(environment: Mapping[str, Any]) -> None:
    """Validate recorded runtime metadata.

    Parameters
    ----------
    environment : mapping of str to Any
        Runtime metadata from the manifest.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If required runtime metadata is missing or malformed.
    """
    _required_nested_fields(environment, ENVIRONMENT_FIELDS, "environment")
    for field in ENVIRONMENT_FIELDS - {"packages"}:
        if not isinstance(environment[field], str):
            raise ValueError(
                f"run provenance manifest has an invalid environment.{field}"
            )
    packages = environment["packages"]
    if not isinstance(packages, Mapping):
        raise ValueError("run provenance manifest has an invalid environment.packages")
    _required_nested_fields(packages, set(RUNTIME_PACKAGES), "environment.packages")
    if any(
        not isinstance(name, str)
        or not name
        or (version is not None and not isinstance(version, str))
        for name, version in packages.items()
    ):
        raise ValueError("run provenance manifest has an invalid environment.packages")


def _validate_code_revision(code_revision: Mapping[str, Any]) -> None:
    """Validate source-revision evidence.

    Parameters
    ----------
    code_revision : mapping of str to Any
        Source revision object from the manifest.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If revision evidence is missing, malformed, or inconsistent.
    """
    _required_nested_fields(
        code_revision, {"commit", "dirty", "source"}, "code_revision"
    )
    commit = code_revision["commit"]
    dirty = code_revision["dirty"]
    source = code_revision["source"]
    if commit is not None and (
        not isinstance(commit, str)
        or len(commit) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("run provenance manifest has an invalid code_revision.commit")
    if dirty is not None and type(dirty) is not bool:
        raise ValueError("run provenance manifest has an invalid code_revision.dirty")
    if source not in {"git", CODE_REVISION_ENV, "unavailable"}:
        raise ValueError("run provenance manifest has an invalid code_revision.source")
    if source == "git" and (commit is None or dirty is None):
        raise ValueError("run provenance manifest has inconsistent code_revision")
    if source == CODE_REVISION_ENV and (commit is None or dirty is not None):
        raise ValueError("run provenance manifest has inconsistent code_revision")
    if source == "unavailable" and (commit is not None or dirty is not None):
        raise ValueError("run provenance manifest has inconsistent code_revision")


def _validate_seeds(seeds: Mapping[str, Any]) -> None:
    """Validate recorded run and code-default seeds.

    Parameters
    ----------
    seeds : mapping of str to Any
        Seed metadata from the manifest.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If seed metadata is missing or malformed.
    """
    _required_nested_fields(seeds, {"run_config", "code_defaults"}, "seeds")
    run_config = seeds["run_config"]
    code_defaults = seeds["code_defaults"]
    if not isinstance(run_config, Mapping):
        raise ValueError("run provenance manifest has an invalid seeds.run_config")
    _canonical_run_config(run_config)
    if any("seed" not in key.lower() for key in run_config):
        raise ValueError("run provenance manifest has an invalid seeds.run_config")
    if not isinstance(code_defaults, Mapping):
        raise ValueError("run provenance manifest has an invalid seeds.code_defaults")
    _required_nested_fields(
        code_defaults,
        {"client_validation_split", "data_partition"},
        "seeds.code_defaults",
    )
    if any(type(code_defaults[key]) is not int for key in code_defaults):
        raise ValueError("run provenance manifest has an invalid seeds.code_defaults")


def _canonical_dataset_identity(identity: Mapping[str, Any]) -> str:
    """Serialize embedded public-dataset identity in its canonical string form.

    Parameters
    ----------
    identity : mapping of str to Any
        Validated public train-dataset identity.

    Returns
    -------
    str
        Compact, key-sorted JSON without insignificant whitespace.
    """
    return json.dumps(
        dict(identity),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_dataset_identity(identity: str) -> None:
    """Strictly decode and validate embedded public-dataset identity JSON.

    Parameters
    ----------
    identity : str
        Canonical JSON string stored in ``dataset.identity``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If decoding, schema, values, or canonical serialization are invalid.
    """
    try:
        decoded = strict_json_loads(
            identity, source="run provenance manifest dataset.identity"
        )
    except ValueError as error:
        raise ValueError(
            "run provenance manifest has an invalid dataset.identity"
        ) from error
    expected = expected_train_dataset()
    if (
        not isinstance(decoded, Mapping)
        or decoded != expected
        or identity != _canonical_dataset_identity(decoded)
    ):
        raise ValueError("run provenance manifest has an invalid dataset.identity")


def _validate_dataset(dataset: Mapping[str, Any]) -> None:
    """Validate public-dataset provenance and the private-data boundary.

    Parameters
    ----------
    dataset : mapping of str to Any
        Dataset metadata from the manifest.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If dataset provenance is missing or malformed.
    """
    _required_nested_fields(
        dataset,
        {
            "identity",
            "checksums",
            "public_manifest",
            "status",
            "private_client_shards",
        },
        "dataset",
    )
    identity = dataset["identity"]
    checksums = dataset["checksums"]
    status = dataset["status"]
    authoritative_public_manifest = dataset["public_manifest"]
    private_shards = dataset["private_client_shards"]
    if identity is not None and (not isinstance(identity, str) or not identity):
        raise ValueError("run provenance manifest has an invalid dataset.identity")
    if identity is not None:
        _validate_dataset_identity(identity)
    if not isinstance(checksums, Mapping) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(checksum, str)
        or not checksum.startswith("sha256:")
        or len(checksum) != 71
        or any(character not in "0123456789abcdef" for character in checksum[7:])
        for name, checksum in checksums.items()
    ):
        raise ValueError("run provenance manifest has an invalid dataset.checksums")
    if status not in {"available", "unavailable"}:
        raise ValueError("run provenance manifest has an invalid dataset.status")
    if status == "available" and (identity is None or not checksums):
        raise ValueError("run provenance manifest has inconsistent dataset metadata")
    if status == "available" and set(checksums) != PUBLIC_ARTIFACT_FILENAMES:
        raise ValueError("run provenance manifest has an invalid dataset.checksums")
    if status == "unavailable" and (
        identity is not None or checksums or authoritative_public_manifest is not None
    ):
        raise ValueError("run provenance manifest has inconsistent dataset metadata")
    if status == "available" and (
        not isinstance(authoritative_public_manifest, Mapping)
        or set(authoritative_public_manifest) != {"filename", "size_bytes", "checksum"}
        or authoritative_public_manifest["filename"] != "manifest.json"
        or type(authoritative_public_manifest["size_bytes"]) is not int
        or authoritative_public_manifest["size_bytes"] < 1
        or authoritative_public_manifest["checksum"] != checksums.get("manifest.json")
    ):
        raise ValueError(
            "run provenance manifest has an invalid dataset.public_manifest"
        )
    if not isinstance(private_shards, Mapping):
        raise ValueError(
            "run provenance manifest has an invalid dataset.private_client_shards"
        )
    private_status = private_shards.get("status")
    if private_status == "not_collected":
        if (
            set(private_shards) != {"status", "reason"}
            or not isinstance(private_shards["reason"], str)
            or not private_shards["reason"]
        ):
            raise ValueError(
                "run provenance manifest has an invalid dataset.private_client_shards"
            )
    elif private_status == "available":
        if set(private_shards) != {"status", "identity", "checksums"}:
            raise ValueError(
                "run provenance manifest has an invalid dataset.private_client_shards"
            )
        shard_identity = private_shards["identity"]
        shard_checksums = private_shards["checksums"]
        if not isinstance(shard_identity, Mapping) or not isinstance(
            shard_checksums, Mapping
        ):
            raise ValueError(
                "run provenance manifest has an invalid dataset.private_client_shards"
            )
        identity_fields = {
            "client_id",
            "dataset",
            "source_split",
            "row_identity",
            "sample_count",
            "label_histogram",
            "public_manifest",
        }
        if set(shard_identity) != identity_fields:
            raise ValueError(
                "run provenance manifest has an invalid dataset.private_client_shards"
            )
        shard_dataset = shard_identity["dataset"]
        shard_histogram = shard_identity["label_histogram"]
        public_manifest = shard_identity["public_manifest"]
        if (
            type(shard_identity["client_id"]) is not int
            or shard_identity["client_id"] < 0
            or shard_identity["source_split"] != "train"
            or shard_identity["row_identity"]
            != "train:{zero_based_official_split_row_index}"
            or type(shard_identity["sample_count"]) is not int
            or shard_identity["sample_count"] < 1
            or not isinstance(shard_dataset, Mapping)
            or dict(shard_dataset) != expected_train_dataset()
            or not isinstance(shard_histogram, Mapping)
            or not shard_histogram
            or any(
                label not in {"0", "1"} or type(count) is not int or count < 1
                for label, count in shard_histogram.items()
            )
            or sum(shard_histogram.values()) != shard_identity["sample_count"]
            or not isinstance(public_manifest, Mapping)
            or set(public_manifest) != {"filename", "size_bytes", "checksum"}
            or public_manifest["filename"] != "manifest.json"
            or type(public_manifest["size_bytes"]) is not int
            or public_manifest["size_bytes"] < 1
            or not isinstance(public_manifest["checksum"], str)
            or len(public_manifest["checksum"]) != 71
            or not public_manifest["checksum"].startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in public_manifest["checksum"][7:]
            )
        ):
            raise ValueError(
                "run provenance manifest has an invalid dataset.private_client_shards"
            )
        if (
            status != "available"
            or not isinstance(authoritative_public_manifest, Mapping)
            or dict(public_manifest) != dict(authoritative_public_manifest)
        ):
            raise ValueError(
                "run provenance private shard public manifest binding differs from "
                "the authoritative public manifest"
            )
        if any(
            not isinstance(name, str)
            or not isinstance(checksum, str)
            or len(checksum) != 71
            or not checksum.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in checksum[7:])
            for name, checksum in shard_checksums.items()
        ):
            raise ValueError(
                "run provenance manifest has an invalid dataset.private_client_shards"
            )
    else:
        raise ValueError(
            "run provenance manifest has an invalid dataset.private_client_shards"
        )


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
    operating_system = platform.system()
    packages: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        distributions = (
            ("tensorflow", "tensorflow-cpu")
            if package == "tensorflow" and operating_system == "Linux"
            else (package,)
        )
        packages[package] = None
        for distribution in distributions:
            try:
                packages[package] = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": operating_system,
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
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "source": "unavailable"}
    return {"commit": commit, "dirty": bool(status), "source": "git"}


def _dataset_metadata(
    public_artifact_dir: str | Path | None,
    client_shard: Mapping[str, Any] | None = None,
    app_manifest: AppManifest | None = None,
) -> dict[str, Any]:
    """Capture public dataset identity and checksums available to the server.

    Parameters
    ----------
    public_artifact_dir : str or pathlib.Path or None
        Directory containing the public model and vocabulary manifest.
    client_shard : mapping of str to Any or None, optional
        Validated private-shard identity and checksums available to local training.
    app_manifest : AppManifest or None, optional
        Already validated public snapshot retained by this operation.

    Returns
    -------
    dict of str to Any
        Available public checksums and an explicit private-data boundary.

    Raises
    ------
    ValueError
        If a declared public artifact escapes its configured directory.
    """
    public_dir = None
    if app_manifest is None and public_artifact_dir:
        public_dir = resolve_public_artifact_dir(
            public_artifact_dir=public_artifact_dir
        )
    private_status = (
        {
            "status": "available",
            "identity": client_shard["identity"],
            "checksums": client_shard["checksums"],
        }
        if client_shard is not None
        else {
            "status": "not_collected",
            "reason": "client shard identity is not collected by the server application",
        }
    )
    if app_manifest is None and (
        public_dir is None or not (public_dir / "manifest.json").exists()
    ):
        return {
            "identity": None,
            "checksums": {},
            "public_manifest": None,
            "status": "unavailable",
            "private_client_shards": private_status,
        }

    manifest = app_manifest or load_app_manifest(public_artifact_dir=public_dir)
    identity = _canonical_dataset_identity(manifest.payload["dataset"])
    return {
        "identity": identity,
        "checksums": {
            "manifest.json": sha256_bytes(manifest.manifest_bytes),
            manifest.vocabulary_path.name: sha256_bytes(manifest.vocabulary_bytes),
        },
        "public_manifest": {
            "filename": "manifest.json",
            "size_bytes": len(manifest.manifest_bytes),
            "checksum": sha256_bytes(manifest.manifest_bytes),
        },
        "status": "available",
        "private_client_shards": private_status,
    }


def write_run_provenance_manifest(
    artifact_dir: str | Path,
    run_config: Mapping[str, Any],
    *,
    public_artifact_dir: str | Path | None = None,
    client_shard: Mapping[str, Any] | None = None,
    app_manifest: AppManifest | None = None,
    flower_run_id: int | None = None,
    created_at: str | None = None,
    run_id: str | None = None,
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
    client_shard : mapping of str to Any or None, optional
        Validated private shard evidence available only to local training.
    app_manifest : AppManifest or None, optional
        Already validated public snapshot retained by this operation.
    flower_run_id : int or None, optional
        Flower's infrastructure-level run identifier when available.
    created_at : str or None, optional
        UTC timestamp override used by deterministic tests.
    run_id : str or None, optional
        UUID override used to align a versioned directory with its manifest.

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
    resolved_run_id = run_id or str(uuid.uuid4())
    try:
        parsed_run_id = uuid.UUID(resolved_run_id)
    except ValueError as error:
        raise ValueError("run ID must be a canonical UUID") from error
    if parsed_run_id.version != 4 or str(parsed_run_id) != resolved_run_id:
        raise ValueError("run ID must be a canonical UUID4")
    payload: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": resolved_run_id,
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
        "dataset": _dataset_metadata(
            public_artifact_dir, client_shard, app_manifest=app_manifest
        ),
    }
    _validate_environment(payload["environment"])
    _validate_code_revision(payload["code_revision"])
    _validate_seeds(payload["seeds"])
    _validate_dataset(payload["dataset"])
    return write_json_atomically(
        run_manifest_path(artifact_dir),
        payload,
        overwrite=False,
    )


def load_run_provenance_manifest(
    path: str | Path, *, manifest_bytes: bytes | None = None
) -> Mapping[str, Any]:
    """Load and validate one training-run provenance manifest.

    Parameters
    ----------
    path : str or pathlib.Path
        Manifest path.
    manifest_bytes : bytes or None, optional
        Exact manifest bytes already read through a secure snapshot boundary.

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
        document = (
            manifest_path.read_bytes() if manifest_bytes is None else manifest_bytes
        )
    except OSError as error:
        raise ValueError(f"invalid run provenance manifest: {manifest_path}") from error
    payload = validate_artifact_schema(
        strict_json_loads(
            document,
            source=f"run provenance manifest: {manifest_path}",
        ),
        "run provenance manifest",
    )
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
    _validate_environment(payload["environment"])
    _validate_code_revision(payload["code_revision"])
    _validate_seeds(payload["seeds"])
    _validate_dataset(payload["dataset"])
    return payload
