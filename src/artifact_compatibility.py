"""Artifact schema compatibility checks shared by producers and consumers."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from src.app_manifest import AppManifest

ARTIFACT_SCHEMA_VERSION = 1
PUBLIC_ARTIFACT_SCHEMA_VERSION = 2
CLIENT_SHARD_SCHEMA_VERSION = 2
SERVER_ARTIFACT_SCHEMA_VERSION = 2
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
REQUIRED_COMPLETED_ARTIFACTS = {
    "global_model.keras",
    "metrics.csv",
    "run_manifest.json",
}
_SERVER_BINDING_FIELDS = {
    "public_manifest_checksum",
    "vocabulary_checksum",
    "model_dimensions",
}
_MODEL_DIMENSION_FIELDS = {"vocabulary_size", "sequence_length", "embedding_dim"}


@dataclass(frozen=True)
class ServerArtifactSnapshot:
    """Validated immutable bytes for one server artifact set.

    Parameters
    ----------
    directory : pathlib.Path
        Canonical source directory used only for display and diagnostics.
    manifest : mapping of str to Any
        Validated artifact manifest payload.
    files : mapping of str to bytes
        Artifact bytes read and verified during the same validation pass.
    """

    directory: Path
    manifest: Mapping[str, Any]
    files: Mapping[str, bytes]


def sha256_bytes(content: bytes) -> str:
    """Return the algorithm-prefixed SHA-256 digest of bytes.

    Parameters
    ----------
    content : bytes
        Content to hash.

    Returns
    -------
    str
        Algorithm-prefixed hexadecimal digest.
    """
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def deep_freeze(value: Any) -> Any:
    """Recursively convert JSON containers into immutable equivalents.

    Parameters
    ----------
    value : Any
        Validated JSON-compatible value.

    Returns
    -------
    Any
        Mappings wrapped in read-only proxies, sequences converted to tuples,
        and scalar values unchanged.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    return value


def _mutable_json(value: Any) -> Any:
    """Convert immutable JSON containers back to encoder-native containers.

    Parameters
    ----------
    value : Any
        Recursively frozen JSON-compatible value.

    Returns
    -------
    Any
        Dictionaries, lists, and unchanged scalar values.
    """
    if isinstance(value, Mapping):
        return {key: _mutable_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json(item) for item in value]
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a JSON object in the repository artifact format.

    Parameters
    ----------
    payload : mapping of str to Any
        JSON-compatible object.

    Returns
    -------
    bytes
        Indented UTF-8 JSON with one trailing LF byte.
    """
    return (
        json.dumps(_mutable_json(payload), indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def read_regular_file(path: Path, *, parent: Path) -> bytes:
    """Read one contained regular file without following a final symlink.

    Parameters
    ----------
    path : pathlib.Path
        File to read.
    parent : pathlib.Path
        Canonical directory that must directly contain the file.

    Returns
    -------
    bytes
        Exact bytes read from the opened file descriptor.

    Raises
    ------
    ValueError
        If the path escapes its parent or is not a regular file.
    """
    canonical_parent = parent.resolve(strict=True)
    if path.parent.resolve(strict=True) != canonical_parent or path.is_symlink():
        raise ValueError(f"artifact must be a contained regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    parent_descriptor = -1
    try:
        if os.open in os.supports_dir_fd:
            parent_descriptor = os.open(
                canonical_parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        else:
            descriptor = os.open(path, flags)
    except OSError as error:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        raise ValueError(
            f"artifact must be a contained regular file: {path.name}"
        ) from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise ValueError(f"artifact must be a contained regular file: {path.name}")
        with os.fdopen(descriptor, "rb") as file:
            descriptor = -1
            return file.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def sha256_file(path: Path) -> str:
    """Return the algorithm-prefixed SHA-256 digest of one file.

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
    content = canonical_json_bytes(payload).decode("utf-8")
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


def validate_artifact_schema(
    payload: object,
    artifact_name: str,
    *,
    supported_version: int = ARTIFACT_SCHEMA_VERSION,
) -> Mapping[str, Any]:
    """Validate an artifact manifest against the supported schema.

    Parameters
    ----------
    payload : object
        Decoded JSON payload to validate.
    artifact_name : str
        Human-readable artifact name used in error messages.
    supported_version : int, optional
        Schema version supported for this artifact kind.

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
    if version != supported_version:
        direction = "older" if version < supported_version else "newer"
        raise ValueError(
            f"{artifact_name} schema_version {version} is {direction} than supported "
            f"version {supported_version}; regenerate or migrate with this project "
            "version"
        )

    return payload


def server_artifact_binding(app_manifest: AppManifest) -> dict[str, Any]:
    """Build the exact public-artifact and model-dimension server binding.

    Parameters
    ----------
    app_manifest : AppManifest
        Validated ``AppManifest`` snapshot used to construct the server model.

    Returns
    -------
    dict of str to Any
        Canonical binding recorded in the server artifact manifest.

    Raises
    ------
    ValueError
        If the supplied object is not a complete validated app-manifest snapshot.
    """
    try:
        payload = app_manifest.payload
        manifest_bytes = app_manifest.manifest_bytes
        vocabulary_bytes = app_manifest.vocabulary_bytes
        dimensions = {
            name: payload[name]
            for name in ("vocabulary_size", "sequence_length", "embedding_dim")
        }
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(
            "server artifacts require a validated public manifest; regenerate training "
            "artifacts with the current project version"
        ) from error
    if not isinstance(manifest_bytes, bytes) or not isinstance(vocabulary_bytes, bytes):
        raise ValueError("server artifacts require immutable public artifact bytes")
    if any(type(value) is not int or value <= 0 for value in dimensions.values()):
        raise ValueError("server model dimensions must be positive built-in integers")
    return {
        "public_manifest_checksum": sha256_bytes(manifest_bytes),
        "vocabulary_checksum": sha256_bytes(vocabulary_bytes),
        "model_dimensions": dimensions,
    }


def _validate_server_binding(
    value: object, *, app_manifest: AppManifest | None = None
) -> Mapping[str, Any]:
    """Validate a server binding and optionally match its public snapshot.

    Parameters
    ----------
    value : object
        Candidate decoded server binding.
    app_manifest : AppManifest or None, optional
        Public snapshot required by the consumer.

    Returns
    -------
    mapping of str to Any
        Validated binding.

    Raises
    ------
    ValueError
        If the binding is absent, malformed, or incompatible.
    """
    guidance = (
        "server artifact is not bound to a compatible public manifest and vocabulary; "
        "regenerate the server run with the current project version"
    )
    if not isinstance(value, Mapping) or set(value) != _SERVER_BINDING_FIELDS:
        raise ValueError(guidance)
    dimensions = value.get("model_dimensions")
    if (
        not isinstance(dimensions, Mapping)
        or set(dimensions) != _MODEL_DIMENSION_FIELDS
        or any(type(item) is not int or item <= 0 for item in dimensions.values())
    ):
        raise ValueError(guidance)
    for field in ("public_manifest_checksum", "vocabulary_checksum"):
        checksum = value.get(field)
        if (
            not isinstance(checksum, str)
            or len(checksum) != 71
            or not checksum.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in checksum[7:])
        ):
            raise ValueError(guidance)
    if app_manifest is not None and dict(value) != server_artifact_binding(
        app_manifest
    ):
        raise ValueError(
            "server artifact public-manifest, vocabulary, or model-dimension binding "
            "does not match the configured public artifacts; regenerate the server run"
        )
    return value


def write_server_artifact_manifest(
    artifact_dir: Path,
    *,
    app_manifest: AppManifest | None = None,
    finalized: bool = False,
) -> Path:
    """Write the compatibility contract for one server artifact directory.

    Parameters
    ----------
    artifact_dir : pathlib.Path
        Server output directory.
    app_manifest : AppManifest or None, optional
        Validated public snapshot used to construct the model. Finalization may omit
        it only when the existing in-progress manifest already contains the binding.
    finalized : bool, optional
        Require completed outputs and record their checksums.

    Returns
    -------
    pathlib.Path
        Path to the written artifact manifest.

    Raises
    ------
    ValueError
        If no validated public binding is available or final artifacts are invalid.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = artifact_dir.resolve(strict=True)
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError("server artifact directory must be a regular directory")
    path = artifact_dir / SERVER_ARTIFACT_MANIFEST_FILENAME
    existing: Mapping[str, Any] | None = None
    if path.exists():
        try:
            decoded = json.loads(read_regular_file(path, parent=canonical_dir))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("existing server artifact manifest is invalid") from error
        existing = validate_artifact_schema(
            decoded,
            "server artifact manifest",
            supported_version=SERVER_ARTIFACT_SCHEMA_VERSION,
        )
    binding = (
        server_artifact_binding(app_manifest)
        if app_manifest is not None
        else dict(_validate_server_binding(existing.get("binding")))
        if existing is not None
        else None
    )
    if binding is None:
        raise ValueError(
            "server artifacts require a validated public manifest; regenerate training "
            "artifacts with the current project version"
        )
    payload: dict[str, Any] = {
        "schema_version": SERVER_ARTIFACT_SCHEMA_VERSION,
        "artifacts": SERVER_ARTIFACTS,
        "binding": binding,
    }
    if finalized:
        if existing is not None:
            if existing.get("lifecycle") == "complete":
                raise ValueError(
                    "completed artifact manifest cannot be finalized again"
                )
        missing = [
            name
            for name in REQUIRED_COMPLETED_ARTIFACTS
            if not (artifact_dir / name).exists()
        ]
        if missing:
            raise ValueError(
                "cannot finalize run with missing artifacts: "
                + ", ".join(sorted(missing))
            )
        artifact_bytes: dict[str, bytes] = {}
        for artifact_path in sorted(artifact_dir.iterdir()):
            if artifact_path.name == SERVER_ARTIFACT_MANIFEST_FILENAME:
                continue
            artifact_bytes[artifact_path.name] = read_regular_file(
                artifact_path, parent=canonical_dir
            )
        payload["lifecycle"] = "complete"
        payload["checksums"] = {
            name: sha256_bytes(content) for name, content in artifact_bytes.items()
        }
    return write_json_atomically(path, payload)


def load_server_artifact_snapshot(
    artifact_dir: Path,
    *,
    manifest_bytes: bytes | None = None,
    app_manifest: AppManifest | None = None,
) -> ServerArtifactSnapshot:
    """Load validated artifact bytes without reopening them after verification.

    Parameters
    ----------
    artifact_dir : pathlib.Path
        Directory containing ``artifact_manifest.json``.
    manifest_bytes : bytes or None, optional
        Exact manifest bytes already bound by a current-run pointer.
    app_manifest : AppManifest or None, optional
        Configured public snapshot that the server artifact must exactly match.

    Returns
    -------
    ServerArtifactSnapshot
        Validated manifest and artifact byte snapshots.

    Raises
    ------
    ValueError
        If any artifact is invalid, mutable through a symlink, or outside the run.
    """
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError("server artifact directory must be a regular directory")
    canonical_dir = artifact_dir.resolve(strict=True)
    path = artifact_dir / SERVER_ARTIFACT_MANIFEST_FILENAME
    if manifest_bytes is None:
        try:
            manifest_bytes = read_regular_file(path, parent=canonical_dir)
        except ValueError as error:
            raise ValueError(
                "server artifact manifest has no valid schema_version; regenerate its "
                "artifacts"
            ) from error
    try:
        payload = validate_artifact_schema(
            json.loads(manifest_bytes.decode("utf-8")),
            "server artifact manifest",
            supported_version=SERVER_ARTIFACT_SCHEMA_VERSION,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid server artifact manifest: {path}") from error
    artifacts = payload.get("artifacts")
    mismatch_message = (
        "server artifact manifest does not match schema_version "
        f"{SERVER_ARTIFACT_SCHEMA_VERSION}; regenerate its artifacts"
    )
    if not isinstance(artifacts, Mapping):
        raise ValueError(mismatch_message)
    for name, layout in SERVER_ARTIFACTS.items():
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping) or any(
            artifact.get(key) != value for key, value in layout.items()
        ):
            raise ValueError(mismatch_message)
    _validate_server_binding(payload.get("binding"), app_manifest=app_manifest)

    lifecycle = payload.get("lifecycle")
    checksums = payload.get("checksums")
    if lifecycle is None and checksums is None:
        filenames = {
            str(layout["filename"])
            for layout in SERVER_ARTIFACTS.values()
            if (artifact_dir / str(layout["filename"])).exists()
        }
    else:
        if lifecycle != "complete" or not isinstance(checksums, Mapping):
            raise ValueError("server artifact manifest has invalid completion metadata")
        if not REQUIRED_COMPLETED_ARTIFACTS <= checksums.keys():
            raise ValueError("server artifact manifest is missing required checksums")
        filenames = set(checksums)

    files: dict[str, bytes] = {}
    for filename in filenames:
        expected = checksums.get(filename) if isinstance(checksums, Mapping) else None
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or (
                expected is not None
                and (
                    not isinstance(expected, str)
                    or len(expected) != 71
                    or not expected.startswith("sha256:")
                )
            )
        ):
            raise ValueError("server artifact manifest has an invalid checksum")
        try:
            content = read_regular_file(artifact_dir / filename, parent=canonical_dir)
        except ValueError as error:
            raise ValueError(
                f"server artifact is missing or unsafe: {filename}"
            ) from error
        if expected is not None and not hmac.compare_digest(
            sha256_bytes(content), expected
        ):
            raise ValueError(f"server artifact checksum does not match: {filename}")
        files[filename] = content
    return ServerArtifactSnapshot(
        directory=canonical_dir,
        manifest=deep_freeze(payload),
        files=MappingProxyType(files),
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
    return load_server_artifact_snapshot(artifact_dir).manifest
