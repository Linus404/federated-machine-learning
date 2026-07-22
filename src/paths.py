from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from src.artifact_compatibility import (
    canonical_json_bytes,
    read_regular_file,
    reject_duplicate_json_keys,
    sha256_bytes,
)

PUBLIC_ARTIFACT_DIR_ENV = "FML_PUBLIC_ARTIFACT_DIR"
SERVER_ARTIFACT_DIR_ENV = "FML_SERVER_ARTIFACT_DIR"
EVALUATION_ARTIFACT_DIR_ENV = "FML_EVALUATION_ARTIFACT_DIR"
DEFAULT_PUBLIC_ARTIFACT_DIR = Path("artifacts/public")
DEFAULT_SERVER_ARTIFACT_DIR = Path("artifacts/server")
DEFAULT_EVALUATION_ARTIFACT_DIR = Path("artifacts/evaluation")
PREPARED_CURRENT_FILENAME = ".prepared-current"
PREPARED_GENERATIONS_DIRECTORY = ".prepared-generations"
PREPARED_LEGACY_DIRECTORY = ".prepared-legacy"
PREPARED_MIGRATION_FILENAME = ".prepared-migration.json"
PREPARATION_LOCK_FILENAME = ".fml-prepare.lock"
PREPARED_CONTROL_BASENAME_PREFIXES = (
    ".prepared-",
    ".prepare-",
    f".{PREPARED_CURRENT_FILENAME}.",
)
PREPARED_GENERATION_SCHEMA_VERSION = 3
PREPARED_ARTIFACT_KINDS = frozenset({"client", "public", "evaluation"})


def validate_prepared_generation_inventory(value: object) -> dict[str, dict[str, Any]]:
    """Validate and canonicalize one prepared-generation file inventory.

    Parameters
    ----------
    value : object
        Candidate mapping from generation-relative filenames to byte bindings.

    Returns
    -------
    dict of str to dict of str to Any
        Canonically ordered exact file inventory.

    Raises
    ------
    ValueError
        If a path, size, checksum, or field set is invalid.
    """
    if not isinstance(value, Mapping) or not value:
        raise ValueError("prepared generation inventory must be a non-empty object")
    inventory: dict[str, dict[str, Any]] = {}
    for filename in sorted(value):
        binding = value[filename]
        path = Path(filename) if isinstance(filename, str) else Path()
        if (
            not isinstance(filename, str)
            or path.is_absolute()
            or len(path.parts) < 2
            or path.parts[0] not in PREPARED_ARTIFACT_KINDS
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != filename
            or not isinstance(binding, Mapping)
            or set(binding) != {"size_bytes", "checksum"}
        ):
            raise ValueError("prepared generation inventory entry is invalid")
        size = binding["size_bytes"]
        checksum = binding["checksum"]
        if (
            type(size) is not int
            or size < 0
            or not isinstance(checksum, str)
            or len(checksum) != 71
            or not checksum.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in checksum[7:])
        ):
            raise ValueError("prepared generation inventory binding is invalid")
        inventory[filename] = {"size_bytes": size, "checksum": checksum}
    return inventory


def prepared_generation_inventory(
    generation: Path, *, artifact_kind: str | None = None
) -> dict[str, dict[str, Any]]:
    """Snapshot exact regular files beneath one prepared generation.

    Parameters
    ----------
    generation : pathlib.Path
        Canonical generation directory containing artifact-kind subdirectories.
    artifact_kind : str or None, optional
        Restrict the snapshot to one selected artifact kind.

    Returns
    -------
    dict of str to dict of str to Any
        Sorted relative filenames bound to exact sizes and SHA-256 checksums.

    Raises
    ------
    ValueError
        If the generation contains unsafe directories or files.
    """
    if artifact_kind is not None and artifact_kind not in PREPARED_ARTIFACT_KINDS:
        raise ValueError(f"unsupported prepared artifact kind: {artifact_kind}")
    if generation.is_symlink() or not generation.is_dir():
        raise ValueError("prepared generation is missing or unsafe")
    canonical_generation = generation.resolve(strict=True)
    roots = (
        [generation / artifact_kind]
        if artifact_kind is not None
        else [generation / name for name in sorted(PREPARED_ARTIFACT_KINDS)]
    )
    inventory: dict[str, dict[str, Any]] = {}

    def visit(directory: Path) -> None:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("prepared generation directory is unsafe")
        canonical_directory = directory.resolve(strict=True)
        if canonical_generation not in canonical_directory.parents:
            raise ValueError("prepared generation directory escapes its root")
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            child_stat = child.lstat()
            if stat.S_ISDIR(child_stat.st_mode):
                visit(child)
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                raise ValueError("prepared generation file is unsafe")
            content = read_regular_file(child, parent=canonical_directory)
            relative = child.relative_to(generation).as_posix()
            inventory[relative] = {
                "size_bytes": len(content),
                "checksum": sha256_bytes(content),
            }

    for root in roots:
        visit(root)
    return inventory


def validate_preparation_request(value: object) -> dict[str, int]:
    """Validate artifact-affecting fields for one preparation request.

    Parameters
    ----------
    value : object
        Candidate request persisted in a journal or generation index.

    Returns
    -------
    dict of str to int
        Canonical preparation request.

    Raises
    ------
    ValueError
        If fields, types, or values differ from the supported request contract.
    """
    if not isinstance(value, Mapping) or set(value) != {"partitions"}:
        raise ValueError("prepared generation request has an invalid field set")
    partitions = value["partitions"]
    if type(partitions) is not int or partitions < 1:
        raise ValueError("prepared generation request has invalid partitions")
    return {"partitions": partitions}


class RunArtifactLock:
    """Hold an operating-system lock for one server artifact directory."""

    def __init__(self, path: Path, file: BinaryIO) -> None:
        self.path = path
        self._file = file
        self._released = False

    def release(self) -> None:
        """Release the artifact lock.

        Returns
        -------
        None
        """
        if self._released:
            return
        if os.name == "nt":
            import msvcrt

            self._file.seek(0)
            locking = msvcrt.locking  # type: ignore[attr-defined]
            unlock_mode = msvcrt.LK_UNLCK  # type: ignore[attr-defined]
            locking(self._file.fileno(), unlock_mode, 1)
        else:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._released = True


def resolve_dir(value: str | Path) -> Path:
    """Return an absolute path, treating relative paths as launch-dir relative."""
    path = Path(value).expanduser()
    if path.is_absolute():
        if path.is_symlink():
            return path.parent.resolve() / path.name
        return path.resolve()
    return Path.cwd() / path


def resolve_prepared_artifact_dir(value: str | Path, artifact_kind: str) -> Path:
    """Resolve a logical prepared-data root through one atomic generation index.

    Parameters
    ----------
    value : str or pathlib.Path
        Configured logical artifact root.
    artifact_kind : str
        One of ``client``, ``public``, or ``evaluation``.

    Returns
    -------
    pathlib.Path
        Immutable selected generation directory, or the legacy logical root when no
        generation index exists.

    Raises
    ------
    ValueError
        If the index, configured logical name, or selected generation is unsafe.
    """
    if artifact_kind not in PREPARED_ARTIFACT_KINDS:
        raise ValueError(f"unsupported prepared artifact kind: {artifact_kind}")
    logical_root = resolve_dir(value)
    parent = logical_root.parent
    pointer = parent / PREPARED_CURRENT_FILENAME
    if not pointer.exists() and not pointer.is_symlink():
        return logical_root
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("prepared artifact parent must be a regular directory")
    if not pointer.is_symlink():
        raise ValueError("prepared generation pointer must be an atomic directory link")
    expected_alias = Path(PREPARED_CURRENT_FILENAME) / artifact_kind
    if (
        not logical_root.is_symlink()
        or Path(os.readlink(logical_root)) != expected_alias
    ):
        raise ValueError("prepared logical root does not select the current generation")
    target_text = os.readlink(pointer)
    target = Path(target_text)
    if target.is_absolute() or len(target.parts) != 2:
        raise ValueError("prepared generation pointer target is invalid")
    generations_name, pointer_generation_id = target.parts
    if generations_name != PREPARED_GENERATIONS_DIRECTORY:
        raise ValueError("prepared generation pointer target is invalid")
    try:
        parsed_pointer_id = uuid.UUID(pointer_generation_id)
    except ValueError as error:
        raise ValueError("prepared generation pointer identity is invalid") from error
    if (
        parsed_pointer_id.version != 4
        or str(parsed_pointer_id) != pointer_generation_id
        or target_text != f"{PREPARED_GENERATIONS_DIRECTORY}/{pointer_generation_id}"
    ):
        raise ValueError("prepared generation pointer identity is invalid")
    generations = parent / PREPARED_GENERATIONS_DIRECTORY
    generation = generations / pointer_generation_id
    index_path = generation / "index.json"
    if generations.is_symlink() or generation.is_symlink() or not generation.is_dir():
        raise ValueError("selected prepared generation is missing or unsafe")
    canonical_generations = generations.resolve(strict=True)
    canonical_generation = generation.resolve(strict=True)
    if canonical_generation.parent != canonical_generations:
        raise ValueError("selected prepared generation escapes its artifact root")

    try:
        index_bytes = read_regular_file(index_path, parent=canonical_generation)
        payload = json.loads(
            index_bytes.decode("utf-8"), object_pairs_hook=reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("prepared generation index is invalid or unsafe") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "generation_id",
        "inventory",
        "logical_roots",
        "preparation_request",
    }:
        raise ValueError("prepared generation index has an invalid field set")
    if type(payload["schema_version"]) is not int:
        raise ValueError("prepared generation index has an invalid schema_version")
    if payload["schema_version"] != PREPARED_GENERATION_SCHEMA_VERSION:
        raise ValueError("prepared generation index has an unsupported schema_version")
    logical_roots = payload["logical_roots"]
    if (
        not isinstance(logical_roots, dict)
        or set(logical_roots) != PREPARED_ARTIFACT_KINDS
        or any(
            not isinstance(name, str)
            or Path(name).name != name
            or name in {"", ".", ".."}
            for name in logical_roots.values()
        )
        or logical_roots[artifact_kind] != logical_root.name
    ):
        raise ValueError("prepared generation index does not match configured roots")
    preparation_request = validate_preparation_request(payload["preparation_request"])
    inventory = validate_prepared_generation_inventory(payload["inventory"])
    canonical_payload = {
        "schema_version": PREPARED_GENERATION_SCHEMA_VERSION,
        "generation_id": payload["generation_id"],
        "inventory": inventory,
        "logical_roots": {
            name: logical_roots[name] for name in sorted(PREPARED_ARTIFACT_KINDS)
        },
        "preparation_request": preparation_request,
    }
    if index_bytes != canonical_json_bytes(canonical_payload):
        raise ValueError("prepared generation index bytes are not canonical")
    index_generation_id = payload["generation_id"]
    try:
        parsed_id = (
            uuid.UUID(index_generation_id)
            if isinstance(index_generation_id, str)
            else None
        )
    except ValueError as error:
        raise ValueError(
            "prepared generation index has an invalid generation_id"
        ) from error
    if (
        parsed_id is None
        or parsed_id.version != 4
        or str(parsed_id) != index_generation_id
    ):
        raise ValueError("prepared generation index has an invalid generation_id")
    if index_generation_id != pointer_generation_id:
        raise ValueError("prepared generation pointer and index identities differ")

    selected_inventory = {
        filename: binding
        for filename, binding in inventory.items()
        if Path(filename).parts[0] == artifact_kind
    }
    if (
        prepared_generation_inventory(canonical_generation, artifact_kind=artifact_kind)
        != selected_inventory
    ):
        raise ValueError(
            "selected prepared generation differs from its bound inventory"
        )

    selected = generation / artifact_kind
    if selected.is_symlink() or not selected.is_dir():
        raise ValueError("selected prepared generation is missing or unsafe")
    canonical_selected = selected.resolve(strict=True)
    if canonical_selected.parent != canonical_generation:
        raise ValueError("selected prepared generation escapes its artifact root")
    return canonical_selected


def default_public_artifact_dir() -> Path:
    """Return the configured public artifact directory."""
    return resolve_dir(
        os.environ.get(PUBLIC_ARTIFACT_DIR_ENV, DEFAULT_PUBLIC_ARTIFACT_DIR)
    )


def default_server_artifact_dir() -> Path:
    """Return the configured server artifact directory."""
    return resolve_dir(
        os.environ.get(SERVER_ARTIFACT_DIR_ENV, DEFAULT_SERVER_ARTIFACT_DIR)
    )


def default_evaluation_artifact_dir() -> Path:
    """Return the configured evaluation-only artifact directory.

    Returns
    -------
    pathlib.Path
        Evaluation artifact directory from the environment or project default.
    """
    return resolve_dir(
        os.environ.get(EVALUATION_ARTIFACT_DIR_ENV, DEFAULT_EVALUATION_ARTIFACT_DIR)
    )


def acquire_run_artifact_lock(artifact_dir: str | Path) -> RunArtifactLock:
    """Acquire exclusive ownership of one server artifact directory.

    Parameters
    ----------
    artifact_dir : str or pathlib.Path
        Directory whose run artifacts require a single writer.

    Returns
    -------
    RunArtifactLock
        Lock held until ``release`` is called or the owning process exits.

    Raises
    ------
    RuntimeError
        If another process already owns the artifact directory.
    """
    artifact_path = resolve_dir(artifact_dir)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_path.parent / f".{artifact_path.name}.run.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise ValueError(
            "artifact lock path must be a single-link regular file"
        ) from error
    lock_stat = os.fstat(descriptor)
    if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
        os.close(descriptor)
        raise ValueError("artifact lock path must be a single-link regular file")
    lock_file = os.fdopen(descriptor, "a+b")

    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            locking = msvcrt.locking  # type: ignore[attr-defined]
            nonblocking_mode = msvcrt.LK_NBLCK  # type: ignore[attr-defined]
            locking(lock_file.fileno(), nonblocking_mode, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        lock_file.close()
        raise RuntimeError(
            f"Another server run is already writing to {artifact_path}"
        ) from error

    return RunArtifactLock(lock_path, lock_file)


def global_model_path(artifact_dir: str | Path) -> Path:
    return resolve_dir(artifact_dir) / "global_model.keras"


def metrics_path(artifact_dir: str | Path) -> Path:
    return resolve_dir(artifact_dir) / "metrics.csv"


def client_metrics_path(artifact_dir: str | Path) -> Path:
    return resolve_dir(artifact_dir) / "client_metrics.csv"


def run_manifest_path(artifact_dir: str | Path) -> Path:
    """Return the immutable run provenance path for an artifact directory.

    Parameters
    ----------
    artifact_dir : str or pathlib.Path
        Server artifact directory.

    Returns
    -------
    pathlib.Path
        Run manifest path.
    """
    return resolve_dir(artifact_dir) / "run_manifest.json"
