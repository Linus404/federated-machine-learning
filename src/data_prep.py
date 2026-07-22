from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import shutil
import stat
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.protocol_runtime import validate_protocol_runtime

import numpy as np

from src.app_manifest import (
    PUBLIC_ARTIFACT_FILENAMES,
    load_app_manifest,
    protocol_model_dimensions,
)
from src.artifact_compatibility import (
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    RetainedDirectoryChain,
    canonical_json_bytes,
    link_unnamed_file_at,
    read_regular_file,
    require_secure_artifact_platform,
    sha256_bytes,
    sha256_file,
    strict_json_loads,
    write_json_atomically,
)
from src.contracts import (
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_SPLIT_SEED,
    canonical_client_row_bytes,
    client_shard_metadata,
    dirichlet_split,
)
from src.evaluation_artifact import (
    canonical_source_row_bytes,
    load_scientific_protocol,
    load_evaluation_artifact_snapshot,
    publish_evaluation_artifact,
)
from src.paths import (
    PREPARED_CURRENT_FILENAME,
    PREPARED_CONTROL_BASENAME_PREFIXES,
    PREPARED_GENERATIONS_DIRECTORY,
    PREPARED_GENERATION_SCHEMA_VERSION,
    PREPARED_LEGACY_DIRECTORY,
    PREPARED_MIGRATION_FILENAME,
    PREPARATION_LOCK_FILENAME,
    RunArtifactLock,
    default_evaluation_artifact_dir,
    default_public_artifact_dir,
    prepared_generation_inventory,
    validate_preparation_request,
    validate_prepared_generation_inventory,
)
from src.text_preprocessing import create_text_vectorizer

_PREPARATION_STAGE_STATE_FILENAME = ".prepare-stage.state"
_PREPARATION_STAGE_STATE_SCHEMA_VERSION = 1
_PREPARATION_STAGE_STATE_FIELDS = {
    "schema_version",
    "operation",
    "parent_device",
    "parent_inode",
    "nonce",
    "stage_name",
    "device",
    "inode",
    "tombstone_name",
    "generation",
}
_LEGACY_PREPARATION_STAGE_STATE_FIELDS = _PREPARATION_STAGE_STATE_FIELDS - {
    "generation"
}
_STAGE_STATE_GENERATION_WIDTH = 20


@dataclass(frozen=True)
class _PreparationStageState:
    """Bind one private preparation stage to durable descriptor identity."""

    operation: str
    parent_device: int
    parent_inode: int
    nonce: str
    stage_name: str
    device: int | None = None
    inode: int | None = None
    tombstone_name: str | None = None

    def payload(self) -> dict[str, object]:
        """Return the canonical serialized state.

        Returns
        -------
        dict of str to object
            Exact durable ownership record.
        """
        return {
            "schema_version": _PREPARATION_STAGE_STATE_SCHEMA_VERSION,
            "operation": self.operation,
            "parent_device": self.parent_device,
            "parent_inode": self.parent_inode,
            "nonce": self.nonce,
            "stage_name": self.stage_name,
            "device": self.device,
            "inode": self.inode,
            "tombstone_name": self.tombstone_name,
        }


@dataclass(frozen=True)
class _PreparationStateRecord:
    """Retain one committed generation of preparation ownership state.

    Parameters
    ----------
    state : _PreparationStageState
        Parsed ownership transition.
    document : bytes
        Exact canonical state bytes.
    identity : tuple of int
        Device and inode of the committed record.
    name : str
        Descriptor-relative generation filename.
    generation : int
        Monotonic generation selected from the filename and document.
    """

    state: _PreparationStageState
    document: bytes
    identity: tuple[int, int]
    name: str
    generation: int


def _validate_absolute_output_ancestors(
    output_dir: str | Path, artifact_name: str
) -> Path:
    """Anchor an output path and validate its ancestors without following links.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Configured artifact root.
    artifact_name : str
        Human-readable artifact kind used in errors.

    Returns
    -------
    pathlib.Path
        Lexically normalized absolute path anchored to the launch directory.

    Raises
    ------
    ValueError
        If an existing ancestor is a symlink or not a directory.
    """
    candidate = Path(output_dir).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    output_path = Path(os.path.abspath(candidate))
    anchor = Path(output_path.anchor)
    current = anchor
    for component in output_path.parent.parts[len(anchor.parts) :]:
        current /= component
        try:
            component_stat = current.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise ValueError(
                f"{artifact_name} artifact path component is unsafe"
            ) from error
        if not stat.S_ISDIR(component_stat.st_mode):
            raise ValueError(
                f"{artifact_name} artifact path component must be a regular directory"
            )
    return output_path


def _preflight_output_root(
    output_dir: str | Path,
    artifact_name: str,
    *,
    reusable: bool,
    allow_prepared_alias: bool = False,
) -> Path:
    """Validate an artifact root without creating or following it.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Configured artifact root.
    artifact_name : str
        Human-readable artifact kind used in errors.
    reusable : bool
        Permit an existing real directory for retryable outputs.
    allow_prepared_alias : bool, optional
        Permit the controlled logical link used by atomic prepared generations.

    Returns
    -------
    pathlib.Path
        Absolute validated output root.

    Raises
    ------
    FileExistsError
        If an immutable output root already exists.
    ValueError
        If the root or its existing parent is a symlink or invalid path type.
    """
    basename = Path(output_dir).expanduser().name
    if basename == PREPARATION_LOCK_FILENAME or basename.startswith(
        PREPARED_CONTROL_BASENAME_PREFIXES
    ):
        raise ValueError(
            f"{artifact_name} artifact root uses a reserved preparation name"
        )
    output_path = _validate_absolute_output_ancestors(output_dir, artifact_name)
    if output_path.is_symlink():
        expected = Path(PREPARED_CURRENT_FILENAME) / artifact_name
        if not (
            allow_prepared_alias
            and reusable
            and Path(os.readlink(output_path)) == expected
        ):
            raise ValueError(f"{artifact_name} artifact root must not be a symlink")
        return output_path
    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError(
                f"{artifact_name} artifact root must be a regular directory"
            )
        if not reusable:
            raise FileExistsError(
                f"{artifact_name} artifact path already exists; use a new path"
            )
    parent = output_path.parent
    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
        raise ValueError(f"{artifact_name} artifact parent must be a regular directory")
    return output_path


def _validate_output_child(path: Path, parent: Path, *, directory: bool) -> None:
    """Reject unsafe existing publication children before replacement.

    Parameters
    ----------
    path : pathlib.Path
        Direct child output path.
    parent : pathlib.Path
        Validated canonical parent directory.
    directory : bool
        Require a real directory instead of a single-link regular file.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the child escapes, is a link, or has an unsafe file type.
    """
    if path.parent.resolve(strict=True) != parent.resolve(strict=True):
        raise ValueError(f"artifact output escapes its root: {path.name}")
    if not path.exists() and not path.is_symlink():
        return
    try:
        child_stat = path.lstat()
    except OSError as error:
        raise ValueError(f"artifact output is unsafe: {path.name}") from error
    valid = (
        stat.S_ISDIR(child_stat.st_mode)
        if directory
        else stat.S_ISREG(child_stat.st_mode) and child_stat.st_nlink == 1
    )
    if path.is_symlink() or not valid:
        raise ValueError(f"artifact output is unsafe: {path.name}")


def _write_bytes_atomically(path: Path, content: bytes) -> None:
    """Write bytes atomically without following or sharing an existing child.

    Parameters
    ----------
    path : pathlib.Path
        Destination beneath an existing validated directory.
    content : bytes
        Exact bytes to publish.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an existing destination is not a single-link regular file.
    """
    _validate_output_child(path, path.parent, directory=False)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_client_directory(path: Path, parent: Path) -> None:
    """Validate an existing generated client directory and all direct children.

    Parameters
    ----------
    path : pathlib.Path
        Existing ``client-N`` directory.
    parent : pathlib.Path
        Canonical client artifact root.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the directory or any child is linked or non-regular.
    """
    _validate_output_child(path, parent, directory=True)
    canonical_path = path.resolve(strict=True)
    for child in path.iterdir():
        _validate_output_child(child, canonical_path, directory=False)


def _acquire_preparation_lock(roots: Mapping[str, Path]) -> RunArtifactLock:
    """Acquire a nonblocking exclusive lock for one preparation destination set.

    Parameters
    ----------
    roots : mapping of str to pathlib.Path
        Validated final artifact roots.

    Returns
    -------
    RunArtifactLock
        Held lock that the caller must release.

    Raises
    ------
    RuntimeError
        If another process is preparing the same roots.
    ValueError
        If the retained lock-owner chain is unsafe or replaced.
    """
    lock_path = roots["client"].parent / PREPARATION_LOCK_FILENAME
    chain = RetainedDirectoryChain.open(
        lock_path.parent,
        error_message="preparation lock owner changed",
        check_platform=False,
    )
    descriptor = os.dup(chain.directory.descriptor)
    try:
        import fcntl

        chain.verify()
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        chain.verify()
    except OSError as error:
        os.close(descriptor)
        chain.close()
        raise RuntimeError(
            "Another artifact preparation is already in progress"
        ) from error
    except BaseException:
        os.close(descriptor)
        chain.close()
        raise
    return RunArtifactLock(
        lock_path,
        descriptor,
        verify=chain.verify,
        close_owner=chain.close,
    )


def _fsync_directory(path: Path) -> None:
    """Flush one directory entry set on platforms that support it.

    Parameters
    ----------
    path : pathlib.Path
        Existing directory to flush.

    Returns
    -------
    None
    """
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    """Flush a completed owned artifact tree before final renames.

    Parameters
    ----------
    root : pathlib.Path
        Validated staging root containing only regular files and directories.

    Returns
    -------
    None
    """
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in [*directories, root]:
        _fsync_directory(directory)


def _preparation_state_name(generation: int) -> str:
    """Return one sequence-numbered preparation-state name.

    Parameters
    ----------
    generation : int
        Nonnegative committed state generation.

    Returns
    -------
    str
        Descriptor-relative state filename.
    """
    return (
        f"{_PREPARATION_STAGE_STATE_FILENAME}."
        f"{generation:0{_STAGE_STATE_GENERATION_WIDTH}d}"
    )


def _read_preparation_record_at(
    parent_descriptor: int, name: str
) -> tuple[bytes, tuple[int, int]]:
    """Read one stable private preparation-stage state generation.

    Parameters
    ----------
    parent_descriptor : int
        Retained prepared-generations descriptor.
    name : str
        Direct committed state-generation name.

    Returns
    -------
    tuple of bytes and tuple of int
        Exact state bytes and identity.
    """
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=parent_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise ValueError("preparation stage state is unsafe")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or sum(map(len, chunks)) != after.st_size:
            raise ValueError("preparation stage state changed while retained")
        return b"".join(chunks), (after.st_dev, after.st_ino)
    finally:
        os.close(descriptor)


def _parse_preparation_stage_state(
    document: bytes,
) -> tuple[_PreparationStageState, int]:
    """Validate one canonical preparation-stage ownership record.

    Parameters
    ----------
    document : bytes
        Exact state bytes.

    Returns
    -------
    tuple of _PreparationStageState and int
        Strict validated state and committed generation.
    """
    payload = strict_json_loads(document, source="preparation stage state")
    fields = set(payload) if isinstance(payload, dict) else set()
    legacy = fields == _LEGACY_PREPARATION_STAGE_STATE_FIELDS
    if (
        not isinstance(payload, dict)
        or (fields != _PREPARATION_STAGE_STATE_FIELDS and not legacy)
        or payload["schema_version"] != _PREPARATION_STAGE_STATE_SCHEMA_VERSION
        or canonical_json_bytes(payload) != document
        or payload["operation"] not in {"reserved", "build", "delete"}
        or type(payload["parent_device"]) is not int
        or payload["parent_device"] < 0
        or type(payload["parent_inode"]) is not int
        or payload["parent_inode"] < 1
        or not isinstance(payload["nonce"], str)
        or len(payload["nonce"]) != 32
        or any(character not in "0123456789abcdef" for character in payload["nonce"])
        or payload["stage_name"] != f".prepare-{payload['nonce']}.staging"
        or (
            not legacy
            and (type(payload["generation"]) is not int or payload["generation"] < 0)
        )
    ):
        raise ValueError("preparation stage state is unsafe")
    operation = payload["operation"]
    device = payload["device"]
    inode = payload["inode"]
    tombstone = payload["tombstone_name"]
    if operation == "reserved":
        valid_phase = device is None and inode is None and tombstone is None
    else:
        valid_phase = (
            type(device) is int and device >= 0 and type(inode) is int and inode >= 1
        )
        if operation == "build":
            valid_phase = valid_phase and tombstone is None
        else:
            valid_phase = (
                valid_phase
                and tombstone
                == f".{payload['stage_name']}.{device:x}-{inode:x}.deleting"
            )
    if not valid_phase:
        raise ValueError("preparation stage state is unsafe")
    return (
        _PreparationStageState(
            operation,
            payload["parent_device"],
            payload["parent_inode"],
            payload["nonce"],
            payload["stage_name"],
            device,
            inode,
            tombstone,
        ),
        0 if legacy else payload["generation"],
    )


def _read_preparation_state_at(
    parent_descriptor: int,
) -> _PreparationStateRecord | None:
    """Select the newest strictly valid preparation-state generation.

    Parameters
    ----------
    parent_descriptor : int
        Retained prepared-generations descriptor.

    Returns
    -------
    _PreparationStateRecord or None
        Newest committed generation, or ``None`` when absent.
    """
    prefix = f"{_PREPARATION_STAGE_STATE_FILENAME}."
    records: list[_PreparationStateRecord] = []
    for name in os.listdir(parent_descriptor):
        if name == _PREPARATION_STAGE_STATE_FILENAME:
            document, identity = _read_preparation_record_at(parent_descriptor, name)
            state, generation = _parse_preparation_stage_state(document)
            if generation != 0:
                raise ValueError("preparation stage state is unsafe")
            records.append(
                _PreparationStateRecord(state, document, identity, name, generation)
            )
            continue
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if (
            len(suffix) != _STAGE_STATE_GENERATION_WIDTH
            or not suffix.isascii()
            or not suffix.isdigit()
        ):
            raise ValueError("preparation stage state is unsafe")
        document, identity = _read_preparation_record_at(parent_descriptor, name)
        state, generation = _parse_preparation_stage_state(document)
        if generation != int(suffix) or name != _preparation_state_name(generation):
            raise ValueError("preparation stage state is unsafe")
        records.append(
            _PreparationStateRecord(state, document, identity, name, generation)
        )
    if not records:
        return None
    records.sort(key=lambda candidate: candidate.generation)
    if len({candidate.generation for candidate in records}) != len(records):
        raise ValueError("preparation stage state is ambiguous")
    newest = records[-1]
    lineage = (
        newest.state.parent_device,
        newest.state.parent_inode,
        newest.state.nonce,
        newest.state.stage_name,
    )
    if any(
        (
            candidate.state.parent_device,
            candidate.state.parent_inode,
            candidate.state.nonce,
            candidate.state.stage_name,
        )
        != lineage
        for candidate in records[:-1]
    ):
        raise ValueError("preparation stage state is ambiguous")
    for candidate in records:
        if _read_preparation_record_at(parent_descriptor, candidate.name) != (
            candidate.document,
            candidate.identity,
        ):
            raise ValueError("preparation stage state changed while retained")
    return newest


def _write_preparation_stage_state(
    chain: RetainedDirectoryChain,
    state: _PreparationStageState,
    previous: _PreparationStateRecord | None,
) -> _PreparationStateRecord:
    """Install and durably flush one stage-ownership transition.

    Parameters
    ----------
    chain : RetainedDirectoryChain
        Retained chain through the prepared-generations directory.
    state : _PreparationStageState
        New exact ownership state.
    previous : _PreparationStateRecord or None
        Exact prior committed generation, or ``None`` for exclusive creation.

    Returns
    -------
    _PreparationStateRecord
        Newly committed state generation.
    """
    chain.verify()
    parent_descriptor = chain.directory.descriptor
    current = os.fstat(parent_descriptor)
    if (current.st_dev, current.st_ino) != (
        state.parent_device,
        state.parent_inode,
    ):
        raise ValueError("preparation stage state selects another parent")
    if _read_preparation_state_at(parent_descriptor) != previous:
        raise ValueError("preparation stage state changed while retained")
    generation = 0 if previous is None else previous.generation + 1
    if generation >= 10**_STAGE_STATE_GENERATION_WIDTH:
        raise ValueError("preparation stage state generation is exhausted")
    document = canonical_json_bytes({**state.payload(), "generation": generation})
    name = _preparation_state_name(generation)
    try:
        descriptor = os.open(
            ".",
            os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise RuntimeError("Linux O_TMPFILE support is required") from error
    try:
        written = 0
        while written < len(document):
            count = os.write(descriptor, document[written:])
            if count == 0:
                raise OSError("preparation stage-state write made no progress")
            written += count
        os.fsync(descriptor)
        chain.verify()
        link_unnamed_file_at(descriptor, parent_descriptor, name)
        os.fsync(parent_descriptor)
        chain.verify()
        installed_document, installed_identity = _read_preparation_record_at(
            parent_descriptor, name
        )
        installed = _PreparationStateRecord(
            state, document, installed_identity, name, generation
        )
        if (
            installed_document != document
            or _read_preparation_state_at(parent_descriptor) != installed
        ):
            raise ValueError("preparation stage state changed while retained")
        prefix = f"{_PREPARATION_STAGE_STATE_FILENAME}."
        retired = False
        for candidate_name in os.listdir(parent_descriptor):
            if candidate_name == name or (
                candidate_name != _PREPARATION_STAGE_STATE_FILENAME
                and not candidate_name.startswith(prefix)
            ):
                continue
            suffix = candidate_name[len(prefix) :]
            if candidate_name != _PREPARATION_STAGE_STATE_FILENAME and (
                len(suffix) != _STAGE_STATE_GENERATION_WIDTH or not suffix.isdigit()
            ):
                continue
            candidate_document, candidate_identity = _read_preparation_record_at(
                parent_descriptor, candidate_name
            )
            candidate_state, candidate_generation = _parse_preparation_stage_state(
                candidate_document
            )
            if (
                candidate_generation >= generation
                or candidate_state.nonce != state.nonce
                or candidate_state.stage_name != state.stage_name
                or candidate_identity
                != _read_preparation_record_at(parent_descriptor, candidate_name)[1]
            ):
                raise ValueError("preparation stage state is ambiguous")
            os.unlink(candidate_name, dir_fd=parent_descriptor)
            retired = True
        if retired:
            os.fsync(parent_descriptor)
            chain.verify()
        return installed
    finally:
        os.close(descriptor)


def _remove_preparation_stage_state(
    chain: RetainedDirectoryChain, record: _PreparationStateRecord
) -> None:
    """Durably remove only the exact stage-ownership record.

    Parameters
    ----------
    chain : RetainedDirectoryChain
        Retained prepared-generations chain.
    record : _PreparationStateRecord
        Exact newest state generation authorized for removal.

    Returns
    -------
    None
    """
    chain.verify()
    descriptor = chain.directory.descriptor
    if _read_preparation_state_at(descriptor) != record:
        raise ValueError("preparation stage state changed while retained")
    prefix = f"{_PREPARATION_STAGE_STATE_FILENAME}."
    names = sorted(
        name
        for name in os.listdir(descriptor)
        if name == _PREPARATION_STAGE_STATE_FILENAME
        or (
            name.startswith(prefix)
            and len(name[len(prefix) :]) == _STAGE_STATE_GENERATION_WIDTH
            and name[len(prefix) :].isdigit()
        )
    )
    for name in names:
        _read_preparation_record_at(descriptor, name)
        os.unlink(name, dir_fd=descriptor)
    os.fsync(descriptor)
    chain.verify()


def _open_bound_preparation_stage(
    chain: RetainedDirectoryChain,
    name: str,
    state: _PreparationStageState,
) -> int:
    """Open and validate one exact state-bound stage directory.

    Parameters
    ----------
    chain : RetainedDirectoryChain
        Retained prepared-generations chain.
    name : str
        State-selected source or tombstone name.
    state : _PreparationStageState
        State containing the exact expected identity.

    Returns
    -------
    int
        Open no-follow directory descriptor.
    """
    descriptor = chain.open_child(name, os.O_RDONLY | os.O_DIRECTORY)
    current = os.fstat(descriptor)
    if (current.st_dev, current.st_ino) != (state.device, state.inode):
        os.close(descriptor)
        raise ValueError("preparation stage residue was replaced")
    return descriptor


def _recover_preparation_stage(chain: RetainedDirectoryChain) -> None:
    """Recover only the exact invocation-owned preparation stage.

    Parameters
    ----------
    chain : RetainedDirectoryChain
        Retained prepared-generations chain.

    Returns
    -------
    None
    """
    parent_descriptor = chain.directory.descriptor
    record = _read_preparation_state_at(parent_descriptor)
    if record is None:
        return
    state = record.state
    current = os.fstat(parent_descriptor)
    if (current.st_dev, current.st_ino) != (
        state.parent_device,
        state.parent_inode,
    ):
        raise ValueError("preparation stage state selects another parent")

    names = (
        (state.stage_name, state.tombstone_name)
        if state.operation == "delete"
        else (state.stage_name,)
    )
    present: list[str] = []
    for name in names:
        if name is None:
            continue
        try:
            os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        present.append(name)
    if state.operation == "reserved":
        if present:
            raise ValueError("preparation stage residue lacks bound identity")
        _remove_preparation_stage_state(chain, record)
        return
    if len(present) > 1:
        raise ValueError("preparation stage state is ambiguous")
    if not present:
        _remove_preparation_stage_state(chain, record)
        return

    name = present[0]
    descriptor = _open_bound_preparation_stage(chain, name, state)
    try:
        if state.operation == "build":
            tombstone = f".{state.stage_name}.{state.device:x}-{state.inode:x}.deleting"
            deleting = _PreparationStageState(
                "delete",
                state.parent_device,
                state.parent_inode,
                state.nonce,
                state.stage_name,
                state.device,
                state.inode,
                tombstone,
            )
            record = _write_preparation_stage_state(chain, deleting, record)
            chain.remove_child_tree(name, descriptor, tombstone_name=tombstone)
        elif name == state.stage_name:
            assert state.tombstone_name is not None
            chain.remove_child_tree(
                name,
                descriptor,
                tombstone_name=state.tombstone_name,
            )
        else:
            chain.remove_detached_child_tree(name, descriptor)
    finally:
        os.close(descriptor)
    _remove_preparation_stage_state(chain, record)


def _publication_checkpoint(phase: str) -> None:
    """Expose a no-op process-death checkpoint for durability regression tests.

    Parameters
    ----------
    phase : str
        Stable publication phase identifier.

    Returns
    -------
    None
    """


def _migration_target(generation_id: str) -> str:
    """Return the canonical relative pointer target for one generation.

    Parameters
    ----------
    generation_id : str
        Canonical prepared generation UUID.

    Returns
    -------
    str
        Relative target beneath the prepared generations directory.
    """
    return f"{PREPARED_GENERATIONS_DIRECTORY}/{generation_id}"


def _validate_migration_journal(
    roots: Mapping[str, Path], journal_bytes: bytes
) -> dict[str, Any]:
    """Validate and normalize one durable prepared-root migration journal.

    Parameters
    ----------
    roots : mapping of str to pathlib.Path
        Configured logical artifact roots.
    journal_bytes : bytes
        Exact journal bytes read through the secure artifact boundary.

    Returns
    -------
    dict of str to Any
        Canonically ordered validated transaction fields.

    Raises
    ------
    ValueError
        If the journal is malformed, noncanonical, or belongs to other roots.
    """
    payload = strict_json_loads(journal_bytes, source="prepared migration journal")
    legacy_fields = {
        "schema_version",
        "generation_id",
        "stage_name",
        "logical_roots",
        "legacy_roots",
        "alias_roots",
        "previous_pointer_target",
    }
    if not isinstance(payload, dict):
        raise ValueError("prepared migration journal must be an object")
    schema_version = payload.get("schema_version")
    fields = legacy_fields
    if schema_version in {2, PREPARED_GENERATION_SCHEMA_VERSION}:
        fields |= {"preparation_request"}
    if schema_version == PREPARED_GENERATION_SCHEMA_VERSION:
        fields |= {
            "generation_index_checksum",
            "parent_device",
            "parent_inode",
            "generations_device",
            "generations_inode",
            "stage_device",
            "stage_inode",
        }
    if set(payload) != fields:
        raise ValueError("prepared migration journal has an invalid field set")
    generation_id = payload["generation_id"]
    try:
        parsed_id = uuid.UUID(generation_id) if isinstance(generation_id, str) else None
    except ValueError as error:
        raise ValueError("prepared migration journal generation is invalid") from error
    if (
        type(schema_version) is not int
        or schema_version not in {1, 2, PREPARED_GENERATION_SCHEMA_VERSION}
        or parsed_id is None
        or parsed_id.version != 4
        or str(parsed_id) != generation_id
    ):
        raise ValueError("prepared migration journal identity is invalid")
    stage_name = payload["stage_name"]
    if (
        not isinstance(stage_name, str)
        or Path(stage_name).name != stage_name
        or not stage_name.startswith(".prepare-")
        or not stage_name.endswith(".staging")
    ):
        raise ValueError("prepared migration journal stage is invalid")
    expected_roots = {name: roots[name].name for name in sorted(roots)}
    logical_roots = payload["logical_roots"]
    if not isinstance(logical_roots, dict) or logical_roots != expected_roots:
        raise ValueError("prepared migration journal does not match configured roots")
    request = (
        validate_preparation_request(payload["preparation_request"])
        if schema_version in {2, PREPARED_GENERATION_SCHEMA_VERSION}
        else None
    )
    index_checksum = payload.get("generation_index_checksum")
    if schema_version == PREPARED_GENERATION_SCHEMA_VERSION and (
        not isinstance(index_checksum, str)
        or len(index_checksum) != 71
        or not index_checksum.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in index_checksum[7:])
    ):
        raise ValueError("prepared migration journal index checksum is invalid")
    identity_fields = (
        "parent_device",
        "parent_inode",
        "generations_device",
        "generations_inode",
        "stage_device",
        "stage_inode",
    )
    if schema_version == PREPARED_GENERATION_SCHEMA_VERSION and any(
        type(payload[field]) is not int
        or payload[field] < (1 if field.endswith("inode") else 0)
        for field in identity_fields
    ):
        raise ValueError("prepared migration journal identity is invalid")
    legacy_roots = payload["legacy_roots"]
    alias_roots = payload["alias_roots"]
    for value, name in (
        (legacy_roots, "legacy roots"),
        (alias_roots, "alias roots"),
    ):
        if (
            not isinstance(value, list)
            or value != sorted(value)
            or len(value) != len(set(value))
            or any(item not in roots for item in value)
        ):
            raise ValueError(f"prepared migration journal {name} are invalid")
    if not set(legacy_roots) <= set(alias_roots):
        raise ValueError("prepared migration journal root state is inconsistent")
    previous_target = payload["previous_pointer_target"]
    if previous_target is not None:
        if not isinstance(previous_target, str):
            raise ValueError("prepared migration journal previous pointer is invalid")
        parts = Path(previous_target).parts
        if (
            len(parts) != 2
            or parts[0] != PREPARED_GENERATIONS_DIRECTORY
            or previous_target != "/".join(parts)
        ):
            raise ValueError("prepared migration journal previous pointer is invalid")
        try:
            previous_id = uuid.UUID(parts[1])
        except ValueError as error:
            raise ValueError(
                "prepared migration journal previous pointer is invalid"
            ) from error
        if previous_id.version != 4 or str(previous_id) != parts[1]:
            raise ValueError("prepared migration journal previous pointer is invalid")
    canonical = {
        "schema_version": schema_version,
        "generation_id": generation_id,
        "stage_name": stage_name,
        **(
            {"generation_index_checksum": index_checksum}
            if index_checksum is not None
            else {}
        ),
        **(
            {field: payload[field] for field in identity_fields}
            if schema_version == PREPARED_GENERATION_SCHEMA_VERSION
            else {}
        ),
        "logical_roots": expected_roots,
        **({"preparation_request": request} if request is not None else {}),
        "legacy_roots": legacy_roots,
        "alias_roots": alias_roots,
        "previous_pointer_target": previous_target,
    }
    if journal_bytes != canonical_json_bytes(canonical):
        raise ValueError("prepared migration journal bytes are not canonical")
    return canonical


def _load_migration_journal(roots: Mapping[str, Path]) -> dict[str, Any] | None:
    """Load a pending migration journal without following unsafe paths.

    Parameters
    ----------
    roots : mapping of str to pathlib.Path
        Configured logical artifact roots sharing one parent.

    Returns
    -------
    dict of str to Any or None
        Validated transaction, or ``None`` when no recovery is pending.
    """
    parent = roots["client"].parent
    path = parent / PREPARED_MIGRATION_FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    journal_bytes = read_regular_file(path, parent=parent.resolve(strict=True))
    return _validate_migration_journal(roots, journal_bytes)


class _PreparedGenerationValidationError(ValueError):
    """Identify recovery validation failures that must retain their journal."""


def _require_migration_generation_identity(
    generation: Path, transaction: Mapping[str, Any]
) -> None:
    """Require a journal candidate and both parents to retain exact identities.

    Parameters
    ----------
    generation : pathlib.Path
        Staged or final generation candidate.
    transaction : mapping of str to Any
        Validated migration journal.

    Returns
    -------
    None
    """
    if "stage_device" not in transaction:
        return
    generations = generation.parent
    parent = generations.parent
    parent_stat = parent.stat(follow_symlinks=False)
    generations_stat = generations.stat(follow_symlinks=False)
    generation_stat = generation.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or not stat.S_ISDIR(generations_stat.st_mode)
        or not stat.S_ISDIR(generation_stat.st_mode)
        or (parent_stat.st_dev, parent_stat.st_ino)
        != (transaction["parent_device"], transaction["parent_inode"])
        or (generations_stat.st_dev, generations_stat.st_ino)
        != (
            transaction["generations_device"],
            transaction["generations_inode"],
        )
        or (generation_stat.st_dev, generation_stat.st_ino)
        != (transaction["stage_device"], transaction["stage_inode"])
    ):
        raise ValueError("prepared migration generation identity changed")


def _validate_owned_generation_index(
    generation: Path, transaction: Mapping[str, Any]
) -> None:
    """Validate ownership metadata before discarding a mismatched generation.

    Parameters
    ----------
    generation : pathlib.Path
        Sole staged or final candidate named by the durable journal.
    transaction : mapping of str to Any
        Validated current or legacy migration journal.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the candidate is unsafe or its canonical index differs from the journal.
    """
    generations = generation.parent
    _require_migration_generation_identity(generation, transaction)
    if (
        generations.is_symlink()
        or not generations.is_dir()
        or generation.is_symlink()
        or not generation.is_dir()
        or generation.resolve(strict=True).parent != generations.resolve(strict=True)
    ):
        raise ValueError("prepared migration generation is unsafe")
    index_bytes = read_regular_file(
        generation / "index.json", parent=generation.resolve(strict=True)
    )
    if "generation_index_checksum" in transaction:
        matches = sha256_bytes(index_bytes) == transaction["generation_index_checksum"]
    else:
        expected_index = {
            "schema_version": transaction["schema_version"],
            "generation_id": transaction["generation_id"],
            "logical_roots": transaction["logical_roots"],
            **(
                {"preparation_request": transaction["preparation_request"]}
                if "preparation_request" in transaction
                else {}
            ),
        }
        matches = index_bytes == canonical_json_bytes(expected_index)
    if not matches:
        raise ValueError("prepared migration generation index differs from its journal")


def _discard_mismatched_prepared_migration(
    roots: Mapping[str, Path], transaction: Mapping[str, Any]
) -> None:
    """Rollback and remove one journal-owned generation for another request.

    Parameters
    ----------
    roots : mapping of str to pathlib.Path
        Configured logical client, public, and evaluation roots.
    transaction : mapping of str to Any
        Validated migration journal whose request cannot satisfy the caller.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ownership is ambiguous or the candidate cannot be safely identified.
    """
    parent = roots["client"].parent
    generations = parent / PREPARED_GENERATIONS_DIRECTORY
    stage = generations / transaction["stage_name"]
    generation = generations / transaction["generation_id"]
    candidates = [
        candidate
        for candidate in (stage, generation)
        if candidate.exists() or candidate.is_symlink()
    ]
    if len(candidates) != 1:
        raise ValueError("prepared migration generation ownership is ambiguous")
    candidate = candidates[0]
    _validate_owned_generation_index(candidate, transaction)
    _rollback_prepared_migration(roots, transaction)
    shutil.rmtree(candidate)
    _fsync_directory(generations)


def _validate_recovery_generation(
    generation: Path, transaction: Mapping[str, Any]
) -> None:
    """Securely validate one complete journaled generation candidate.

    Parameters
    ----------
    generation : pathlib.Path
        Staged or final generation directory owned by the recovery journal.
    transaction : mapping of str to Any
        Validated canonical migration journal.

    Returns
    -------
    None

    Raises
    ------
    _PreparedGenerationValidationError
        If the index or any public, client, or evaluation artifact is incomplete,
        corrupt, unsafe, or inconsistent.
    """
    try:
        generations = generation.parent
        _require_migration_generation_identity(generation, transaction)
        if (
            generations.is_symlink()
            or not generations.is_dir()
            or generation.is_symlink()
            or not generation.is_dir()
        ):
            raise ValueError("prepared migration generation is unsafe")
        canonical_generations = generations.resolve(strict=True)
        canonical_generation = generation.resolve(strict=True)
        if canonical_generation.parent != canonical_generations:
            raise ValueError("prepared migration generation escapes its root")

        index_bytes = read_regular_file(
            generation / "index.json", parent=canonical_generation
        )
        if sha256_bytes(index_bytes) != transaction["generation_index_checksum"]:
            raise ValueError(
                "prepared migration generation index differs from its journal"
            )
        index = strict_json_loads(index_bytes, source="prepared generation index")
        if not isinstance(index, Mapping) or set(index) != {
            "schema_version",
            "generation_id",
            "inventory",
            "logical_roots",
            "preparation_request",
        }:
            raise ValueError("prepared migration generation index is invalid")
        inventory = validate_prepared_generation_inventory(index["inventory"])
        expected_index = {
            "schema_version": transaction["schema_version"],
            "generation_id": transaction["generation_id"],
            "inventory": inventory,
            "logical_roots": transaction["logical_roots"],
            "preparation_request": transaction["preparation_request"],
        }
        if index_bytes != canonical_json_bytes(expected_index):
            raise ValueError("prepared migration generation index is invalid")
        if prepared_generation_inventory(canonical_generation) != inventory:
            raise ValueError("prepared migration generation inventory differs")

        protocol = load_scientific_protocol()
        app_manifest = load_app_manifest(
            public_artifact_dir=generation / "public", protocol=protocol
        )
        client_root = generation / "client"
        if client_root.is_symlink() or not client_root.is_dir():
            raise ValueError("prepared migration client generation is unsafe")
        canonical_client_root = client_root.resolve(strict=True)
        if canonical_client_root.parent != canonical_generation:
            raise ValueError("prepared migration client generation escapes its root")

        client_ids: list[int] = []
        for child in client_root.iterdir():
            if child.is_symlink() or not child.is_dir():
                raise ValueError("prepared migration client child is unsafe")
            prefix, separator, identifier = child.name.partition("-")
            if (
                prefix != "client"
                or separator != "-"
                or not identifier.isascii()
                or not identifier.isdigit()
                or str(int(identifier)) != identifier
            ):
                raise ValueError("prepared migration client directory is invalid")
            client_ids.append(int(identifier))
        client_ids.sort()
        partitions = transaction["preparation_request"]["partitions"]
        if client_ids != list(range(partitions)):
            raise ValueError("prepared migration client generation is incomplete")

        from src.local_training import load_client_shard_snapshot

        train_spec = protocol["dataset"]["splits"]["train"]
        ordered_rows: list[tuple[str, int] | None] = [None] * train_spec["rows"]
        for client_id in client_ids:
            snapshot = load_client_shard_snapshot(
                client_root / f"client-{client_id}", app_manifest, client_id
            )
            for row_id, text, label in snapshot.rows:
                source_index = int(row_id.removeprefix("train:"))
                if ordered_rows[source_index] is not None:
                    raise ValueError("prepared migration client row identities overlap")
                ordered_rows[source_index] = (text, label)
        if any(row is None for row in ordered_rows):
            raise ValueError(
                "prepared migration clients do not exactly partition the train split"
            )
        train_content = hashlib.sha256()
        train_labels: Counter[int] = Counter()
        for row in ordered_rows:
            assert row is not None
            text, label = row
            source_bytes = canonical_source_row_bytes(text, label)
            train_content.update(source_bytes)
            train_labels[label] += 1
        if train_content.hexdigest() != train_spec["content_sha256"]:
            raise ValueError(
                "prepared migration train content differs from frozen data"
            )
        expected_counts = train_spec.get("label_counts")
        if (
            expected_counts is not None
            and [train_labels[label] for label in range(len(expected_counts))]
            != expected_counts
        ):
            raise ValueError("prepared migration train labels differ from frozen data")

        load_evaluation_artifact_snapshot(generation / "evaluation", protocol=protocol)
    except _PreparedGenerationValidationError:
        raise
    except Exception as error:
        raise _PreparedGenerationValidationError(
            "prepared migration generation validation failed"
        ) from error


def _validate_recovery_generation_or_rollback(
    roots: Mapping[str, Path],
    generation: Path,
    transaction: Mapping[str, Any],
) -> None:
    """Validate a recovery candidate and restore journal-owned visibility on failure.

    Parameters
    ----------
    roots : mapping of str to pathlib.Path
        Configured logical client, public, and evaluation roots.
    generation : pathlib.Path
        Staged or final generation named by the durable journal.
    transaction : mapping of str to Any
        Validated durable migration transaction.

    Returns
    -------
    None
    """
    try:
        _validate_recovery_generation(generation, transaction)
    except _PreparedGenerationValidationError:
        _rollback_prepared_migration(roots, transaction, retain_journal=True)
        raise


def _recover_prepared_migration(
    roots: Mapping[str, Path], preparation_request: Mapping[str, Any]
) -> bool:
    """Finish one journaled prepared-root publication after process death.

    Parameters
    ----------
    roots : mapping of str to pathlib.Path
        Configured logical client, public, and evaluation roots.
    preparation_request : mapping of str to Any
        Current artifact-affecting request that a recovered generation must match.

    Returns
    -------
    bool
        ``True`` when a pending transaction was recovered.

    Raises
    ------
    ValueError
        If recovery encounters missing data or state not owned by the journal.
    """
    transaction = _load_migration_journal(roots)
    if transaction is None:
        return False
    request = validate_preparation_request(preparation_request)
    if transaction["schema_version"] != PREPARED_GENERATION_SCHEMA_VERSION:
        raise ValueError("prepared migration journal lacks durable stage identity")
    if transaction.get("preparation_request") != request:
        _discard_mismatched_prepared_migration(roots, transaction)
        return False
    parent = roots["client"].parent
    generations = parent / PREPARED_GENERATIONS_DIRECTORY
    generation_id = transaction["generation_id"]
    stage = generations / transaction["stage_name"]
    generation = generations / generation_id
    stage_exists = stage.exists() or stage.is_symlink()
    generation_exists = generation.exists() or generation.is_symlink()
    if stage_exists and generation_exists:
        raise ValueError("prepared migration has both staged and final generations")
    if stage_exists:
        _validate_recovery_generation_or_rollback(roots, stage, transaction)
        os.rename(stage, generation)
        _publication_checkpoint("generation:renamed")
    elif not generation_exists:
        raise ValueError("prepared migration generation data is missing")
    else:
        _validate_recovery_generation_or_rollback(roots, generation, transaction)
    _fsync_directory(generations)
    _publication_checkpoint("generations:fsynced")
    _validate_recovery_generation_or_rollback(roots, generation, transaction)

    legacy_names = transaction["legacy_roots"]
    archive: Path | None = None
    if legacy_names:
        legacy_root = parent / PREPARED_LEGACY_DIRECTORY
        if legacy_root.is_symlink() or (
            legacy_root.exists() and not legacy_root.is_dir()
        ):
            raise ValueError("prepared legacy archive must be a regular directory")
        if not legacy_root.exists():
            legacy_root.mkdir(mode=0o755)
            _publication_checkpoint("legacy-root:created")
            _fsync_directory(parent)
            _publication_checkpoint("parent:legacy-root-fsynced")
        archive = legacy_root / generation_id
        if archive.is_symlink() or (archive.exists() and not archive.is_dir()):
            raise ValueError("prepared legacy transaction archive is unsafe")
        if not archive.exists():
            archive.mkdir(mode=0o700)
            _publication_checkpoint("legacy-archive:created")
            _fsync_directory(legacy_root)
            _publication_checkpoint("legacy-root:archive-fsynced")

        for name in legacy_names:
            root = roots[name]
            archived = archive / root.name
            if archived.exists() or archived.is_symlink():
                if archived.is_symlink() or not archived.is_dir():
                    raise ValueError(f"prepared {name} legacy archive is unsafe")
                if root.exists() and not root.is_symlink():
                    raise ValueError(f"prepared {name} legacy root is duplicated")
            elif root.exists() and not root.is_symlink():
                if not root.is_dir():
                    raise ValueError(f"prepared {name} legacy root is unsafe")
                os.rename(root, archived)
                _publication_checkpoint(f"archive:{name}:renamed")
            elif not root.is_symlink():
                raise ValueError(f"prepared {name} legacy data is missing")
            _fsync_directory(archive)
            _publication_checkpoint(f"archive:{name}:fsynced")
            _fsync_directory(parent)
            _publication_checkpoint(f"parent:archive-{name}-fsynced")

    for name in sorted(roots):
        root = roots[name]
        expected = Path(PREPARED_CURRENT_FILENAME) / name
        if root.is_symlink():
            if Path(os.readlink(root)) != expected:
                raise ValueError(f"{name} artifact root has an unsafe prepared alias")
        elif root.exists():
            raise ValueError(f"{name} artifact root was not archived")
        else:
            root.symlink_to(expected, target_is_directory=True)
            _publication_checkpoint(f"alias:{name}:created")
        _fsync_directory(parent)
        _publication_checkpoint(f"parent:alias-{name}-fsynced")

    pointer = parent / PREPARED_CURRENT_FILENAME
    new_target = _migration_target(generation_id)
    previous_target = transaction["previous_pointer_target"]
    if pointer.exists() and not pointer.is_symlink():
        raise ValueError("prepared generation pointer is unsafe")
    current_target = os.readlink(pointer) if pointer.is_symlink() else None
    if current_target not in {previous_target, new_target}:
        raise ValueError("prepared generation pointer changed during migration")
    temporary_pointer = parent / f".{PREPARED_CURRENT_FILENAME}.{generation_id}.tmp"
    _validate_recovery_generation_or_rollback(roots, generation, transaction)
    if current_target != new_target:
        if temporary_pointer.exists() or temporary_pointer.is_symlink():
            if (
                not temporary_pointer.is_symlink()
                or os.readlink(temporary_pointer) != new_target
            ):
                raise ValueError("prepared temporary pointer is unsafe")
        else:
            temporary_pointer.symlink_to(new_target, target_is_directory=True)
            _publication_checkpoint("pointer-temporary:created")
            _fsync_directory(parent)
            _publication_checkpoint("parent:pointer-temporary-fsynced")
        _validate_recovery_generation_or_rollback(roots, generation, transaction)
        os.replace(temporary_pointer, pointer)
        _publication_checkpoint("pointer:replaced")
    else:
        temporary_pointer.unlink(missing_ok=True)
    _fsync_directory(parent)
    _publication_checkpoint("parent:pointer-fsynced")

    journal = parent / PREPARED_MIGRATION_FILENAME
    journal.unlink()
    _publication_checkpoint("journal:removed")
    _fsync_directory(parent)
    _publication_checkpoint("parent:journal-removal-fsynced")
    return True


def _rollback_prepared_migration(
    roots: Mapping[str, Path],
    transaction: Mapping[str, Any],
    *,
    retain_journal: bool = False,
) -> None:
    """Restore pre-transaction visibility after an in-process publication error.

    Parameters
    ----------
    roots : mapping of str to pathlib.Path
        Configured logical artifact roots.
    transaction : mapping of str to Any
        Validated durable migration transaction.
    retain_journal : bool, optional
        Preserve the durable journal so corrected generation contents can retry.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If rollback encounters content whose ownership is not proven by the journal.
    """
    parent = roots["client"].parent
    generation_id = transaction["generation_id"]
    pointer = parent / PREPARED_CURRENT_FILENAME
    new_target = _migration_target(generation_id)
    previous_target = transaction["previous_pointer_target"]
    temporary_pointer = parent / f".{PREPARED_CURRENT_FILENAME}.{generation_id}.tmp"
    ambiguous: list[str] = []
    if temporary_pointer.is_symlink():
        if os.readlink(temporary_pointer) == new_target:
            temporary_pointer.unlink()
        else:
            ambiguous.append("temporary pointer")
    elif temporary_pointer.exists():
        ambiguous.append("temporary pointer")

    current_target = os.readlink(pointer) if pointer.is_symlink() else None
    if current_target == new_target:
        if previous_target is None:
            pointer.unlink()
        else:
            rollback_pointer = parent / f".{PREPARED_CURRENT_FILENAME}.rollback.tmp"
            if rollback_pointer.is_symlink():
                if os.readlink(rollback_pointer) != previous_target:
                    ambiguous.append("rollback pointer")
                else:
                    os.replace(rollback_pointer, pointer)
            elif rollback_pointer.exists():
                ambiguous.append("rollback pointer")
            else:
                rollback_pointer.symlink_to(previous_target, target_is_directory=True)
                os.replace(rollback_pointer, pointer)
    elif pointer.is_symlink():
        if current_target != previous_target:
            ambiguous.append("generation pointer")
    elif pointer.exists() or previous_target is not None:
        ambiguous.append("generation pointer")

    for name in reversed(transaction["alias_roots"]):
        root = roots[name]
        expected = Path(PREPARED_CURRENT_FILENAME) / name
        if root.is_symlink():
            if Path(os.readlink(root)) == expected:
                root.unlink()
            else:
                ambiguous.append(f"{name} artifact root")

    legacy_root = parent / PREPARED_LEGACY_DIRECTORY
    archive = legacy_root / generation_id
    archive_is_safe = not legacy_root.is_symlink() and (
        not legacy_root.exists() or legacy_root.is_dir()
    )
    if not archive_is_safe:
        ambiguous.append("legacy archive root")
    elif archive.is_symlink() or (archive.exists() and not archive.is_dir()):
        ambiguous.append("legacy transaction archive")
        archive_is_safe = False

    if archive_is_safe:
        for name in reversed(transaction["legacy_roots"]):
            root = roots[name]
            archived = archive / root.name
            root_present = root.exists() or root.is_symlink()
            archived_present = archived.exists() or archived.is_symlink()
            if archived.is_symlink() or (archived_present and not archived.is_dir()):
                ambiguous.append(f"{name} legacy archive")
            elif archived_present and not root_present:
                os.rename(archived, root)
            elif archived_present:
                ambiguous.append(f"{name} legacy root")
            elif root.is_symlink() or (root_present and not root.is_dir()):
                ambiguous.append(f"{name} legacy root")
            elif not root_present:
                ambiguous.append(f"{name} legacy data")

        for name in set(transaction["alias_roots"]) - set(transaction["legacy_roots"]):
            root = roots[name]
            if root.exists() or root.is_symlink():
                ambiguous.append(f"{name} artifact root")

    if archive_is_safe and archive.is_dir():
        _fsync_directory(archive)
        _fsync_directory(parent)
        if not any(archive.iterdir()):
            archive.rmdir()
            _fsync_directory(legacy_root)
    _fsync_directory(parent)
    if ambiguous:
        raise ValueError(
            "prepared migration rollback ownership is ambiguous: "
            + ", ".join(sorted(set(ambiguous)))
        )
    if not retain_journal:
        (parent / PREPARED_MIGRATION_FILENAME).unlink(missing_ok=True)
        _fsync_directory(parent)


def _publish_prepared_roots(
    roots: Mapping[str, Path],
    stages: Mapping[str, Path],
    preparation_request: Mapping[str, Any],
) -> None:
    """Publish one immutable generation through a single atomic index update.

    Parameters
    ----------
    roots : mapping of str to pathlib.Path
        Logical client, public, and evaluation roots sharing one parent.
    stages : mapping of str to pathlib.Path
        Complete artifact directories within one owned generation staging root.
    preparation_request : mapping of str to Any
        Artifact-affecting request bound to the generation and recovery journal.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If roots or stages do not form one contained generation.
    OSError
        If immutable publication or the atomic index update fails.
    """
    if set(roots) != {"client", "public", "evaluation"} or set(stages) != set(roots):
        raise ValueError("prepared publication requires all three artifact kinds")
    request = validate_preparation_request(preparation_request)
    parents = {root.parent for root in roots.values()}
    stage_parents = {stage.parent for stage in stages.values()}
    if len(parents) != 1 or len(stage_parents) != 1:
        raise ValueError("prepared roots and stages must each share one parent")
    parent = next(iter(parents))
    generation_stage = next(iter(stage_parents))
    generations = parent / PREPARED_GENERATIONS_DIRECTORY
    if generation_stage.parent != generations:
        raise ValueError("prepared generation staging directory escapes its root")

    generation_id = str(uuid.uuid4())
    index = {
        "schema_version": PREPARED_GENERATION_SCHEMA_VERSION,
        "generation_id": generation_id,
        "inventory": prepared_generation_inventory(generation_stage),
        "logical_roots": {name: roots[name].name for name in sorted(roots)},
        "preparation_request": request,
    }
    index_bytes = canonical_json_bytes(index)
    write_json_atomically(generation_stage / "index.json", index, overwrite=False)
    _fsync_directory_tree(generation_stage)
    pointer = parent / PREPARED_CURRENT_FILENAME
    if pointer.exists() and not pointer.is_symlink():
        raise ValueError("prepared generation pointer is unsafe")
    legacy_roots = {
        name: root
        for name, root in roots.items()
        if root.exists() and not root.is_symlink()
    }
    for name, root in roots.items():
        expected_alias = Path(PREPARED_CURRENT_FILENAME) / name
        if root.is_symlink() and Path(os.readlink(root)) != expected_alias:
            raise ValueError(f"{name} artifact root has an unsafe prepared alias")
    previous_pointer_target = os.readlink(pointer) if pointer.is_symlink() else None
    parent_stat = parent.stat(follow_symlinks=False)
    generations_stat = generations.stat(follow_symlinks=False)
    stage_stat = generation_stage.stat(follow_symlinks=False)
    if not all(
        stat.S_ISDIR(item.st_mode)
        for item in (parent_stat, generations_stat, stage_stat)
    ):
        raise ValueError("prepared migration generation is unsafe")
    transaction = {
        "schema_version": PREPARED_GENERATION_SCHEMA_VERSION,
        "generation_id": generation_id,
        "stage_name": generation_stage.name,
        "generation_index_checksum": sha256_bytes(index_bytes),
        "parent_device": parent_stat.st_dev,
        "parent_inode": parent_stat.st_ino,
        "generations_device": generations_stat.st_dev,
        "generations_inode": generations_stat.st_ino,
        "stage_device": stage_stat.st_dev,
        "stage_inode": stage_stat.st_ino,
        "logical_roots": {name: roots[name].name for name in sorted(roots)},
        "preparation_request": request,
        "legacy_roots": sorted(legacy_roots),
        "alias_roots": sorted(
            name for name, root in roots.items() if not root.is_symlink()
        ),
        "previous_pointer_target": previous_pointer_target,
    }
    journal = parent / PREPARED_MIGRATION_FILENAME
    write_json_atomically(journal, transaction, overwrite=False)
    _publication_checkpoint("journal:published")
    _fsync_directory(parent)
    _publication_checkpoint("parent:journal-fsynced")
    try:
        if not _recover_prepared_migration(roots, request):
            raise RuntimeError("prepared migration journal disappeared")
    except _PreparedGenerationValidationError:
        raise
    except BaseException:
        _rollback_prepared_migration(roots, transaction)
        raise


def _raw_split_path(
    dataset_id: str,
    config: str,
    revision: str,
    split: str,
    *,
    local_files_only: bool,
) -> Path:
    """Resolve the immutable raw Parquet file for one dataset split.

    Parameters
    ----------
    dataset_id : str
        Hugging Face dataset repository identifier.
    config : str
        Dataset configuration name.
    revision : str
        Immutable dataset revision.
    split : str
        Split whose Parquet file is required.
    local_files_only : bool
        Require the file to exist in the local cache.

    Returns
    -------
    pathlib.Path
        Resolved cached Parquet path.

    Raises
    ------
    huggingface_hub.errors.HfHubHTTPError
        If the immutable file cannot be resolved from the configured source.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=dataset_id,
            filename=f"{config}/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def _validate_loaded_dataset(
    dataset: Any,
    protocol: Mapping[str, Any],
    raw_paths: Mapping[str, Path],
) -> None:
    """Validate loaded dataset identity, rows, labels, and content hashes.

    Parameters
    ----------
    dataset : Any
        Loaded dataset mapping keyed by split name.
    protocol : mapping of str to Any
        Frozen scientific protocol.
    raw_paths : mapping of str to pathlib.Path
        Immutable raw Parquet paths keyed by split name.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any split differs from its frozen identity or checksum contract.
    """
    dataset_spec = protocol["dataset"]
    split_specs = dataset_spec["splits"]
    if set(dataset) != set(split_specs):
        raise ValueError("loaded dataset splits differ from the frozen protocol")

    for split, split_spec in split_specs.items():
        raw_digest = sha256_file(raw_paths[split]).removeprefix("sha256:")
        if raw_digest != split_spec["raw_parquet_sha256"]:
            raise ValueError(f"{split} raw Parquet SHA-256 differs from the protocol")

        loaded_split = dataset[split]
        info = loaded_split.info
        if (
            info.dataset_name != dataset_spec["id"].split("/", maxsplit=1)[1]
            or info.config_name != dataset_spec["config"]
            or loaded_split.column_names != ["text", "label"]
        ):
            raise ValueError("loaded dataset identity differs from the frozen protocol")
        if len(loaded_split) != split_spec["rows"]:
            raise ValueError(f"{split} row count differs from the frozen protocol")

        content_hash = hashlib.sha256()
        label_counts: Counter[int] = Counter()
        label_ranges = split_spec.get("label_index_ranges")
        for index, row in enumerate(loaded_split):
            text = row.get("text")
            label = row.get("label")
            if not isinstance(text, str) or type(label) is not int:
                raise ValueError(f"{split} row {index} has invalid text or label data")
            if label_ranges is not None:
                expected_label = next(
                    (
                        range_label
                        for range_label, (start, end) in enumerate(label_ranges)
                        if start <= index <= end
                    ),
                    None,
                )
                if label != expected_label:
                    raise ValueError(
                        f"{split} labels differ from the frozen official row ranges"
                    )
            content_hash.update(canonical_source_row_bytes(text, label))
            label_counts[label] += 1

        expected_counts = split_spec.get("label_counts")
        if (
            expected_counts is not None
            and [label_counts[label] for label in range(len(expected_counts))]
            != expected_counts
        ):
            raise ValueError(f"{split} label counts differ from the frozen protocol")
        if content_hash.hexdigest() != split_spec["content_sha256"]:
            raise ValueError(
                f"{split} canonical content SHA-256 differs from the frozen protocol"
            )


def load_verified_imdb_dataset(*, local_files_only: bool = False) -> Any:
    """Load and verify every frozen IMDB split before artifact preparation.

    Parameters
    ----------
    local_files_only : bool, optional
        Require all dataset metadata and bytes to be available in local caches.

    Returns
    -------
    datasets.DatasetDict
        Verified official train, test, and unsupervised splits.

    Raises
    ------
    ValueError
        If package version, identity, rows, labels, or checksums differ from the
        frozen scientific protocol.
    """
    from datasets import DownloadConfig, load_dataset

    protocol = load_scientific_protocol()
    dataset_spec = protocol["dataset"]
    installed_version = importlib.metadata.version("datasets")
    if installed_version != dataset_spec["datasets_version"]:
        raise ValueError(
            "datasets version differs from the frozen protocol: "
            f"expected {dataset_spec['datasets_version']}, got {installed_version}"
        )

    raw_paths = {
        split: _raw_split_path(
            dataset_spec["id"],
            dataset_spec["config"],
            dataset_spec["revision"],
            split,
            local_files_only=local_files_only,
        )
        for split in dataset_spec["splits"]
    }
    dataset = load_dataset(
        dataset_spec["id"],
        dataset_spec["config"],
        revision=dataset_spec["revision"],
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    _validate_loaded_dataset(dataset, protocol, raw_paths)
    return dataset


def _validated_frameworks() -> tuple[Any, Any, Mapping[str, Any]]:
    """Load frameworks only when their versions match the frozen protocol.

    Returns
    -------
    tuple
        Keras, TensorFlow, and the parsed frozen protocol.

    Raises
    ------
    ValueError
        If any registered framework version differs from the runtime.
    """
    import keras
    import tensorflow as tf

    protocol = validate_protocol_runtime()
    return keras, tf, protocol


def build_vectorizer(texts: Any) -> Any:
    """Build and verify the shared public token-ID contract.

    Parameters
    ----------
    texts : Any
        Complete official training texts in ascending row-index order.

    Returns
    -------
    keras.layers.TextVectorization
        Adapted vectorizer matching the frozen vocabulary contract.
    """
    _, tf, protocol = _validated_frameworks()
    preprocessing = protocol["preprocessing"]
    dimensions = protocol_model_dimensions(protocol)
    vectorizer = create_text_vectorizer(
        sequence_length=dimensions["sequence_length"],
        max_tokens=preprocessing["max_tokens"],
    )
    vectorizer.adapt(
        tf.data.Dataset.from_tensor_slices(list(texts)).batch(256, drop_remainder=False)
    )
    vocabulary = vectorizer.get_vocabulary()
    vocabulary_bytes = b"".join(item.encode("utf-8") + b"\n" for item in vocabulary)
    if len(vocabulary) != preprocessing["vocabulary_size"]:
        raise ValueError("vocabulary size differs from the frozen protocol")
    if (
        hashlib.sha256(vocabulary_bytes).hexdigest()
        != preprocessing["vocabulary_sha256"]
    ):
        raise ValueError("vocabulary SHA-256 differs from the frozen protocol")
    return vectorizer


def package_raw_client_shards(
    texts: Any,
    labels: Any,
    output_dir: str | Path,
    num_clients: int,
    *,
    alpha: float = DEFAULT_DIRICHLET_ALPHA,
    seed: int = DEFAULT_SPLIT_SEED,
    manifest: Mapping[str, Any],
    dataset: Mapping[str, Any],
    source_split: str = "train",
) -> list[Path]:
    """Write raw review text and labels into one directory per client.

    Parameters
    ----------
    texts : Any
        Review texts to partition.
    labels : Any
        Labels corresponding to ``texts``.
    output_dir : str or pathlib.Path
        Parent directory for client shards.
    num_clients : int
        Number of client shards to create.
    alpha : float, optional
        Dirichlet concentration parameter.
    seed : int, optional
        Random seed for deterministic partitioning.
    manifest : mapping
        Canonical public manifest bound into every shard.
    dataset : mapping
        Frozen official train-dataset identity bound into every shard.
    source_split : str, optional
        Official split name used to qualify stable row identities.

    Returns
    -------
    list of pathlib.Path
        Created client shard directories.
    """
    text_array = np.asarray(texts)
    label_array = np.asarray(labels)
    if len(text_array) != len(label_array):
        raise ValueError(
            "text/label sample count mismatch: "
            f"len(texts)={len(text_array)} len(labels)={len(label_array)}"
        )
    if source_split != "train" or any(
        type(label) is not int or label not in (0, 1) for label in label_array.tolist()
    ):
        raise ValueError("client shards require binary labels from the train split")

    output_path = _preflight_output_root(output_dir, "client", reusable=True)
    output_path.mkdir(parents=True, exist_ok=True)
    canonical_output = output_path.resolve(strict=True)
    for child in output_path.iterdir():
        if child.name.startswith("client-"):
            if child.is_dir():
                _validate_client_directory(child, canonical_output)
            else:
                _validate_output_child(child, canonical_output, directory=False)
    manifest_bytes = canonical_json_bytes(manifest)

    split = dirichlet_split(
        label_array,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    staging_root = Path(
        tempfile.mkdtemp(dir=output_path, prefix=".shards.", suffix=".tmp")
    )
    try:
        for client_id in range(num_clients):
            indices = split[client_id]
            client_labels = label_array[indices]
            records_bytes = b"".join(
                canonical_client_row_bytes(
                    f"{source_split}:{source_index}",
                    str(text_array[source_index]),
                    int(label),
                )
                for source_index, label in zip(indices, client_labels, strict=True)
            )
            shard_dir = staging_root / f"client-{client_id}"
            shard_dir.mkdir()
            _write_bytes_atomically(shard_dir / "reviews.jsonl", records_bytes)
            shard_metadata = client_shard_metadata(
                client_id,
                client_labels,
                records_bytes=records_bytes,
                public_manifest_bytes=manifest_bytes,
                dataset=dataset,
                source_split=source_split,
                split_seed=seed,
                alpha=alpha,
            )
            write_json_atomically(shard_dir / "client_metadata.json", shard_metadata)

        for child in list(output_path.iterdir()):
            if child.name.startswith("client-"):
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        shard_paths = []
        for client_id in range(num_clients):
            shard_dir = output_path / f"client-{client_id}"
            os.rename(staging_root / shard_dir.name, shard_dir)
            shard_paths.append(shard_dir)
        return shard_paths
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def publish_public_artifacts(
    vectorizer: Any,
    output_dir: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the vocabulary and its train-only provenance contract.

    Parameters
    ----------
    vectorizer : Any
        Verified training-split vectorizer.
    output_dir : str or pathlib.Path
        Public artifact directory.
    protocol : mapping or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    dict of str to Any
        Published public manifest.
    """
    output_path = _preflight_output_root(output_dir, "public", reusable=True)
    output_path.mkdir(parents=True, exist_ok=True)
    unexpected = {
        child.name for child in output_path.iterdir()
    } - PUBLIC_ARTIFACT_FILENAMES
    if unexpected:
        raise ValueError(
            "public artifact contains unexpected files: "
            + ", ".join(sorted(unexpected))
        )
    vocabulary = vectorizer.get_vocabulary()
    vocabulary_bytes = b"".join(item.encode("utf-8") + b"\n" for item in vocabulary)
    frozen = protocol or load_scientific_protocol()
    preprocessing = frozen["preprocessing"]
    dimensions = protocol_model_dimensions(frozen)
    vocabulary_sha256 = hashlib.sha256(vocabulary_bytes).hexdigest()
    if len(vocabulary) != preprocessing["vocabulary_size"]:
        raise ValueError("vocabulary size differs from the frozen protocol")
    if vocabulary_sha256 != preprocessing["vocabulary_sha256"]:
        raise ValueError("vocabulary SHA-256 differs from the frozen protocol")
    canonical_output = output_path.resolve(strict=True)
    for filename in ("vocab.txt", "manifest.json"):
        _validate_output_child(
            output_path / filename, canonical_output, directory=False
        )
    _write_bytes_atomically(output_path / "vocab.txt", vocabulary_bytes)
    dataset = frozen["dataset"]
    train = dataset["splits"]["train"]
    manifest = {
        "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        **dimensions,
        "vocabulary": {
            "filename": "vocab.txt",
            "sha256": vocabulary_sha256,
            "size_bytes": len(vocabulary_bytes),
        },
        "dataset": {
            "id": dataset["id"],
            "config": dataset["config"],
            "revision": dataset["revision"],
            "datasets_version": dataset["datasets_version"],
            "split": "train",
            "rows": train["rows"],
            "raw_parquet_sha256": train["raw_parquet_sha256"],
            "content_sha256": train["content_sha256"],
        },
    }
    write_json_atomically(output_path / "manifest.json", manifest)
    if {child.name for child in output_path.iterdir()} != PUBLIC_ARTIFACT_FILENAMES:
        raise ValueError("public artifact contains unexpected files")
    return manifest


def prepare_all(
    partitions: int,
    client_shard_dir: str | Path,
    public_artifact_dir: str | Path,
    evaluation_artifact_dir: str | Path,
) -> None:
    """Create train-only client/public artifacts and the separate test artifact.

    Parameters
    ----------
    partitions : int
        Number of client training shards.
    client_shard_dir : str or pathlib.Path
        Parent directory for train-only client shards.
    public_artifact_dir : str or pathlib.Path
        Directory for the shared train-derived vocabulary.
    evaluation_artifact_dir : str or pathlib.Path
        New immutable directory for the untouched official test split.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If direct preparation runs outside the supported Linux platform.
    """
    require_secure_artifact_platform()
    request = validate_preparation_request({"partitions": partitions})
    client_dir = _preflight_output_root(
        client_shard_dir, "client", reusable=True, allow_prepared_alias=True
    )
    public_dir = _preflight_output_root(
        public_artifact_dir, "public", reusable=True, allow_prepared_alias=True
    )
    evaluation_dir = _preflight_output_root(
        evaluation_artifact_dir,
        "evaluation",
        reusable=True,
        allow_prepared_alias=True,
    )
    roots = {
        "client": client_dir,
        "public": public_dir,
        "evaluation": evaluation_dir,
    }
    for first_name, first in roots.items():
        for second_name, second in roots.items():
            if first_name >= second_name:
                continue
            if (
                first == second
                or first.is_relative_to(second)
                or second.is_relative_to(first)
            ):
                raise ValueError(
                    f"{first_name} and {second_name} artifact roots must be separate"
                )
    for root in roots.values():
        root.parent.mkdir(parents=True, exist_ok=True)
    if len({root.parent for root in roots.values()}) != 1:
        raise ValueError(
            "client, public, and evaluation artifact roots must share one parent for "
            "atomic generation publication"
        )
    lock = _acquire_preparation_lock(roots)
    stages: dict[str, Path] = {}
    generation_chain: RetainedDirectoryChain | None = None
    try:
        lock.verify()
        parent = next(iter({root.parent for root in roots.values()}))
        generations = parent / PREPARED_GENERATIONS_DIRECTORY
        if generations.is_symlink() or (
            generations.exists() and not generations.is_dir()
        ):
            raise ValueError("prepared generations root must be a regular directory")
        generations.mkdir(mode=0o755, exist_ok=True)
        lock.verify()
        generation_chain = RetainedDirectoryChain.open(
            generations,
            error_message="prepared generations directory changed during preparation",
            check_platform=False,
        )
        if _recover_prepared_migration(roots, request):
            _recover_preparation_stage(generation_chain)
            lock.verify()
            return
        _recover_preparation_stage(generation_chain)
        lock.verify()
        parent_descriptor = generation_chain.directory.descriptor
        parent_stat = os.fstat(parent_descriptor)
        nonce = uuid.uuid4().hex
        stage_name = f".prepare-{nonce}.staging"
        stage_state = _PreparationStageState(
            "reserved",
            parent_stat.st_dev,
            parent_stat.st_ino,
            nonce,
            stage_name,
        )
        stage_document = _write_preparation_stage_state(
            generation_chain, stage_state, None
        )
        generation_chain.verify()
        os.mkdir(stage_name, mode=0o700, dir_fd=parent_descriptor)
        generation_chain.verify()
        stage_descriptor = generation_chain.open_child(
            stage_name, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            stage_stat = os.fstat(stage_descriptor)
            stage_state = _PreparationStageState(
                "build",
                parent_stat.st_dev,
                parent_stat.st_ino,
                nonce,
                stage_name,
                stage_stat.st_dev,
                stage_stat.st_ino,
            )
            _write_preparation_stage_state(
                generation_chain, stage_state, stage_document
            )
        finally:
            os.close(stage_descriptor)
        generation_stage = generations / stage_name
        stages = {name: generation_stage / name for name in roots}
        stages["public"].mkdir()
        stages["client"].mkdir()

        _validated_frameworks()
        dataset = load_verified_imdb_dataset()
        train = dataset["train"]
        texts = np.asarray(train["text"])
        labels = np.asarray(train["label"], dtype="int32")
        vectorizer = build_vectorizer(texts)
        protocol = load_scientific_protocol()
        manifest = publish_public_artifacts(
            vectorizer, stages["public"], protocol=protocol
        )
        package_raw_client_shards(
            texts,
            labels,
            stages["client"],
            num_clients=partitions,
            manifest=manifest,
            dataset=manifest["dataset"],
        )
        publish_evaluation_artifact(
            dataset["test"], stages["evaluation"], protocol=protocol
        )

        app_manifest = load_app_manifest(
            public_artifact_dir=stages["public"], protocol=protocol
        )
        identities: set[str] = set()
        for client_id in range(partitions):
            from src.local_training import load_client_shard_snapshot

            snapshot = load_client_shard_snapshot(
                stages["client"] / f"client-{client_id}",
                app_manifest,
                client_id,
            )
            shard_identities = {row[0] for row in snapshot.rows}
            if identities & shard_identities:
                raise ValueError("client shard row identities overlap")
            identities.update(shard_identities)
        if identities != {
            f"train:{index}" for index in range(manifest["dataset"]["rows"])
        }:
            raise ValueError("client shards do not exactly partition the train split")
        load_evaluation_artifact_snapshot(stages["evaluation"], protocol=protocol)
        for stage in stages.values():
            _fsync_directory_tree(stage)
        generation_chain.verify()
        verification_descriptor = _open_bound_preparation_stage(
            generation_chain, stage_name, stage_state
        )
        os.close(verification_descriptor)
        lock.verify()
        _publish_prepared_roots(roots, stages, request)
        lock.verify()
        _recover_preparation_stage(generation_chain)
        lock.verify()
        stages = {}
    finally:
        try:
            if stages and generation_chain is not None:
                _recover_preparation_stage(generation_chain)
        finally:
            if generation_chain is not None:
                generation_chain.close()
            lock.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse data-preparation command-line arguments.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    argparse.Namespace
        Validated preparation options.
    """
    parser = argparse.ArgumentParser(description="Prepare sentiment data partitions.")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument(
        "--client-shard-dir", type=Path, default=Path("artifacts/clients")
    )
    parser.add_argument(
        "--public-artifact-dir", type=Path, default=default_public_artifact_dir()
    )
    parser.add_argument(
        "--evaluation-artifact-dir",
        type=Path,
        default=default_evaluation_artifact_dir(),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run verified train/test artifact preparation.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit command-line arguments, or process arguments when omitted.

    Returns
    -------
    None
    """
    args = parse_args(argv)
    prepare_all(
        partitions=args.partitions,
        client_shard_dir=args.client_shard_dir,
        public_artifact_dir=args.public_artifact_dir,
        evaluation_artifact_dir=args.evaluation_artifact_dir,
    )


if __name__ == "__main__":
    main()
