"""Manage immutable run directories and deterministic artifact retention."""

from __future__ import annotations

import base64
import binascii
import hmac
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator, Mapping

if TYPE_CHECKING:
    from src.app_manifest import AppManifest

from src.artifact_compatibility import (
    ARTIFACT_SCHEMA_VERSION,
    RetainedDirectory as _RetainedDirectory,
    RetainedDirectoryChain,
    SERVER_ARTIFACT_MANIFEST_FILENAME,
    RegularFileSnapshot,
    ServerArtifactSnapshot,
    ServerFinalizationSnapshot,
    canonical_json_bytes,
    load_server_artifact_snapshot,
    read_regular_file,
    read_regular_file_snapshot,
    require_secure_artifact_platform,
    sha256_bytes,
    strict_json_loads,
    validate_run_provenance_evidence,
    write_bytes_atomically,
    write_server_artifact_manifest,
)
from src.paths import run_manifest_path
from src.run_provenance import (
    load_run_provenance_manifest,
    write_run_provenance_manifest,
)

RUNS_DIRECTORY = "runs"
CURRENT_RUN_FILENAME = "current.json"
DEFAULT_ARTIFACT_RETENTION_RUNS = 10
_PUBLICATION_PENDING = "pending"
_PUBLICATION_COMPLETE = "complete"
_FINALIZATION_STATE_SCHEMA_VERSION = 1
_TEMPORARY_NAME_ATTEMPTS = 128
_ACTIVE_HISTORY_CHAIN: ContextVar[RetainedDirectoryChain | None] = ContextVar(
    "active_history_chain", default=None
)


@dataclass(frozen=True)
class _RetainedFile:
    """Keep one regular-file descriptor and its immutable capture."""

    descriptor: int
    snapshot: RegularFileSnapshot


@dataclass(frozen=True)
class _HistoryDescriptors:
    """Retain the canonical root-to-run directory chain."""

    chain: RetainedDirectoryChain
    root_parent: _RetainedDirectory
    root: _RetainedDirectory
    runs: _RetainedDirectory
    run: _RetainedDirectory
    root_name: str
    run_name: str


@dataclass(frozen=True)
class _HistoryRootDescriptors:
    """Retain the canonical root and runs directory chain."""

    chain: RetainedDirectoryChain
    root_parent: _RetainedDirectory
    root: _RetainedDirectory
    runs: _RetainedDirectory
    root_name: str


def _directory_identity(descriptor: int) -> tuple[int, int]:
    """Return the stable device and inode identity of a directory descriptor.

    Parameters
    ----------
    descriptor : int
        Open directory descriptor.

    Returns
    -------
    tuple of int
        Device and inode identity.
    """
    file_stat = os.fstat(descriptor)
    if not stat.S_ISDIR(file_stat.st_mode):
        raise ValueError("artifact history changed during finalization")
    return file_stat.st_dev, file_stat.st_ino


def _open_retained_directory(
    name: str | Path, *, parent_descriptor: int | None = None
) -> _RetainedDirectory:
    """Open one directory without following its final path component.

    Parameters
    ----------
    name : str or pathlib.Path
        Absolute path or descriptor-relative entry name.
    parent_descriptor : int or None, optional
        Parent directory descriptor for a relative entry.

    Returns
    -------
    _RetainedDirectory
        Open descriptor and stable identity.

    Raises
    ------
    ValueError
        If the entry is missing, linked, or not a directory.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = (
            os.open(name, flags)
            if parent_descriptor is None
            else os.open(name, flags, dir_fd=parent_descriptor)
        )
    except OSError as error:
        raise ValueError("artifact history changed during finalization") from error
    try:
        device, inode = _directory_identity(descriptor)
        return _RetainedDirectory(descriptor, device, inode)
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def _retain_history_descriptors(
    root: Path, runs_root: Path, run_dir: Path
) -> Iterator[_HistoryDescriptors]:
    """Retain the exact root, runs, and selected-run directory chain.

    Parameters
    ----------
    root : pathlib.Path
        Canonical artifact root.
    runs_root : pathlib.Path
        Canonical ``runs`` directory.
    run_dir : pathlib.Path
        Canonical selected run directory.

    Yields
    ------
    _HistoryDescriptors
        Secure open descriptors for finalization.
    """
    require_secure_artifact_platform()
    chain = RetainedDirectoryChain.open(
        run_dir,
        error_message="artifact history changed during finalization",
        check_platform=False,
    )
    with chain:
        root_parent = chain.at(root.parent)
        root_descriptor = chain.at(root)
        runs_descriptor = chain.at(runs_root)
        run_descriptor = chain.at(run_dir)
        descriptors = _HistoryDescriptors(
            chain,
            root_parent,
            root_descriptor,
            runs_descriptor,
            run_descriptor,
            root.name,
            run_dir.name,
        )
        _verify_history_descriptors(descriptors)
        token = _ACTIVE_HISTORY_CHAIN.set(chain)
        try:
            yield descriptors
        finally:
            _ACTIVE_HISTORY_CHAIN.reset(token)


def _verify_directory_entry(
    parent_descriptor: int, name: str, retained: _RetainedDirectory
) -> None:
    """Require one directory entry to retain its captured identity.

    Parameters
    ----------
    parent_descriptor : int
        Retained parent directory descriptor.
    name : str
        Direct child entry name.
    retained : _RetainedDirectory
        Expected directory identity.

    Returns
    -------
    None
    """
    if not RetainedDirectoryChain.entry_matches_descriptor(
        parent_descriptor, name, retained.descriptor
    ) or _directory_identity(retained.descriptor) != (
        retained.device,
        retained.inode,
    ):
        raise ValueError("artifact history changed during finalization")


def _verify_history_descriptors(descriptors: _HistoryDescriptors) -> None:
    """Revalidate every retained directory entry in the history chain.

    Parameters
    ----------
    descriptors : _HistoryDescriptors
        Retained root-to-run descriptor chain.

    Returns
    -------
    None
    """
    try:
        descriptors.chain.verify()
    except ValueError as error:
        raise ValueError("artifact history changed during finalization") from error


@contextmanager
def _retain_history_root(
    root: Path, runs_root: Path
) -> Iterator[_HistoryRootDescriptors]:
    """Retain and verify the canonical root-to-runs directory chain.

    Parameters
    ----------
    root : pathlib.Path
        Canonical artifact root.
    runs_root : pathlib.Path
        Canonical runs directory.

    Yields
    ------
    _HistoryRootDescriptors
        Retained root and runs descriptors.
    """
    require_secure_artifact_platform()
    chain = RetainedDirectoryChain.open(
        runs_root,
        error_message="artifact history changed during finalization",
        check_platform=False,
    )
    with chain:
        root_parent = chain.at(root.parent)
        retained_root = chain.at(root)
        retained_runs = chain.at(runs_root)
        descriptors = _HistoryRootDescriptors(
            chain, root_parent, retained_root, retained_runs, root.name
        )
        _verify_history_root(descriptors)
        token = _ACTIVE_HISTORY_CHAIN.set(chain)
        try:
            yield descriptors
        finally:
            _ACTIVE_HISTORY_CHAIN.reset(token)


def _verify_history_root(descriptors: _HistoryRootDescriptors) -> None:
    """Revalidate the retained artifact root and runs entries.

    Parameters
    ----------
    descriptors : _HistoryRootDescriptors
        Retained root-to-runs descriptor chain.

    Returns
    -------
    None
    """
    try:
        descriptors.chain.verify()
    except ValueError as error:
        raise ValueError("artifact history changed during finalization") from error


def _read_descriptor_bytes(descriptor: int) -> tuple[bytes, os.stat_result]:
    """Read exact bytes while requiring stable descriptor metadata.

    Parameters
    ----------
    descriptor : int
        Open regular-file descriptor.

    Returns
    -------
    tuple of bytes and os.stat_result
        Exact content and stable descriptor metadata.
    """
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("server artifact changed during finalization")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    content = b"".join(chunks)
    after = os.fstat(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if (
        identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        or len(content) != after.st_size
    ):
        raise ValueError("server artifact changed during finalization")
    return content, after


def _open_retained_file(parent_descriptor: int, name: str) -> _RetainedFile:
    """Open and capture one no-follow, single-link regular file.

    Parameters
    ----------
    parent_descriptor : int
        Owning directory descriptor.
    name : str
        Direct child filename.

    Returns
    -------
    _RetainedFile
        Open descriptor and exact captured bytes and identity.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ValueError("server artifact changed during finalization") from error
    try:
        content, file_stat = _read_descriptor_bytes(descriptor)
        snapshot = RegularFileSnapshot(
            content,
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )
        return _RetainedFile(descriptor, snapshot)
    except BaseException:
        os.close(descriptor)
        raise


def _retained_file_entry_matches(
    parent_descriptor: int,
    name: str,
    retained: _RetainedFile,
    expected_content: bytes,
) -> bool:
    """Return whether a name still selects one retained exact file.

    Parameters
    ----------
    parent_descriptor : int
        Owning directory descriptor.
    name : str
        Direct child filename.
    retained : _RetainedFile
        Retained file identity and descriptor.
    expected_content : bytes
        Exact bytes required from the retained file.

    Returns
    -------
    bool
        Whether the entry, descriptor identity, and bytes all still match.
    """
    try:
        entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        content, current = _read_descriptor_bytes(retained.descriptor)
    except (OSError, ValueError):
        return False
    captured = retained.snapshot
    return (
        stat.S_ISREG(entry.st_mode)
        and entry.st_nlink == 1
        and (entry.st_dev, entry.st_ino) == (captured.device, captured.inode)
        and (current.st_dev, current.st_ino) == (captured.device, captured.inode)
        and content == expected_content
    )


def _require_retained_file_entry(
    parent_descriptor: int,
    name: str,
    retained: _RetainedFile,
    expected_content: bytes,
) -> None:
    """Require a name to select one retained file with exact bytes.

    Parameters
    ----------
    parent_descriptor : int
        Owning directory descriptor.
    name : str
        Direct child filename.
    retained : _RetainedFile
        Retained file identity and descriptor.
    expected_content : bytes
        Exact bytes required from the retained file.

    Returns
    -------
    None
    """
    if not _retained_file_entry_matches(
        parent_descriptor, name, retained, expected_content
    ):
        raise ValueError("installed artifact entry changed during finalization")


def _unlink_retained_file_entry(
    parent_descriptor: int,
    name: str,
    retained: _RetainedFile,
    expected_content: bytes,
) -> bool:
    """Unlink a name only while it still selects the retained exact file.

    Parameters
    ----------
    parent_descriptor : int
        Owning directory descriptor.
    name : str
        Direct child filename.
    retained : _RetainedFile
        Retained file identity and descriptor.
    expected_content : bytes
        Exact bytes required from the retained file.

    Returns
    -------
    bool
        Whether the proven-owned entry was unlinked.
    """
    if not _retained_file_entry_matches(
        parent_descriptor, name, retained, expected_content
    ):
        return False
    os.unlink(name, dir_fd=parent_descriptor)
    return True


@contextmanager
def _capture_run_inventory(
    run_descriptor: int, *, exclude_manifest: bool
) -> Iterator[Mapping[str, _RetainedFile]]:
    """Capture and retain every regular entry in one run directory.

    Parameters
    ----------
    run_descriptor : int
        Retained selected-run directory descriptor.
    exclude_manifest : bool
        Omit the artifact manifest while constructing its checksums.

    Yields
    ------
    collections.abc.Mapping
        Immutable filename-to-retained-file mapping.
    """
    files: dict[str, _RetainedFile] = {}
    try:
        for name in sorted(os.listdir(run_descriptor)):
            if exclude_manifest and name == SERVER_ARTIFACT_MANIFEST_FILENAME:
                continue
            try:
                files[name] = _open_retained_file(run_descriptor, name)
            except ValueError as error:
                raise ValueError("server artifact is missing or unsafe") from error
        yield MappingProxyType(files)
    finally:
        for retained in files.values():
            os.close(retained.descriptor)


def _snapshot_from_inventory(
    inventory: Mapping[str, _RetainedFile],
) -> ServerFinalizationSnapshot:
    """Adapt retained files to the compatibility manifest writer.

    Parameters
    ----------
    inventory : mapping of str to _RetainedFile
        Securely captured finalizable files.

    Returns
    -------
    ServerFinalizationSnapshot
        Immutable compatibility-layer snapshot.
    """
    return ServerFinalizationSnapshot(
        MappingProxyType(
            {name: retained.snapshot for name, retained in inventory.items()}
        )
    )


def _verify_run_inventory(
    run_descriptor: int,
    inventory: Mapping[str, _RetainedFile],
    expected_content: Mapping[str, bytes],
) -> None:
    """Revalidate exact entries, identities, metadata, and bytes.

    Parameters
    ----------
    run_descriptor : int
        Retained selected-run directory descriptor.
    inventory : mapping of str to _RetainedFile
        Entire retained completed-run inventory.
    expected_content : mapping of str to bytes
        Validated manifest and artifact bytes.

    Returns
    -------
    None
    """
    if set(os.listdir(run_descriptor)) != set(inventory) or set(inventory) != set(
        expected_content
    ):
        raise ValueError("server artifact inventory changed during finalization")
    for name, retained in inventory.items():
        try:
            entry = os.stat(name, dir_fd=run_descriptor, follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"server artifact changed during finalization: {name}"
            ) from error
        captured = retained.snapshot
        identity = (
            entry.st_dev,
            entry.st_ino,
            entry.st_size,
            entry.st_mtime_ns,
            entry.st_ctime_ns,
        )
        content, current = _read_descriptor_bytes(retained.descriptor)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or identity
            != (
                captured.device,
                captured.inode,
                captured.size_bytes,
                captured.modified_ns,
                captured.changed_ns,
            )
            or (current.st_dev, current.st_ino) != (entry.st_dev, entry.st_ino)
            or content != captured.content
            or content != expected_content[name]
        ):
            raise ValueError(f"server artifact changed during finalization: {name}")


def _sync_retained_files(inventory: Mapping[str, _RetainedFile]) -> None:
    """Flush every retained artifact descriptor.

    Parameters
    ----------
    inventory : mapping of str to _RetainedFile
        Retained regular-file descriptors.

    Returns
    -------
    None
    """
    for retained in inventory.values():
        os.fsync(retained.descriptor)


def _sync_retained_directory(directory: _RetainedDirectory) -> None:
    """Flush one retained directory descriptor.

    Parameters
    ----------
    directory : _RetainedDirectory
        Retained directory descriptor.

    Returns
    -------
    None
    """
    os.fsync(directory.descriptor)


def _sync_visible_directory(
    directory: _RetainedDirectory,
    chain: RetainedDirectoryChain | None = None,
) -> None:
    """Flush one directory while its complete retained chain stays visible.

    Parameters
    ----------
    directory : _RetainedDirectory
        Retained directory descriptor to flush.
    chain : RetainedDirectoryChain or None, optional
        Complete visible chain, inferred from the active history operation.

    Returns
    -------
    None
    """
    retained_chain = chain or _ACTIVE_HISTORY_CHAIN.get()
    if retained_chain is not None:
        retained_chain.verify()
    _sync_retained_directory(directory)
    if retained_chain is not None:
        retained_chain.verify()


@dataclass(frozen=True)
class _PointerBinding:
    """Bind exact current-pointer bytes to their validated identity."""

    run_id: str
    artifact_manifest_checksum: str
    content: bytes

    def payload(self) -> Mapping[str, str]:
        """Return the exact-schema JSON representation.

        Returns
        -------
        collections.abc.Mapping
            Canonical pointer binding payload.
        """
        return {
            "run_id": self.run_id,
            "artifact_manifest_checksum": self.artifact_manifest_checksum,
            "bytes_base64": base64.b64encode(self.content).decode("ascii"),
            "bytes_checksum": sha256_bytes(self.content),
        }


@dataclass(frozen=True)
class _FinalizationState:
    """Durably bind one publication attempt to old and new pointer bytes."""

    status: str
    candidate: _PointerBinding
    previous: _PointerBinding | None

    def payload(self) -> Mapping[str, Any]:
        """Return the exact-schema JSON representation.

        Returns
        -------
        collections.abc.Mapping
            Canonical finalization-state payload.
        """
        return {
            "schema_version": _FINALIZATION_STATE_SCHEMA_VERSION,
            "state": self.status,
            "candidate_pointer": self.candidate.payload(),
            "previous_pointer": (
                None if self.previous is None else self.previous.payload()
            ),
        }

    def completed(self) -> _FinalizationState:
        """Return the corresponding completed publication state.

        Returns
        -------
        _FinalizationState
            State with the same exact pointer bindings and ``complete`` status.
        """
        return _FinalizationState(_PUBLICATION_COMPLETE, self.candidate, self.previous)


@contextmanager
def _finalization_lock(
    root_descriptor: int,
    *,
    chain: RetainedDirectoryChain | None = None,
) -> Iterator[None]:
    """Serialize every current-pointer finalization on the retained root inode.

    Parameters
    ----------
    root_descriptor : int
        Retained no-follow artifact-root descriptor.
    chain : RetainedDirectoryChain or None, optional
        Complete visible artifact-root chain.
    Yields
    ------
    None
    """
    if os.name == "nt":
        raise RuntimeError("secure artifact finalization requires Linux")
    import fcntl

    chain = chain or _ACTIVE_HISTORY_CHAIN.get()
    if chain is not None:
        chain.verify()
    fcntl.flock(root_descriptor, fcntl.LOCK_EX)
    try:
        if chain is not None:
            chain.verify()
        yield
    finally:
        if chain is not None:
            chain.verify()
        fcntl.flock(root_descriptor, fcntl.LOCK_UN)
        if chain is not None:
            chain.verify()


def _history_paths(artifact_root: str | Path) -> tuple[Path, Path]:
    """Resolve a history root while refusing destructive symlink traversal.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Configured history root.

    Returns
    -------
    tuple of pathlib.Path
        Artifact root and its runs directory.
    """
    candidate = Path(artifact_root).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    root = Path(os.path.abspath(candidate))
    runs_root = root / RUNS_DIRECTORY
    if root.is_symlink() or runs_root.is_symlink():
        raise ValueError(
            "artifact history root and runs directory must not be symlinks"
        )
    return root, runs_root


def create_run_artifact_dir(
    artifact_root: str | Path,
    run_config: Mapping[str, Any],
    *,
    public_artifact_dir: str | Path | None = None,
    client_shard: Mapping[str, Any] | None = None,
    app_manifest: AppManifest | None = None,
    flower_run_id: int | None = None,
) -> Path:
    """Create one immutable run namespace and its provenance manifest.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Root containing versioned run directories.
    run_config : mapping of str to Any
        Complete training configuration.
    public_artifact_dir : str or pathlib.Path or None, optional
        Public artifacts used by this run.
    client_shard : mapping of str to Any or None, optional
        Validated private-shard evidence for a local-training run.
    app_manifest : AppManifest or None, optional
        Already validated public snapshot retained by this operation.
    flower_run_id : int or None, optional
        Flower infrastructure run identifier.

    Returns
    -------
    pathlib.Path
        New ``runs/<run_id>`` directory.
    """
    root, runs_root = _history_paths(artifact_root)
    require_secure_artifact_platform()
    run_id = str(uuid.uuid4())
    run_dir = runs_root / run_id
    chain = RetainedDirectoryChain.open(
        run_dir,
        create=True,
        error_message="artifact history changed during creation",
        check_platform=False,
    )
    with chain:
        try:
            chain.verify()
            write_run_provenance_manifest(
                run_dir,
                run_config,
                public_artifact_dir=public_artifact_dir,
                client_shard=client_shard,
                app_manifest=app_manifest,
                flower_run_id=flower_run_id,
                run_id=run_id,
            )
            chain.verify()
        except BaseException:
            try:
                chain.remove_created_target()
            except (FileNotFoundError, OSError, ValueError):
                pass
            raise
        chain.commit()
        chain.verify()
        return run_dir


def _validate_current_index(document: bytes, *, source: str) -> Mapping[str, Any]:
    """Validate exact current-run index bytes.

    Parameters
    ----------
    document : bytes
        Exact current pointer bytes.
    source : str
        Diagnostic source label.

    Returns
    -------
    collections.abc.Mapping
        Validated current-run index.
    """
    payload = strict_json_loads(document, source=source)
    if not isinstance(payload, Mapping):
        raise ValueError("current-run index must be a JSON object")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int:
        raise ValueError("current-run index has an invalid schema_version")
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("current-run index has an unsupported schema_version")
    run_id = payload.get("run_id")
    checksum = payload.get("artifact_manifest_checksum")
    try:
        parsed_run_id = uuid.UUID(run_id) if isinstance(run_id, str) else None
    except ValueError as error:
        raise ValueError("current-run index has an invalid run_id") from error
    if (
        parsed_run_id is None
        or parsed_run_id.version != 4
        or str(parsed_run_id) != run_id
    ):
        raise ValueError("current-run index has an invalid run_id")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 71
        or not checksum.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in checksum[7:])
    ):
        raise ValueError("current-run index has an invalid artifact manifest checksum")
    return payload


def _load_current_index(artifact_root: str | Path) -> Mapping[str, Any] | None:
    """Load and validate the atomic current-run index when present.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Root containing ``current.json``.

    Returns
    -------
    collections.abc.Mapping or None
        Validated index, or ``None`` for a legacy flat directory.
    """
    root, _ = _history_paths(artifact_root)
    path = root / CURRENT_RUN_FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    try:
        document = read_regular_file(path, parent=root)
    except ValueError as error:
        raise ValueError(f"invalid current-run index: {path}") from error
    return _validate_current_index(document, source=f"current-run index: {path}")


def _load_current_index_at(
    root: Path, root_descriptor: int
) -> tuple[Mapping[str, Any], bytes] | None:
    """Load the current index through a retained artifact-root descriptor.

    Parameters
    ----------
    root : pathlib.Path
        Canonical artifact root used for diagnostics.
    root_descriptor : int
        Retained artifact-root descriptor.

    Returns
    -------
    tuple of mapping and bytes or None
        Validated payload and exact bytes, or ``None`` when absent.
    """
    try:
        retained = _open_retained_file(root_descriptor, CURRENT_RUN_FILENAME)
    except ValueError as error:
        try:
            os.stat(
                CURRENT_RUN_FILENAME,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        raise ValueError(
            f"invalid current-run index: {root / CURRENT_RUN_FILENAME}"
        ) from error
    try:
        document = retained.snapshot.content
    finally:
        os.close(retained.descriptor)
    return (
        _validate_current_index(
            document, source=f"current-run index: {root / CURRENT_RUN_FILENAME}"
        ),
        document,
    )


def _pointer_binding(
    current_record: tuple[Mapping[str, Any], bytes],
) -> _PointerBinding:
    """Bind a validated current-pointer record to its exact bytes.

    Parameters
    ----------
    current_record : tuple of mapping and bytes
        Validated current-pointer identity and exact serialized bytes.

    Returns
    -------
    _PointerBinding
        Exact pointer binding suitable for durable finalization state.
    """
    current, content = current_record
    return _PointerBinding(
        str(current["run_id"]),
        str(current["artifact_manifest_checksum"]),
        content,
    )


def _parse_pointer_binding(payload: object) -> _PointerBinding:
    """Validate one exact-schema pointer binding from finalization state.

    Parameters
    ----------
    payload : object
        Decoded candidate or previous pointer binding.

    Returns
    -------
    _PointerBinding
        Validated identity, checksums, and exact pointer bytes.

    Raises
    ------
    ValueError
        If the binding schema, encoding, checksum, or identity is invalid.
    """
    expected_keys = {
        "run_id",
        "artifact_manifest_checksum",
        "bytes_base64",
        "bytes_checksum",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise ValueError("run finalization state is unsafe")
    encoded = payload["bytes_base64"]
    if not isinstance(encoded, str):
        raise ValueError("run finalization state is unsafe")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("run finalization state is unsafe") from error
    if base64.b64encode(content).decode("ascii") != encoded:
        raise ValueError("run finalization state is unsafe")
    current = _validate_current_index(content, source="run finalization state pointer")
    if (
        payload["run_id"] != current["run_id"]
        or payload["artifact_manifest_checksum"]
        != current["artifact_manifest_checksum"]
        or payload["bytes_checksum"] != sha256_bytes(content)
    ):
        raise ValueError("run finalization state is unsafe")
    return _PointerBinding(
        str(current["run_id"]),
        str(current["artifact_manifest_checksum"]),
        content,
    )


def _parse_finalization_state(document: bytes) -> _FinalizationState:
    """Parse strict canonical finalization state with its exact schema.

    Parameters
    ----------
    document : bytes
        Exact state-file bytes.

    Returns
    -------
    _FinalizationState
        Validated durable publication state.

    Raises
    ------
    ValueError
        If syntax, canonical encoding, schema, or pointer bindings are invalid.
    """
    payload = strict_json_loads(document, source="run finalization state")
    expected_keys = {
        "schema_version",
        "state",
        "candidate_pointer",
        "previous_pointer",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_keys
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != _FINALIZATION_STATE_SCHEMA_VERSION
        or not isinstance(payload["state"], str)
        or payload["state"] not in {_PUBLICATION_PENDING, _PUBLICATION_COMPLETE}
        or canonical_json_bytes(payload) != document
    ):
        raise ValueError("run finalization state is unsafe")
    previous_payload = payload["previous_pointer"]
    previous = (
        None if previous_payload is None else _parse_pointer_binding(previous_payload)
    )
    return _FinalizationState(
        str(payload["state"]),
        _parse_pointer_binding(payload["candidate_pointer"]),
        previous,
    )


def _finalization_state_name(run_id: str) -> str:
    """Return the direct-child state filename for one canonical run UUID.

    Parameters
    ----------
    run_id : str
        Canonical run identity.

    Returns
    -------
    str
        Per-run finalization-state filename.
    """
    return f".{run_id}.finalize.state"


def _finalization_state_run_id(name: str) -> str | None:
    """Return the canonical UUID4 encoded by a finalization-state filename.

    Parameters
    ----------
    name : str
        Direct artifact-root entry name.

    Returns
    -------
    str or None
        Canonical run identity, or ``None`` for any other filename.
    """
    suffix = ".finalize.state"
    if not name.startswith(".") or not name.endswith(suffix):
        return None
    run_id = name[1 : -len(suffix)]
    try:
        parsed = uuid.UUID(run_id)
    except ValueError:
        return None
    if parsed.version != 4 or str(parsed) != run_id:
        return None
    return run_id


def _load_finalization_state_at(
    root_descriptor: int, run_id: str
) -> tuple[_FinalizationState, bytes] | None:
    """Load a no-follow, single-link, private finalization-state file.

    Parameters
    ----------
    root_descriptor : int
        Retained artifact-root descriptor.
    run_id : str
        Canonical run identity.

    Returns
    -------
    tuple of _FinalizationState and bytes or None
        Validated state and exact bytes, or ``None`` when absent.

    Raises
    ------
    ValueError
        If the state entry or serialized state is unsafe.
    """
    name = _finalization_state_name(run_id)
    try:
        retained = _open_retained_file(root_descriptor, name)
    except ValueError as error:
        try:
            os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None
        raise ValueError("run finalization state is unsafe") from error
    try:
        if stat.S_IMODE(os.fstat(retained.descriptor).st_mode) != 0o600:
            raise ValueError("run finalization state is unsafe")
        document = retained.snapshot.content
        return _parse_finalization_state(document), document
    finally:
        os.close(retained.descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    """Write all bytes to one descriptor.

    Parameters
    ----------
    descriptor : int
        Writable regular-file descriptor.
    content : bytes
        Exact bytes to write.

    Returns
    -------
    None
    """
    written = 0
    while written < len(content):
        count = os.write(descriptor, content[written:])
        if count == 0:
            raise OSError("artifact write made no progress")
        written += count


def _replace_bytes_at(
    root_descriptor: int,
    name: str,
    content: bytes,
    *,
    mode: int = 0o644,
    retain: bool = False,
    chain: RetainedDirectoryChain | None = None,
) -> _RetainedFile | None:
    """Atomically replace one file descriptor-relatively.

    Parameters
    ----------
    root_descriptor : int
        Retained owning directory descriptor.
    name : str
        Direct child destination name.
    content : bytes
        Exact replacement bytes.
    mode : int, optional
        Exact permissions installed on the replacement file.
    retain : bool, optional
        Keep the installed file descriptor open for a later commit check.
    chain : RetainedDirectoryChain or None, optional
        Complete visible owner chain to verify around every mutation.

    Returns
    -------
    _RetainedFile or None
        Retained installed file when requested, otherwise ``None``.
    """

    chain = chain or _ACTIVE_HISTORY_CHAIN.get()

    def verify_chain() -> None:
        if chain is not None:
            chain.verify()

    descriptor: int | None = None
    temporary: str | None = None
    for _ in range(_TEMPORARY_NAME_ATTEMPTS):
        candidate = f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            verify_chain()
            descriptor = os.open(
                candidate,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode,
                dir_fd=root_descriptor,
            )
            verify_chain()
        except FileExistsError:
            continue
        temporary = candidate
        break
    if descriptor is None or temporary is None:
        raise FileExistsError("could not allocate an exclusive temporary file")
    retained: _RetainedFile | None = None
    try:
        verify_chain()
        _write_all(descriptor, content)
        verify_chain()
        os.fchmod(descriptor, mode)
        verify_chain()
        os.fsync(descriptor)
        verify_chain()
        captured, file_stat = _read_descriptor_bytes(descriptor)
        if captured != content:
            raise ValueError("atomic replacement changed during write")
        retained = _RetainedFile(
            descriptor,
            RegularFileSnapshot(
                captured,
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_size,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
            ),
        )
        _require_retained_file_entry(root_descriptor, temporary, retained, content)
        verify_chain()
        os.rename(
            temporary,
            name,
            src_dir_fd=root_descriptor,
            dst_dir_fd=root_descriptor,
        )
        verify_chain()
        temporary = None
        _require_retained_file_entry(root_descriptor, name, retained, content)
        if retain:
            descriptor = None
            return retained
        os.close(descriptor)
        descriptor = None
        return None
    except BaseException:
        if temporary is not None:
            try:
                verify_chain()
                if retained is not None:
                    _unlink_retained_file_entry(
                        root_descriptor, temporary, retained, content
                    )
                elif descriptor is not None:
                    temporary_stat = os.stat(
                        temporary,
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                    owned_stat = os.fstat(descriptor)
                    if (temporary_stat.st_dev, temporary_stat.st_ino) == (
                        owned_stat.st_dev,
                        owned_stat.st_ino,
                    ):
                        os.unlink(temporary, dir_fd=root_descriptor)
                verify_chain()
            except (FileNotFoundError, ValueError):
                pass
        if descriptor is not None:
            os.close(descriptor)
        raise


def _record_finalization_state(
    root: _RetainedDirectory,
    run_id: str,
    state: _FinalizationState,
    *,
    chain: RetainedDirectoryChain | None = None,
) -> tuple[_FinalizationState, bytes]:
    """Atomically replace and durably flush one finalization-state transition.

    Parameters
    ----------
    root : _RetainedDirectory
        Retained artifact root that owns the state file.
    run_id : str
        Canonical run identity.
    state : _FinalizationState
        Exact pending or complete state to persist.
    chain : RetainedDirectoryChain or None, optional
        Complete visible artifact-root chain.

    Returns
    -------
    tuple of _FinalizationState and bytes
        Persisted state and exact canonical bytes.

    Raises
    ------
    ValueError
        If existing or newly persisted state is unsafe.
    OSError
        If writing or a durability barrier fails.
    """
    chain = chain or _ACTIVE_HISTORY_CHAIN.get()
    name = _finalization_state_name(run_id)
    if chain is not None:
        chain.verify()
    previous_record = _load_finalization_state_at(root.descriptor, run_id)
    document = canonical_json_bytes(state.payload())
    installed = _replace_bytes_at(
        root.descriptor,
        name,
        document,
        mode=0o600,
        retain=True,
        chain=chain,
    )
    assert installed is not None
    try:
        if chain is None:
            _sync_retained_directory(root)
        else:
            _sync_visible_directory(root, chain)
        _require_retained_file_entry(root.descriptor, name, installed, document)
        persisted = _load_finalization_state_at(root.descriptor, run_id)
        if persisted != (state, document):
            raise ValueError("run finalization state changed during finalization")
        if chain is not None:
            chain.verify()
    except BaseException:
        if chain is not None:
            try:
                chain.verify()
            except ValueError:
                raise
        if _retained_file_entry_matches(root.descriptor, name, installed, document):
            if previous_record is None:
                _unlink_retained_file_entry(root.descriptor, name, installed, document)
            else:
                _replace_bytes_at(
                    root.descriptor,
                    name,
                    previous_record[1],
                    mode=0o600,
                    chain=chain,
                )
            if chain is None:
                _sync_retained_directory(root)
            else:
                _sync_visible_directory(root, chain)
        raise
    finally:
        os.close(installed.descriptor)
    return state, document


def _rollback_current_index(
    descriptors: _HistoryDescriptors,
    previous_bytes: bytes | None,
    installed: _RetainedFile,
    candidate_bytes: bytes,
) -> None:
    """Restore the pre-publication pointer after an identity failure.

    Parameters
    ----------
    descriptors : _HistoryDescriptors
        Retained history descriptor chain.
    previous_bytes : bytes or None
        Exact prior pointer, or ``None`` when publication created it.
    installed : _RetainedFile
        Retained candidate pointer installed by this publication.
    candidate_bytes : bytes
        Exact candidate pointer bytes.

    Returns
    -------
    None
    """
    try:
        _verify_history_descriptors(descriptors)
    except ValueError:
        return
    if not _retained_file_entry_matches(
        descriptors.root.descriptor,
        CURRENT_RUN_FILENAME,
        installed,
        candidate_bytes,
    ):
        return
    if previous_bytes is None:
        _verify_history_descriptors(descriptors)
        _unlink_retained_file_entry(
            descriptors.root.descriptor,
            CURRENT_RUN_FILENAME,
            installed,
            candidate_bytes,
        )
        _verify_history_descriptors(descriptors)
    else:
        _replace_bytes_at(
            descriptors.root.descriptor,
            CURRENT_RUN_FILENAME,
            previous_bytes,
            chain=descriptors.chain,
        )
    _sync_visible_directory(descriptors.root, descriptors.chain)
    _verify_history_descriptors(descriptors)


def _write_current_index_at(
    root: Path,
    descriptors: _HistoryDescriptors,
    index: Mapping[str, Any],
    previous_bytes: bytes | None,
) -> tuple[Path, _RetainedFile]:
    """Publish and durably flush ``current.json`` under the retained root.

    Parameters
    ----------
    root : pathlib.Path
        Canonical artifact root used for the returned path.
    descriptors : _HistoryDescriptors
        Retained history descriptor chain.
    index : mapping of str to Any
        Valid current-run index payload.
    previous_bytes : bytes or None
        Exact prior pointer for fail-closed rollback.

    Returns
    -------
    tuple of pathlib.Path and _RetainedFile
        Published path and retained installed candidate pointer.
    """
    _verify_history_descriptors(descriptors)
    candidate_bytes = canonical_json_bytes(index)
    installed = _replace_bytes_at(
        descriptors.root.descriptor,
        CURRENT_RUN_FILENAME,
        candidate_bytes,
        retain=True,
        chain=descriptors.chain,
    )
    assert installed is not None
    try:
        _verify_history_descriptors(descriptors)
        _require_retained_file_entry(
            descriptors.root.descriptor,
            CURRENT_RUN_FILENAME,
            installed,
            candidate_bytes,
        )
    except BaseException:
        _rollback_current_index(descriptors, previous_bytes, installed, candidate_bytes)
        os.close(installed.descriptor)
        raise
    try:
        _sync_visible_directory(descriptors.root, descriptors.chain)
    except BaseException:
        os.close(installed.descriptor)
        raise
    try:
        _verify_history_descriptors(descriptors)
        _require_retained_file_entry(
            descriptors.root.descriptor,
            CURRENT_RUN_FILENAME,
            installed,
            candidate_bytes,
        )
        reopened = _load_current_index_at(root, descriptors.root.descriptor)
        if reopened is None or reopened[1] != candidate_bytes:
            raise ValueError("current-run index changed during finalization")
    except BaseException:
        _rollback_current_index(descriptors, previous_bytes, installed, candidate_bytes)
        os.close(installed.descriptor)
        raise
    return root / CURRENT_RUN_FILENAME, installed


def _commit_complete_state(
    root: Path,
    descriptors: _HistoryDescriptors,
    run_id: str,
    state: _FinalizationState,
    current: _RetainedFile,
    completed_inventory: Mapping[str, _RetainedFile],
    expected_content: Mapping[str, bytes],
) -> None:
    """Commit complete only while the candidate remains visibly installed.

    Parameters
    ----------
    root : pathlib.Path
        Canonical artifact root used for pointer validation diagnostics.
    descriptors : _HistoryDescriptors
        Retained root-to-run descriptor chain.
    run_id : str
        Canonical run identity.
    state : _FinalizationState
        Exact pending publication state.
    current : _RetainedFile
        Retained installed candidate pointer.
    completed_inventory : mapping of str to _RetainedFile
        Retained completed-run inventory.
    expected_content : mapping of str to bytes
        Exact completed-run bytes.

    Returns
    -------
    None
    """

    def verify_commit_point() -> None:
        _verify_history_descriptors(descriptors)
        _verify_run_inventory(
            descriptors.run.descriptor, completed_inventory, expected_content
        )
        _require_retained_file_entry(
            descriptors.root.descriptor,
            CURRENT_RUN_FILENAME,
            current,
            state.candidate.content,
        )
        reopened = _load_current_index_at(root, descriptors.root.descriptor)
        if reopened is None or reopened[1] != state.candidate.content:
            raise ValueError("current-run index changed during finalization")

    verify_commit_point()
    completed = False
    try:
        _record_finalization_state(
            descriptors.root,
            run_id,
            state.completed(),
        )
        completed = True
        verify_commit_point()
    except BaseException:
        if completed:
            token = _ACTIVE_HISTORY_CHAIN.set(None)
            try:
                _record_finalization_state(descriptors.root, run_id, state)
            finally:
                _ACTIVE_HISTORY_CHAIN.reset(token)
        raise


def resolve_current_run_dir(artifact_root: str | Path) -> Path:
    """Resolve the selected completed run, with legacy flat-layout fallback.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Root containing run history or legacy flat artifacts.

    Returns
    -------
    pathlib.Path
        Current versioned run directory, or the root for legacy artifacts.
    """
    root, _ = _history_paths(artifact_root)
    if _load_current_index(root) is None:
        return root
    return load_current_run_snapshot(root).directory


def _retain_public_artifacts(run_dir: Path, app_manifest: AppManifest) -> None:
    """Publish immutable public evidence into one run before final capture.

    Parameters
    ----------
    run_dir : pathlib.Path
        Canonical in-progress run directory.
    app_manifest : AppManifest
        Validated public snapshot used by the run.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an existing retained artifact is unsafe or differs from the snapshot.
    """
    for name, content in (
        ("manifest.json", app_manifest.manifest_bytes),
        ("vocab.txt", app_manifest.vocabulary_bytes),
    ):
        path = run_dir / name
        try:
            write_bytes_atomically(path, content, overwrite=False)
        except FileExistsError:
            pass
        retained = read_regular_file_snapshot(path, parent=run_dir)
        if retained.content != content:
            raise ValueError(f"retained public artifact differs from snapshot: {name}")


def load_current_run_snapshot(
    artifact_root: str | Path, *, app_manifest: AppManifest | None = None
) -> ServerArtifactSnapshot:
    """Read the selected run into a checksum-verified immutable snapshot.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Root containing run history or legacy flat artifacts.
    app_manifest : AppManifest or None, optional
        Configured public snapshot that the selected server artifact must match.

    Returns
    -------
    ServerArtifactSnapshot
        Exact artifact bytes validated against the selected manifest.
    """
    root, runs_root = _history_paths(artifact_root)
    index = _load_current_index(root)
    if index is None:
        return load_server_artifact_snapshot(root, app_manifest=app_manifest)
    run_dir = runs_root / str(index["run_id"])
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"current run directory is missing or unsafe: {run_dir}")
    canonical_run_dir = run_dir.resolve(strict=True)
    if canonical_run_dir.parent != runs_root.resolve(strict=True):
        raise ValueError("current run directory escapes the artifact runs root")
    manifest_path = run_dir / SERVER_ARTIFACT_MANIFEST_FILENAME
    try:
        manifest_bytes = read_regular_file(manifest_path, parent=canonical_run_dir)
    except ValueError as error:
        raise ValueError(
            f"current run artifact manifest is missing or unsafe: {manifest_path}"
        ) from error
    if not hmac.compare_digest(
        sha256_bytes(manifest_bytes), index["artifact_manifest_checksum"]
    ):
        raise ValueError("current-run artifact manifest checksum does not match")
    snapshot = load_server_artifact_snapshot(
        canonical_run_dir,
        manifest_bytes=manifest_bytes,
        app_manifest=app_manifest,
    )
    return snapshot


def publish_completed_run(
    artifact_root: str | Path,
    run_dir: str | Path,
    *,
    app_manifest: AppManifest,
) -> Path:
    """Finalize checksums and atomically select one completed run.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Root containing run history.
    run_dir : str or pathlib.Path
        Completed run directory below the root.
    app_manifest : AppManifest
        Already validated public snapshot used throughout the run.

    Returns
    -------
    pathlib.Path
        Written ``current.json`` path.
    """
    root, runs_root = _history_paths(artifact_root)
    candidate_run_dir = Path(run_dir).expanduser()
    if not candidate_run_dir.is_absolute():
        candidate_run_dir = Path.cwd() / candidate_run_dir
    resolved_run_dir = Path(os.path.abspath(candidate_run_dir))
    if resolved_run_dir.parent != runs_root:
        raise ValueError("run directory must be directly below the artifact runs root")
    with _retain_history_descriptors(root, runs_root, resolved_run_dir) as descriptors:
        with _finalization_lock(descriptors.root.descriptor):
            _verify_history_descriptors(descriptors)

            from src.app_manifest import validate_app_manifest_bytes

            validated_manifest = validate_app_manifest_bytes(
                app_manifest.manifest_bytes,
                app_manifest.vocabulary_bytes,
                vocabulary_path=Path("vocab.txt"),
            )
            try:
                os.stat(
                    SERVER_ARTIFACT_MANIFEST_FILENAME,
                    dir_fd=descriptors.run.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                existing_manifest_bytes = None
                existing_snapshot = None
            else:
                retained_existing_manifest = _open_retained_file(
                    descriptors.run.descriptor,
                    SERVER_ARTIFACT_MANIFEST_FILENAME,
                )
                try:
                    existing_manifest_bytes = (
                        retained_existing_manifest.snapshot.content
                    )
                finally:
                    os.close(retained_existing_manifest.descriptor)
                existing_snapshot = load_server_artifact_snapshot(
                    resolved_run_dir,
                    manifest_bytes=existing_manifest_bytes,
                    app_manifest=validated_manifest,
                )
            was_complete = (
                existing_snapshot is not None
                and existing_snapshot.manifest.get("lifecycle") == "complete"
            )
            if not was_complete:
                _verify_history_descriptors(descriptors)
                _retain_public_artifacts(resolved_run_dir, validated_manifest)
                _verify_history_descriptors(descriptors)

            try:
                inventory_context = _capture_run_inventory(
                    descriptors.run.descriptor, exclude_manifest=True
                )
                with inventory_context as artifact_inventory:
                    artifact_snapshot = _snapshot_from_inventory(artifact_inventory)
                    provenance_path = run_manifest_path(resolved_run_dir)
                    try:
                        provenance_bytes = artifact_snapshot.files[
                            "run_manifest.json"
                        ].content
                    except KeyError as error:
                        raise ValueError(
                            "server artifact is missing or unsafe"
                        ) from error
                    provenance = load_run_provenance_manifest(
                        provenance_path, manifest_bytes=provenance_bytes
                    )
                    if provenance["run_id"] != resolved_run_dir.name:
                        raise ValueError(
                            "run directory does not match its provenance run_id"
                        )
                    _verify_history_descriptors(descriptors)
                    _sync_retained_files(artifact_inventory)
                    _verify_history_descriptors(descriptors)

                    if existing_manifest_bytes is not None:
                        snapshot = load_server_artifact_snapshot(
                            resolved_run_dir,
                            manifest_bytes=existing_manifest_bytes,
                            app_manifest=validated_manifest,
                        )
                        validate_run_provenance_evidence(
                            provenance, snapshot.manifest, validated_manifest
                        )
                        if snapshot.manifest.get("lifecycle") != "complete":
                            _verify_history_descriptors(descriptors)
                            write_server_artifact_manifest(
                                resolved_run_dir,
                                app_manifest=validated_manifest,
                                finalized=True,
                                artifact_snapshot=artifact_snapshot,
                            )
                            _verify_history_descriptors(descriptors)
                        elif dict(snapshot.files) != {
                            name: retained.snapshot.content
                            for name, retained in artifact_inventory.items()
                        }:
                            raise ValueError(
                                "completed run differs from its retained snapshot"
                            )
                    else:
                        _verify_history_descriptors(descriptors)
                        write_server_artifact_manifest(
                            resolved_run_dir,
                            app_manifest=validated_manifest,
                            finalized=True,
                            artifact_snapshot=artifact_snapshot,
                        )
                        _verify_history_descriptors(descriptors)
                    _verify_history_descriptors(descriptors)
                    retained_manifest = _open_retained_file(
                        descriptors.run.descriptor,
                        SERVER_ARTIFACT_MANIFEST_FILENAME,
                    )
                    try:
                        manifest_bytes = retained_manifest.snapshot.content
                        snapshot = load_server_artifact_snapshot(
                            resolved_run_dir,
                            manifest_bytes=manifest_bytes,
                            app_manifest=validated_manifest,
                        )
                        validate_run_provenance_evidence(
                            provenance, snapshot.manifest, validated_manifest
                        )
                        expected_content = {
                            SERVER_ARTIFACT_MANIFEST_FILENAME: manifest_bytes,
                            **dict(snapshot.files),
                        }
                        completed_inventory = MappingProxyType(
                            {
                                **dict(artifact_inventory),
                                SERVER_ARTIFACT_MANIFEST_FILENAME: retained_manifest,
                            }
                        )
                        _verify_run_inventory(
                            descriptors.run.descriptor,
                            completed_inventory,
                            expected_content,
                        )
                        checksum = sha256_bytes(manifest_bytes)
                        index = {
                            "schema_version": ARTIFACT_SCHEMA_VERSION,
                            "run_id": resolved_run_dir.name,
                            "artifact_manifest_checksum": checksum,
                        }
                        candidate = _PointerBinding(
                            resolved_run_dir.name,
                            checksum,
                            canonical_json_bytes(index),
                        )
                        state_record = _load_finalization_state_at(
                            descriptors.root.descriptor, resolved_run_dir.name
                        )
                        if state_record is not None:
                            state = state_record[0]
                            if state.status == _PUBLICATION_COMPLETE:
                                raise ValueError(
                                    "completed run cannot be finalized again"
                                )
                            if state.candidate != candidate:
                                raise ValueError("run finalization state is unsafe")
                        else:
                            state = None

                        current_record = _load_current_index_at(
                            root, descriptors.root.descriptor
                        )
                        current_bytes = (
                            None if current_record is None else current_record[1]
                        )
                        if state is None:
                            if current_bytes == candidate.content:
                                raise ValueError(
                                    "completed run cannot be finalized again"
                                )
                            previous = (
                                None
                                if current_record is None
                                else _pointer_binding(current_record)
                            )
                            state = _FinalizationState(
                                _PUBLICATION_PENDING, candidate, previous
                            )
                            _record_finalization_state(
                                descriptors.root,
                                resolved_run_dir.name,
                                state,
                            )
                        elif current_bytes not in {
                            candidate.content,
                            None if state.previous is None else state.previous.content,
                        }:
                            raise ValueError("completed run cannot be finalized again")
                        _sync_retained_files(completed_inventory)
                        _verify_history_descriptors(descriptors)
                        _sync_visible_directory(descriptors.run, descriptors.chain)
                        _sync_visible_directory(descriptors.runs, descriptors.chain)
                        _verify_history_descriptors(descriptors)
                        _verify_run_inventory(
                            descriptors.run.descriptor,
                            completed_inventory,
                            expected_content,
                        )

                        current_record = _load_current_index_at(
                            root, descriptors.root.descriptor
                        )
                        current_bytes = (
                            None if current_record is None else current_record[1]
                        )
                        previous_bytes = (
                            None if state.previous is None else state.previous.content
                        )
                        if current_bytes not in {candidate.content, previous_bytes}:
                            raise ValueError(
                                "current-run index changed during finalization"
                            )
                        if current_bytes == candidate.content:
                            _sync_visible_directory(descriptors.root, descriptors.chain)
                            recovered_pointer = _open_retained_file(
                                descriptors.root.descriptor, CURRENT_RUN_FILENAME
                            )
                            try:
                                _commit_complete_state(
                                    root,
                                    descriptors,
                                    resolved_run_dir.name,
                                    state,
                                    recovered_pointer,
                                    completed_inventory,
                                    expected_content,
                                )
                            finally:
                                os.close(recovered_pointer.descriptor)
                            return root / CURRENT_RUN_FILENAME

                        _verify_history_descriptors(descriptors)
                        _verify_run_inventory(
                            descriptors.run.descriptor,
                            completed_inventory,
                            expected_content,
                        )
                        pointer, installed_pointer = _write_current_index_at(
                            root, descriptors, index, previous_bytes
                        )
                        try:
                            _verify_history_descriptors(descriptors)
                            _verify_run_inventory(
                                descriptors.run.descriptor,
                                completed_inventory,
                                expected_content,
                            )
                            _commit_complete_state(
                                root,
                                descriptors,
                                resolved_run_dir.name,
                                state,
                                installed_pointer,
                                completed_inventory,
                                expected_content,
                            )
                        except ValueError:
                            _rollback_current_index(
                                descriptors,
                                previous_bytes,
                                installed_pointer,
                                candidate.content,
                            )
                            raise
                        finally:
                            os.close(installed_pointer.descriptor)
                        return pointer
                    finally:
                        os.close(retained_manifest.descriptor)
            except FileNotFoundError as error:
                raise ValueError("server artifact is missing or unsafe") from error


def _state_allows_prune(
    root_descriptor: int, run_id: str, current_bytes: bytes
) -> tuple[bool, bool]:
    """Classify whether a run and its finalization state may be pruned.

    Parameters
    ----------
    root_descriptor : int
        Retained artifact-root descriptor.
    run_id : str
        Canonical candidate run identity.
    current_bytes : bytes
        Exact current-pointer bytes.

    Returns
    -------
    tuple of bool and bool
        Whether the run may be pruned and whether its state must then be removed.
    """
    try:
        state_record = _load_finalization_state_at(root_descriptor, run_id)
    except ValueError:
        return False, False
    if state_record is None:
        return True, False
    state = state_record[0]
    if state.status == _PUBLICATION_COMPLETE:
        return True, True
    previous_bytes = None if state.previous is None else state.previous.content
    if current_bytes in {state.candidate.content, previous_bytes}:
        return False, False
    return True, True


def _remove_finalization_state(
    root: _RetainedDirectory,
    run_id: str,
    current_bytes: bytes | None,
    *,
    chain: RetainedDirectoryChain | None = None,
) -> None:
    """Durably remove one proven safe state while preserving it on failure.

    Parameters
    ----------
    root : _RetainedDirectory
        Retained artifact root.
    run_id : str
        Canonical pruned run identity.
    current_bytes : bytes or None
        Exact current pointer used to prove a pending state obsolete, or no pointer.
    chain : RetainedDirectoryChain or None, optional
        Complete visible artifact-root chain.

    Returns
    -------
    None
    """
    chain = chain or _ACTIVE_HISTORY_CHAIN.get()
    if chain is not None:
        chain.verify()
    record = _load_finalization_state_at(root.descriptor, run_id)
    if record is None:
        return
    state, document = record
    previous_bytes = None if state.previous is None else state.previous.content
    if state.status != _PUBLICATION_COMPLETE and current_bytes in {
        state.candidate.content,
        previous_bytes,
    }:
        raise ValueError("pending run finalization state cannot be pruned")
    name = _finalization_state_name(run_id)
    retained = _open_retained_file(root.descriptor, name)
    removed = False
    try:
        if chain is not None:
            chain.verify()
        if retained.snapshot.content != document or not _unlink_retained_file_entry(
            root.descriptor, name, retained, document
        ):
            raise ValueError("run finalization state changed during pruning")
        removed = True
        if chain is None:
            _sync_retained_directory(root)
        else:
            _sync_visible_directory(root, chain)
    except BaseException:
        if chain is not None:
            try:
                chain.verify()
            except ValueError:
                raise
        if removed:
            try:
                os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
            except FileNotFoundError:
                _replace_bytes_at(
                    root.descriptor,
                    name,
                    document,
                    mode=0o600,
                    chain=chain,
                )
                try:
                    if chain is None:
                        _sync_retained_directory(root)
                    else:
                        _sync_visible_directory(root, chain)
                except OSError:
                    pass
        raise
    finally:
        os.close(retained.descriptor)


def _recover_absent_run_states(
    root: Path,
    descriptors: _HistoryRootDescriptors,
    current_bytes: bytes | None,
) -> None:
    """Durably remove safe finalization state left after run deletion.

    Parameters
    ----------
    root : pathlib.Path
        Canonical artifact root used for pointer validation diagnostics.
    descriptors : _HistoryRootDescriptors
        Retained artifact root and runs directory chain.
    current_bytes : bytes or None
        Exact current-pointer bytes, or ``None`` when no pointer exists.

    Returns
    -------
    None
    """
    removable: list[str] = []
    for name in os.listdir(descriptors.root.descriptor):
        run_id = _finalization_state_run_id(name)
        if run_id is None:
            continue
        try:
            os.stat(
                run_id,
                dir_fd=descriptors.runs.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            continue
        try:
            record = _load_finalization_state_at(descriptors.root.descriptor, run_id)
        except ValueError:
            continue
        if record is None or record[0].candidate.run_id != run_id:
            continue
        state = record[0]
        previous_bytes = None if state.previous is None else state.previous.content
        if state.status == _PUBLICATION_PENDING and current_bytes in {
            state.candidate.content,
            previous_bytes,
        }:
            continue
        removable.append(run_id)

    if not removable:
        return
    _sync_visible_directory(descriptors.runs, descriptors.chain)
    for run_id in removable:
        _verify_history_root(descriptors)
        try:
            os.stat(
                run_id,
                dir_fd=descriptors.runs.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            continue
        current = _load_current_index_at(root, descriptors.root.descriptor)
        if (None if current is None else current[1]) != current_bytes:
            raise ValueError("current-run index changed during pruning")
        _remove_finalization_state(
            descriptors.root,
            run_id,
            current_bytes,
        )


def prune_run_history(
    artifact_root: str | Path,
    retention_runs: int,
    *,
    active_run_dir: str | Path | None = None,
) -> list[Path]:
    """Delete oldest validated runs while preserving active and current runs.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Root containing versioned run directories.
    retention_runs : int
        Maximum validated run count after no run is active.
    active_run_dir : str or pathlib.Path or None, optional
        In-progress run that must not be removed.

    Returns
    -------
    list of pathlib.Path
        Deleted run directories in deterministic oldest-first order.
    """
    if type(retention_runs) is not int or retention_runs < 1:
        raise ValueError("artifact retention must be a positive integer")
    root, runs_root = _history_paths(artifact_root)
    if not runs_root.exists():
        return []

    with _retain_history_root(root, runs_root) as descriptors:
        with _finalization_lock(descriptors.root.descriptor):
            _verify_history_root(descriptors)
            current_record = _load_current_index_at(root, descriptors.root.descriptor)
            _recover_absent_run_states(
                root,
                descriptors,
                None if current_record is None else current_record[1],
            )
            if current_record is None:
                return []
            current_bytes = current_record[1]
            protected = {load_current_run_snapshot(root).directory}
            if active_run_dir is not None:
                active_path = Path(active_run_dir)
                if active_path.is_symlink():
                    raise ValueError("active run must be a regular directory")
                active = (
                    active_path.resolve(strict=True) if active_path.exists() else None
                )
                if active is not None and active.parent != runs_root.resolve(
                    strict=True
                ):
                    raise ValueError(
                        "active run must be directly below the artifact runs root"
                    )
                if active is not None:
                    protected.add(active)

            candidates: list[tuple[datetime, str, Path, bool, int, int]] = []
            for name in os.listdir(descriptors.runs.descriptor):
                try:
                    entry = os.stat(
                        name,
                        dir_fd=descriptors.runs.descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                path = runs_root / name
                if not stat.S_ISDIR(entry.st_mode):
                    continue
                try:
                    canonical_path = path.resolve(strict=True)
                    manifest_path = run_manifest_path(canonical_path)
                    manifest_bytes = read_regular_file(
                        manifest_path, parent=canonical_path
                    )
                    manifest = load_run_provenance_manifest(
                        manifest_path, manifest_bytes=manifest_bytes
                    )
                except ValueError:
                    continue
                if manifest["run_id"] != name:
                    continue
                may_prune, remove_state = _state_allows_prune(
                    descriptors.root.descriptor, name, current_bytes
                )
                if not may_prune:
                    protected.add(canonical_path)
                created_at = datetime.fromisoformat(
                    str(manifest["created_at"]).replace("Z", "+00:00")
                )
                candidates.append(
                    (
                        created_at,
                        name,
                        canonical_path,
                        remove_state,
                        entry.st_dev,
                        entry.st_ino,
                    )
                )

            candidates.sort(reverse=True)
            keep = set(protected)
            for _, _, path, _, _, _ in candidates:
                if path in keep:
                    continue
                if len(keep) < retention_runs:
                    keep.add(path)

            deleted: list[Path] = []
            for _, name, path, remove_state, device, inode in reversed(candidates):
                if path in keep:
                    continue
                _verify_history_root(descriptors)
                current = _load_current_index_at(root, descriptors.root.descriptor)
                if current is None or current[1] != current_bytes:
                    raise ValueError("current-run index changed during pruning")
                entry = os.stat(
                    name,
                    dir_fd=descriptors.runs.descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(entry.st_mode) or (
                    entry.st_dev,
                    entry.st_ino,
                ) != (device, inode):
                    raise ValueError("run history changed during pruning")
                _verify_history_root(descriptors)
                shutil.rmtree(name, dir_fd=descriptors.runs.descriptor)
                _sync_visible_directory(descriptors.runs, descriptors.chain)
                if remove_state:
                    _remove_finalization_state(
                        descriptors.root,
                        name,
                        current_bytes,
                    )
                _verify_history_root(descriptors)
                deleted.append(path)
            return deleted
