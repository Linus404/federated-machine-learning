"""Manage immutable run directories and deterministic artifact retention."""

from __future__ import annotations

import hmac
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator, Mapping

if TYPE_CHECKING:
    from src.app_manifest import AppManifest

from src.artifact_compatibility import (
    ARTIFACT_SCHEMA_VERSION,
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
from src.paths import resolve_dir, run_manifest_path
from src.run_provenance import (
    load_run_provenance_manifest,
    write_run_provenance_manifest,
)

RUNS_DIRECTORY = "runs"
CURRENT_RUN_FILENAME = "current.json"
DEFAULT_ARTIFACT_RETENTION_RUNS = 10
_PUBLICATION_PENDING = "pending"
_PUBLICATION_COMPLETE = "complete"


@dataclass(frozen=True)
class _RetainedDirectory:
    """Keep one directory descriptor and its original filesystem identity."""

    descriptor: int
    device: int
    inode: int


@dataclass(frozen=True)
class _RetainedFile:
    """Keep one regular-file descriptor and its immutable capture."""

    descriptor: int
    snapshot: RegularFileSnapshot


@dataclass(frozen=True)
class _HistoryDescriptors:
    """Retain the canonical root-to-run directory chain."""

    root_parent: _RetainedDirectory
    root: _RetainedDirectory
    runs: _RetainedDirectory
    run: _RetainedDirectory
    root_name: str
    run_name: str


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
    retained: list[_RetainedDirectory] = []
    try:
        root_parent = _open_retained_directory(root.parent)
        retained.append(root_parent)
        root_descriptor = _open_retained_directory(
            root.name, parent_descriptor=root_parent.descriptor
        )
        retained.append(root_descriptor)
        runs_descriptor = _open_retained_directory(
            runs_root.name, parent_descriptor=root_descriptor.descriptor
        )
        retained.append(runs_descriptor)
        run_descriptor = _open_retained_directory(
            run_dir.name, parent_descriptor=runs_descriptor.descriptor
        )
        retained.append(run_descriptor)
        descriptors = _HistoryDescriptors(
            root_parent,
            root_descriptor,
            runs_descriptor,
            run_descriptor,
            root.name,
            run_dir.name,
        )
        _verify_history_descriptors(descriptors)
        yield descriptors
    finally:
        for directory in reversed(retained):
            os.close(directory.descriptor)


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
    try:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise ValueError("artifact history changed during finalization") from error
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != (retained.device, retained.inode)
        or _directory_identity(retained.descriptor) != (retained.device, retained.inode)
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
    _verify_directory_entry(
        descriptors.root_parent.descriptor,
        descriptors.root_name,
        descriptors.root,
    )
    _verify_directory_entry(
        descriptors.root.descriptor, RUNS_DIRECTORY, descriptors.runs
    )
    _verify_directory_entry(
        descriptors.runs.descriptor, descriptors.run_name, descriptors.run
    )


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


@dataclass(frozen=True)
class _FinalizationLock:
    """Hold the per-run lock descriptor and durable publication state."""

    descriptor: int

    def state(self) -> tuple[str, str] | None:
        """Read the publication state stored in the lock file.

        Returns
        -------
        tuple of str and str or None
            State name and artifact-manifest checksum, or ``None`` when no
            publication attempt has reached pointer replacement.
        """
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        document = os.read(self.descriptor, 256)
        if not document:
            return None
        try:
            state, checksum = document.decode("ascii").rstrip("\n").split(":", 1)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("run finalization state is unsafe") from error
        checksum = f"sha256:{checksum}"
        if state not in {_PUBLICATION_PENDING, _PUBLICATION_COMPLETE} or not (
            len(checksum) == 71
            and all(character in "0123456789abcdef" for character in checksum[7:])
        ):
            raise ValueError("run finalization state is unsafe")
        return state, checksum

    def record(self, state: str, checksum: str) -> None:
        """Durably record one publication state transition.

        Parameters
        ----------
        state : str
            ``pending`` before pointer replacement or ``complete`` after every
            required durability barrier.
        checksum : str
            Exact artifact-manifest checksum bound by the pointer.

        Returns
        -------
        None
        """
        if state not in {_PUBLICATION_PENDING, _PUBLICATION_COMPLETE}:
            raise ValueError("run finalization state is unsafe")
        document = f"{state}:{checksum.removeprefix('sha256:')}\n".encode("ascii")
        os.ftruncate(self.descriptor, 0)
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        _write_all(self.descriptor, document)
        os.fsync(self.descriptor)


@contextmanager
def _finalization_lock(
    root_descriptor: int, run_id: str
) -> Iterator[_FinalizationLock]:
    """Serialize finalization attempts for one run across threads and processes.

    Parameters
    ----------
    root_descriptor : int
        Retained no-follow artifact-root descriptor.
    run_id : str
        Run identity used to isolate unrelated finalizations.

    Yields
    ------
    _FinalizationLock
        Locked descriptor and publication-state access.

    Raises
    ------
    ValueError
        If the lock path is not a contained single-link regular file.
    """
    name = f".{run_id}.finalize.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=root_descriptor)
    except OSError as error:
        raise ValueError("run finalization lock is unsafe") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise ValueError("run finalization lock is unsafe")
        if os.name == "nt":
            import msvcrt

            if os.lseek(descriptor, 0, os.SEEK_END) == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            locking = msvcrt.locking  # type: ignore[attr-defined]
            lock_mode = msvcrt.LK_LOCK  # type: ignore[attr-defined]
            locking(descriptor, lock_mode, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield _FinalizationLock(descriptor)
    finally:
        os.close(descriptor)


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
    root = resolve_dir(artifact_root)
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
    run_id = str(uuid.uuid4())
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        write_run_provenance_manifest(
            run_dir,
            run_config,
            public_artifact_dir=public_artifact_dir,
            client_shard=client_shard,
            app_manifest=app_manifest,
            flower_run_id=flower_run_id,
            run_id=run_id,
        )
    except BaseException:
        shutil.rmtree(run_dir)
        raise
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


def _replace_bytes_at(root_descriptor: int, name: str, content: bytes) -> None:
    """Atomically replace one file descriptor-relatively.

    Parameters
    ----------
    root_descriptor : int
        Retained owning directory descriptor.
    name : str
        Direct child destination name.
    content : bytes
        Exact replacement bytes.

    Returns
    -------
    None
    """
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_descriptor,
        )
        _write_all(descriptor, content)
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.rename(
            temporary,
            name,
            src_dir_fd=root_descriptor,
            dst_dir_fd=root_descriptor,
        )
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=root_descriptor)
        except FileNotFoundError:
            pass
        raise


def _rollback_current_index(
    descriptors: _HistoryDescriptors, previous_bytes: bytes | None
) -> None:
    """Restore the pre-publication pointer after an identity failure.

    Parameters
    ----------
    descriptors : _HistoryDescriptors
        Retained history descriptor chain.
    previous_bytes : bytes or None
        Exact prior pointer, or ``None`` when publication created it.

    Returns
    -------
    None
    """
    if previous_bytes is None:
        os.unlink(CURRENT_RUN_FILENAME, dir_fd=descriptors.root.descriptor)
    else:
        _replace_bytes_at(
            descriptors.root.descriptor, CURRENT_RUN_FILENAME, previous_bytes
        )
    _sync_retained_directory(descriptors.root)


def _write_current_index_at(
    root: Path,
    descriptors: _HistoryDescriptors,
    index: Mapping[str, Any],
    previous_bytes: bytes | None,
) -> Path:
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
    pathlib.Path
        Published current pointer path.
    """
    _verify_history_descriptors(descriptors)
    _replace_bytes_at(
        descriptors.root.descriptor,
        CURRENT_RUN_FILENAME,
        canonical_json_bytes(index),
    )
    try:
        _verify_history_descriptors(descriptors)
    except ValueError:
        _rollback_current_index(descriptors, previous_bytes)
        raise
    _sync_retained_directory(descriptors.root)
    return root / CURRENT_RUN_FILENAME


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
    candidate_run_dir = Path(run_dir)
    if candidate_run_dir.is_symlink() or not candidate_run_dir.is_dir():
        raise ValueError("run directory must be a regular directory")
    resolved_run_dir = candidate_run_dir.resolve(strict=True)
    expected_parent = runs_root.resolve(strict=True)
    if resolved_run_dir.parent != expected_parent:
        raise ValueError("run directory must be directly below the artifact runs root")
    with _retain_history_descriptors(
        root, expected_parent, resolved_run_dir
    ) as descriptors:
        with _finalization_lock(
            descriptors.root.descriptor, resolved_run_dir.name
        ) as finalization_lock:
            _verify_history_descriptors(descriptors)
            current_record = _load_current_index_at(root, descriptors.root.descriptor)

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
                            write_server_artifact_manifest(
                                resolved_run_dir,
                                app_manifest=validated_manifest,
                                finalized=True,
                                artifact_snapshot=artifact_snapshot,
                            )
                        elif dict(snapshot.files) != {
                            name: retained.snapshot.content
                            for name, retained in artifact_inventory.items()
                        }:
                            raise ValueError(
                                "completed run differs from its retained snapshot"
                            )
                    else:
                        write_server_artifact_manifest(
                            resolved_run_dir,
                            app_manifest=validated_manifest,
                            finalized=True,
                            artifact_snapshot=artifact_snapshot,
                        )
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
                        state = finalization_lock.state()
                        if was_complete:
                            if state != (_PUBLICATION_PENDING, checksum):
                                raise ValueError(
                                    "completed run cannot be finalized again"
                                )
                        elif state is None:
                            finalization_lock.record(_PUBLICATION_PENDING, checksum)
                            _sync_retained_directory(descriptors.root)
                            state = (_PUBLICATION_PENDING, checksum)
                        elif state != (_PUBLICATION_PENDING, checksum):
                            raise ValueError("run finalization state is unsafe")
                        _sync_retained_files(completed_inventory)
                        _sync_retained_directory(descriptors.run)
                        _sync_retained_directory(descriptors.runs)
                        _verify_history_descriptors(descriptors)
                        _verify_run_inventory(
                            descriptors.run.descriptor,
                            completed_inventory,
                            expected_content,
                        )

                        if current_record is not None:
                            current, current_bytes = current_record
                            exact_current = (
                                current["run_id"] == resolved_run_dir.name
                                and current["artifact_manifest_checksum"] == checksum
                            )
                            if exact_current and state == (
                                _PUBLICATION_PENDING,
                                checksum,
                            ):
                                _sync_retained_directory(descriptors.root)
                                _verify_history_descriptors(descriptors)
                                _verify_run_inventory(
                                    descriptors.run.descriptor,
                                    completed_inventory,
                                    expected_content,
                                )
                                recovered = _load_current_index_at(
                                    root, descriptors.root.descriptor
                                )
                                if recovered != current_record:
                                    raise ValueError(
                                        "current-run index changed during finalization"
                                    )
                                finalization_lock.record(
                                    _PUBLICATION_COMPLETE, checksum
                                )
                                return root / CURRENT_RUN_FILENAME
                            if was_complete:
                                raise ValueError(
                                    "completed run cannot be finalized again"
                                )

                        _verify_history_descriptors(descriptors)
                        _verify_run_inventory(
                            descriptors.run.descriptor,
                            completed_inventory,
                            expected_content,
                        )
                        previous_bytes = (
                            None if current_record is None else current_record[1]
                        )
                        pointer = _write_current_index_at(
                            root, descriptors, index, previous_bytes
                        )
                        try:
                            _verify_history_descriptors(descriptors)
                            _verify_run_inventory(
                                descriptors.run.descriptor,
                                completed_inventory,
                                expected_content,
                            )
                        except ValueError:
                            _rollback_current_index(descriptors, previous_bytes)
                            raise
                        finalization_lock.record(_PUBLICATION_COMPLETE, checksum)
                        return pointer
                    finally:
                        os.close(retained_manifest.descriptor)
            except FileNotFoundError as error:
                raise ValueError("server artifact is missing or unsafe") from error


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

    protected: set[Path] = set()
    index = _load_current_index(root)
    if index is None:
        return []
    protected.add(load_current_run_snapshot(root).directory)
    if active_run_dir is not None:
        active_path = Path(active_run_dir)
        if active_path.is_symlink():
            raise ValueError("active run must be a regular directory")
        if not active_path.exists():
            active = None
        else:
            active = active_path.resolve(strict=True)
        if active is not None and active.parent != runs_root.resolve(strict=True):
            raise ValueError("active run must be directly below the artifact runs root")
        if active is not None:
            protected.add(active)

    candidates: list[tuple[datetime, str, Path]] = []
    for path in runs_root.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            canonical_path = path.resolve(strict=True)
            manifest_path = run_manifest_path(canonical_path)
            manifest_bytes = read_regular_file(manifest_path, parent=canonical_path)
            manifest = load_run_provenance_manifest(
                manifest_path, manifest_bytes=manifest_bytes
            )
        except ValueError:
            continue
        if manifest["run_id"] != path.name:
            continue
        created_at = datetime.fromisoformat(
            str(manifest["created_at"]).replace("Z", "+00:00")
        )
        candidates.append((created_at, path.name, path.resolve()))

    candidates.sort(reverse=True)
    keep = set(protected)
    for _, _, path in candidates:
        if path in keep:
            continue
        if len(keep) < retention_runs:
            keep.add(path)

    deleted: list[Path] = []
    for _, _, path in reversed(candidates):
        if path not in keep:
            shutil.rmtree(path)
            deleted.append(path)
    return deleted
