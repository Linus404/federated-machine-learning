"""Publish and validate the fixed untouched global evaluation dataset."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tomllib
import uuid
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.artifact_compatibility import (
    RetainedDirectoryChain,
    canonical_json_bytes,
    deep_freeze,
    read_regular_file_snapshot_at,
    require_secure_artifact_platform,
    sha256_bytes,
)
from src.paths import RunArtifactLock, resolve_prepared_artifact_dir

EVALUATION_ARTIFACT_SCHEMA_VERSION = 1
EVALUATION_MANIFEST_FILENAME = "manifest.json"
EVALUATION_RECORDS_FILENAME = "test.jsonl"
SCIENTIFIC_PROTOCOL_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "scientific-protocol-v1.toml"
)
_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_type",
    "lifecycle",
    "dataset",
    "records",
    "checksums",
}
_DATASET_FIELDS = {
    "id",
    "config",
    "revision",
    "datasets_version",
    "split",
    "rows",
    "label_counts",
    "raw_parquet_sha256",
    "content_sha256",
}
_RECORD_FIELDS = {
    "filename",
    "format",
    "encoding",
    "newline",
    "trailing_newline",
    "row_count",
    "fields",
    "row_identity",
    "order",
}
_EVALUATION_STATE_SCHEMA_VERSION = 1
_EVALUATION_STATE_SUFFIX = ".publication.state"
_EVALUATION_OWNERSHIP_FILENAME = ".publication-owner"
_EVALUATION_STATE_FIELDS = {
    "schema_version",
    "operation",
    "parent_device",
    "parent_inode",
    "output_name",
    "nonce",
    "stage_name",
    "device",
    "inode",
    "source_name",
    "tombstone_name",
}
_STATE_TEMPORARY_NAME_ATTEMPTS = 128


@dataclass(frozen=True)
class EvaluationArtifactSnapshot:
    """Validated immutable bytes for the untouched evaluation dataset.

    Parameters
    ----------
    directory : pathlib.Path
        Canonical evaluation artifact directory.
    manifest : mapping of str to Any
        Strictly validated manifest payload.
    records : bytes
        Canonical JSONL bytes verified against the manifest and frozen protocol.
    """

    directory: Path
    manifest: Mapping[str, Any]
    records: bytes


@dataclass(frozen=True)
class _RetainedEvaluationFile:
    """Retain one staged evaluation file and its exact identity and bytes."""

    descriptor: int
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int
    content: bytes


@dataclass(frozen=True)
class _RetainedEvaluationParent:
    """Retain the complete visible chain through the destination parent."""

    path: Path
    chain: RetainedDirectoryChain

    @property
    def descriptor(self) -> int:
        """Return the destination-parent descriptor.

        Returns
        -------
        int
            Retained final directory descriptor.
        """
        return self.chain.directory.descriptor

    @property
    def owner_descriptor(self) -> int | None:
        """Return the immediate owner descriptor when one exists.

        Returns
        -------
        int or None
            Immediate retained parent, excluding the filesystem anchor.
        """
        directories = self.chain.directories
        return directories[-2].descriptor if len(directories) > 1 else None

    @property
    def name(self) -> str | None:
        """Return the visible destination-parent basename.

        Returns
        -------
        str or None
            Basename, or ``None`` for the filesystem anchor.
        """
        return None if self.path == Path(self.path.anchor) else self.path.name


@dataclass(frozen=True)
class _EvaluationOwnershipState:
    """Bind one evaluation publication mutation to durable ownership evidence."""

    operation: str
    parent_device: int
    parent_inode: int
    output_name: str
    nonce: str
    stage_name: str
    device: int | None = None
    inode: int | None = None
    source_name: str | None = None
    tombstone_name: str | None = None

    def payload(self) -> dict[str, object]:
        """Return the canonical serialization payload.

        Returns
        -------
        dict of str to object
            Exact durable ownership record.
        """
        return {
            "schema_version": _EVALUATION_STATE_SCHEMA_VERSION,
            "operation": self.operation,
            "parent_device": self.parent_device,
            "parent_inode": self.parent_inode,
            "output_name": self.output_name,
            "nonce": self.nonce,
            "stage_name": self.stage_name,
            "device": self.device,
            "inode": self.inode,
            "source_name": self.source_name,
            "tombstone_name": self.tombstone_name,
        }


def load_scientific_protocol() -> Mapping[str, Any]:
    """Load the frozen scientific protocol from the repository.

    Returns
    -------
    collections.abc.Mapping
        Parsed frozen protocol.

    Raises
    ------
    ValueError
        If the protocol is not the frozen version used by this artifact schema.
    """
    protocol = tomllib.loads(SCIENTIFIC_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != 1 or protocol.get("status") != "frozen":
        raise ValueError("scientific protocol must be frozen protocol version 1")
    return protocol


def canonical_source_row_bytes(text: str, label: int) -> bytes:
    """Serialize one upstream row under the frozen content-hash contract.

    Parameters
    ----------
    text : str
        Review text preserved exactly as supplied by the dataset.
    label : int
        Dataset label.

    Returns
    -------
    bytes
        Compact UTF-8 JSON followed by one LF byte.
    """
    return (
        json.dumps(
            {"label": label, "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_evaluation_row_bytes(index: int, text: str, label: int) -> bytes:
    """Serialize one split-qualified untouched evaluation record.

    Parameters
    ----------
    index : int
        Zero-based official test-split row index.
    text : str
        Review text preserved exactly as supplied by the dataset.
    label : int
        Binary sentiment label.

    Returns
    -------
    bytes
        Compact UTF-8 JSON followed by one LF byte.
    """
    return (
        json.dumps(
            {"label": label, "row_id": f"test:{index}", "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _evaluation_dataset_manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen test-dataset identity for an evaluation manifest.

    Parameters
    ----------
    protocol : mapping of str to Any
        Parsed frozen scientific protocol.

    Returns
    -------
    dict of str to Any
        Exact test-split identity permitted at the evaluation boundary.
    """
    dataset = protocol["dataset"]
    split = dataset["splits"]["test"]
    return {
        "id": dataset["id"],
        "config": dataset["config"],
        "revision": dataset["revision"],
        "datasets_version": dataset["datasets_version"],
        "split": "test",
        "rows": split["rows"],
        "label_counts": split["label_counts"],
        "raw_parquet_sha256": split["raw_parquet_sha256"],
        "content_sha256": split["content_sha256"],
    }


def _open_new_artifact_parent(
    output_dir: str | Path,
) -> tuple[_RetainedEvaluationParent, str]:
    """Open a symlink-free parent chain for a new evaluation artifact.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        New evaluation artifact directory.

    Returns
    -------
    tuple of _RetainedEvaluationParent and str
        Retained visible parent chain and destination basename.

    Raises
    ------
    ValueError
        If any existing component is a symlink or unsafe path type.
    """
    candidate = Path(output_dir).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    output_path = Path(os.path.abspath(candidate))
    if output_path.name in {"", ".", ".."}:
        raise ValueError("evaluation artifact path must name a child directory")

    chain = RetainedDirectoryChain.open(
        output_path.parent,
        create=True,
        error_message=(
            "every existing evaluation artifact path component must be a regular "
            "directory"
        ),
        check_platform=False,
    )
    return _RetainedEvaluationParent(output_path.parent, chain), output_path.name


def _verify_evaluation_parent(parent: _RetainedEvaluationParent) -> None:
    """Require the visible parent path to retain its opened identity.

    Parameters
    ----------
    parent : _RetainedEvaluationParent
        Retained destination parent and its owning directory.

    Returns
    -------
    None
    """
    try:
        parent.chain.verify()
    except ValueError:
        raise ValueError("evaluation parent changed during publication")


def _fsync_evaluation_descriptor(
    parent: _RetainedEvaluationParent, descriptor: int
) -> None:
    """Flush one file or directory while its complete path remains visible.

    Parameters
    ----------
    parent : _RetainedEvaluationParent
        Complete retained destination-parent chain.
    descriptor : int
        Open file or directory descriptor to flush.

    Returns
    -------
    None
    """
    _verify_evaluation_parent(parent)
    os.fsync(descriptor)
    _verify_evaluation_parent(parent)


def _acquire_evaluation_lock(
    parent_descriptor: int, output_name: str
) -> RunArtifactLock:
    """Lock one evaluation destination through its validated parent descriptor.

    Parameters
    ----------
    parent_descriptor : int
        Held descriptor for the validated destination parent.
    output_name : str
        Direct child destination basename.

    Returns
    -------
    RunArtifactLock
        Exclusive nonblocking publication lock.

    Raises
    ------
    RuntimeError
        If another process owns the destination lock.
    ValueError
        If the lock entry is not a single-link regular file.
    """
    lock_name = f".{output_name}.run.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_name, flags, 0o600, dir_fd=parent_descriptor)
    except OSError as error:
        raise ValueError("evaluation artifact lock path is unsafe") from error
    lock_stat = os.fstat(descriptor)
    if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
        os.close(descriptor)
        raise ValueError("evaluation artifact lock path is unsafe")
    file = os.fdopen(descriptor, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if file.seek(0, os.SEEK_END) == 0:
                file.write(b"\0")
                file.flush()
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        file.close()
        raise RuntimeError(
            "Another evaluation artifact publication is already in progress"
        ) from error
    return RunArtifactLock(Path(lock_name), file)


def _destination_exists(parent_descriptor: int, output_name: str) -> bool:
    """Return whether a destination entry exists without following it.

    Parameters
    ----------
    parent_descriptor : int
        Held descriptor for the validated parent.
    output_name : str
        Direct child basename.

    Returns
    -------
    bool
        Whether any filesystem entry owns the name.
    """
    try:
        os.stat(output_name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _evaluation_state_name(output_name: str) -> str:
    """Return the private ownership-state name for one destination.

    Parameters
    ----------
    output_name : str
        Direct child destination basename.

    Returns
    -------
    str
        Descriptor-relative state filename.
    """
    return f".{output_name}{_EVALUATION_STATE_SUFFIX}"


def _read_regular_bytes_at(
    parent_descriptor: int, name: str
) -> tuple[bytes, tuple[int, int]]:
    """Read one stable single-link regular file without following links.

    Parameters
    ----------
    parent_descriptor : int
        Retained owning-directory descriptor.
    name : str
        Direct child filename.

    Returns
    -------
    tuple of bytes and tuple of int
        Exact stable file content and device/inode identity.
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
            raise ValueError("evaluation ownership state is unsafe")
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
            raise ValueError("evaluation ownership state changed while retained")
        return b"".join(chunks), (after.st_dev, after.st_ino)
    finally:
        os.close(descriptor)


def _parse_evaluation_state(document: bytes) -> _EvaluationOwnershipState:
    """Validate one canonical evaluation ownership record.

    Parameters
    ----------
    document : bytes
        Exact candidate state bytes.

    Returns
    -------
    _EvaluationOwnershipState
        Strict validated state.
    """
    payload = json.loads(document)
    if (
        not isinstance(payload, dict)
        or set(payload) != _EVALUATION_STATE_FIELDS
        or payload["schema_version"] != _EVALUATION_STATE_SCHEMA_VERSION
        or canonical_json_bytes(payload) != document
        or payload["operation"] not in {"reserved", "build", "install", "delete"}
        or type(payload["parent_device"]) is not int
        or payload["parent_device"] < 0
        or type(payload["parent_inode"]) is not int
        or payload["parent_inode"] < 1
        or not isinstance(payload["output_name"], str)
        or Path(payload["output_name"]).name != payload["output_name"]
        or not isinstance(payload["nonce"], str)
        or len(payload["nonce"]) != 32
        or any(character not in "0123456789abcdef" for character in payload["nonce"])
        or payload["stage_name"]
        != f".{payload['output_name']}.{payload['nonce']}.staging"
    ):
        raise ValueError("evaluation ownership state is unsafe")
    device = payload["device"]
    inode = payload["inode"]
    source_name = payload["source_name"]
    tombstone_name = payload["tombstone_name"]
    operation = payload["operation"]
    if operation == "reserved":
        valid_phase = all(
            value is None for value in (device, inode, source_name, tombstone_name)
        )
    else:
        valid_phase = (
            type(device) is int
            and device >= 0
            and type(inode) is int
            and inode >= 1
            and (source_name is None or Path(source_name).name == source_name)
            and (tombstone_name is None or Path(tombstone_name).name == tombstone_name)
        )
        if operation == "build":
            valid_phase = valid_phase and source_name is None and tombstone_name is None
        elif operation == "install":
            valid_phase = (
                valid_phase
                and source_name == payload["stage_name"]
                and tombstone_name is None
            )
        else:
            valid_phase = (
                valid_phase
                and source_name in {payload["stage_name"], payload["output_name"]}
                and tombstone_name
                == f".{payload['output_name']}.{device:x}-{inode:x}.deleting"
            )
    if not valid_phase:
        raise ValueError("evaluation ownership state is unsafe")
    return _EvaluationOwnershipState(
        operation,
        payload["parent_device"],
        payload["parent_inode"],
        payload["output_name"],
        payload["nonce"],
        payload["stage_name"],
        device,
        inode,
        source_name,
        tombstone_name,
    )


def _load_evaluation_state(
    parent_descriptor: int, output_name: str
) -> tuple[_EvaluationOwnershipState, tuple[bytes, tuple[int, int]]] | None:
    """Load the sole ownership record for one destination.

    Parameters
    ----------
    parent_descriptor : int
        Retained destination-parent descriptor.
    output_name : str
        Direct child destination basename.

    Returns
    -------
    tuple of _EvaluationOwnershipState and bytes or None
        Validated state and exact bytes, or ``None`` when absent.
    """
    name = _evaluation_state_name(output_name)
    try:
        record = _read_regular_bytes_at(parent_descriptor, name)
    except FileNotFoundError:
        return None
    state = _parse_evaluation_state(record[0])
    if state.output_name != output_name:
        raise ValueError("evaluation ownership state is unsafe")
    return state, record


def _write_evaluation_state(
    parent: _RetainedEvaluationParent,
    state: _EvaluationOwnershipState,
    previous: tuple[bytes, tuple[int, int]] | None,
    *,
    require_visible_parent: bool = True,
) -> tuple[bytes, tuple[int, int]]:
    """Install and durably flush an exact ownership-state transition.

    Parameters
    ----------
    parent : _RetainedEvaluationParent
        Retained complete destination-parent chain.
    state : _EvaluationOwnershipState
        New canonical state.
    previous : tuple of bytes and identity or None
        Exact prior state and identity, or ``None`` for exclusive creation.
    require_visible_parent : bool, optional
        Require the retained parent chain to remain visibly selected.

    Returns
    -------
    tuple of bytes and tuple of int
        Canonical installed bytes and state-file identity.
    """
    parent_stat = os.fstat(parent.descriptor)
    if (state.parent_device, state.parent_inode) != (
        parent_stat.st_dev,
        parent_stat.st_ino,
    ):
        raise ValueError("evaluation ownership state selects another parent")
    document = canonical_json_bytes(state.payload())
    name = _evaluation_state_name(state.output_name)
    descriptor: int | None = None
    temporary = ""
    for _ in range(_STATE_TEMPORARY_NAME_ATTEMPTS):
        candidate = f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent.descriptor,
            )
        except FileExistsError:
            continue
        temporary = candidate
        break
    if descriptor is None:
        raise FileExistsError("could not allocate evaluation ownership state")
    try:
        written = 0
        while written < len(document):
            count = os.write(descriptor, document[written:])
            if count == 0:
                raise OSError("evaluation ownership-state write made no progress")
            written += count
        os.fsync(descriptor)
        if require_visible_parent:
            _verify_evaluation_parent(parent)
        if previous is not None:
            if _read_regular_bytes_at(parent.descriptor, name) != previous:
                raise ValueError("evaluation ownership state changed while retained")
            os.unlink(name, dir_fd=parent.descriptor)
        os.link(
            temporary,
            name,
            src_dir_fd=parent.descriptor,
            dst_dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=parent.descriptor)
        temporary = ""
        if require_visible_parent:
            _fsync_evaluation_descriptor(parent, parent.descriptor)
        else:
            current = os.fstat(parent.descriptor)
            if (current.st_dev, current.st_ino) != (
                state.parent_device,
                state.parent_inode,
            ):
                raise ValueError("evaluation ownership state selects another parent")
            os.fsync(parent.descriptor)
        record = _read_regular_bytes_at(parent.descriptor, name)
        if record[0] != document:
            raise ValueError("evaluation ownership state changed while retained")
        return record
    finally:
        os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary, dir_fd=parent.descriptor)
            except FileNotFoundError:
                pass


def _remove_evaluation_state(
    parent: _RetainedEvaluationParent,
    output_name: str,
    record: tuple[bytes, tuple[int, int]],
    *,
    require_visible_parent: bool = True,
) -> None:
    """Durably remove only the exact retained ownership state.

    Parameters
    ----------
    parent : _RetainedEvaluationParent
        Retained complete destination-parent chain.
    output_name : str
        Direct child destination basename.
    record : tuple of bytes and identity
        Exact state bytes and identity authorized for removal.
    require_visible_parent : bool, optional
        Require the retained parent chain to remain visibly selected.

    Returns
    -------
    None
    """
    name = _evaluation_state_name(output_name)
    if _read_regular_bytes_at(parent.descriptor, name) != record:
        raise ValueError("evaluation ownership state changed while retained")
    os.unlink(name, dir_fd=parent.descriptor)
    if require_visible_parent:
        _fsync_evaluation_descriptor(parent, parent.descriptor)
    else:
        os.fsync(parent.descriptor)


def _open_bound_evaluation_directory(
    parent: _RetainedEvaluationParent,
    name: str,
    state: _EvaluationOwnershipState,
) -> int:
    """Open one state-bound directory and require its exact identity.

    Parameters
    ----------
    parent : _RetainedEvaluationParent
        Retained destination parent.
    name : str
        State-selected direct child name.
    state : _EvaluationOwnershipState
        Ownership state containing the expected identity.

    Returns
    -------
    int
        Open no-follow directory descriptor.
    """
    descriptor = parent.chain.open_child(name, os.O_RDONLY | os.O_DIRECTORY)
    current = os.fstat(descriptor)
    if (current.st_dev, current.st_ino) != (state.device, state.inode):
        os.close(descriptor)
        raise ValueError("evaluation ownership residue was replaced")
    return descriptor


def _evaluation_deletion_tombstone(output_name: str, descriptor: int) -> str:
    """Bind a recognizable private deletion name to one retained directory.

    Parameters
    ----------
    output_name : str
        Public evaluation destination basename.
    descriptor : int
        Retained staged or installed directory descriptor.

    Returns
    -------
    str
        Private tombstone containing the retained device and inode.
    """
    current = os.fstat(descriptor)
    if not stat.S_ISDIR(current.st_mode) or current.st_nlink < 1:
        raise ValueError("evaluation deletion ownership changed")
    return f".{output_name}.{current.st_dev:x}-{current.st_ino:x}.deleting"


def _delete_bound_evaluation_directory(
    parent: _RetainedEvaluationParent,
    state: _EvaluationOwnershipState,
    record: tuple[bytes, tuple[int, int]],
    source_name: str,
    descriptor: int,
    *,
    require_visible_parent: bool = True,
) -> None:
    """Transition to deletion and remove one exact owned directory tree.

    Parameters
    ----------
    parent : _RetainedEvaluationParent
        Retained complete destination-parent chain.
    state : _EvaluationOwnershipState
        Current bound ownership state.
    record : tuple of bytes and identity
        Exact current state bytes and identity.
    source_name : str
        Visible owned source name.
    descriptor : int
        Retained descriptor for the exact source directory.
    require_visible_parent : bool, optional
        Require the retained parent chain to remain visibly selected.

    Returns
    -------
    None
    """
    tombstone = _evaluation_deletion_tombstone(state.output_name, descriptor)
    deleting = _EvaluationOwnershipState(
        "delete",
        state.parent_device,
        state.parent_inode,
        state.output_name,
        state.nonce,
        state.stage_name,
        state.device,
        state.inode,
        source_name,
        tombstone,
    )
    deleting_record = _write_evaluation_state(
        parent,
        deleting,
        record,
        require_visible_parent=require_visible_parent,
    )
    parent.chain.remove_child_tree(
        source_name,
        descriptor,
        require_visible_chain=False,
        tombstone_name=tombstone,
    )
    _remove_evaluation_state(
        parent,
        state.output_name,
        deleting_record,
        require_visible_parent=require_visible_parent,
    )


def _recover_evaluation_state(
    parent: _RetainedEvaluationParent, output_name: str
) -> None:
    """Recover only the exact directory bound by durable ownership state.

    Parameters
    ----------
    parent : _RetainedEvaluationParent
        Retained complete destination-parent chain.
    output_name : str
        Direct child destination basename.

    Returns
    -------
    None
    """
    record = _load_evaluation_state(parent.descriptor, output_name)
    if record is None:
        return
    state, document = record
    parent_stat = os.fstat(parent.descriptor)
    if (state.parent_device, state.parent_inode) != (
        parent_stat.st_dev,
        parent_stat.st_ino,
    ):
        raise ValueError("evaluation ownership state selects another parent")
    if state.operation == "reserved":
        if _destination_exists(parent.descriptor, state.stage_name):
            raise ValueError("evaluation staging residue lacks bound identity")
        _remove_evaluation_state(parent, output_name, document)
        return

    if state.operation == "delete":
        assert state.source_name is not None
        assert state.tombstone_name is not None
        present = [
            name
            for name in (state.source_name, state.tombstone_name)
            if _destination_exists(parent.descriptor, name)
        ]
        if len(present) > 1:
            raise ValueError("evaluation deletion state is ambiguous")
        if not present:
            _remove_evaluation_state(parent, output_name, document)
            return
        name = present[0]
        descriptor = _open_bound_evaluation_directory(parent, name, state)
        try:
            if name == state.source_name:
                parent.chain.remove_child_tree(
                    name,
                    descriptor,
                    tombstone_name=state.tombstone_name,
                )
            else:
                parent.chain.remove_detached_child_tree(name, descriptor)
        finally:
            os.close(descriptor)
        _remove_evaluation_state(parent, output_name, document)
        return

    names = [state.stage_name]
    if state.operation == "install":
        names.append(output_name)
    present = [name for name in names if _destination_exists(parent.descriptor, name)]
    if len(present) > 1:
        raise ValueError("evaluation ownership state is ambiguous")
    if not present:
        _remove_evaluation_state(parent, output_name, document)
        return
    name = present[0]
    descriptor = _open_bound_evaluation_directory(parent, name, state)
    try:
        if state.operation == "install" and name == output_name:
            _remove_evaluation_state(parent, output_name, document)
            return
        _delete_bound_evaluation_directory(parent, state, document, name, descriptor)
    finally:
        os.close(descriptor)


def _read_retained_evaluation_file(
    directory_descriptor: int, name: str
) -> _RetainedEvaluationFile:
    """Open and retain one stable, single-link staged regular file.

    Parameters
    ----------
    directory_descriptor : int
        Retained staging directory descriptor.
    name : str
        Direct child filename.

    Returns
    -------
    _RetainedEvaluationFile
        Retained identity and exact bytes.
    """
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        dir_fd=directory_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("evaluation staging file is unsafe")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or len(content) != after.st_size:
            raise ValueError("evaluation staging file changed during publication")
        return _RetainedEvaluationFile(
            descriptor,
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            content,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _verify_evaluation_inventory(
    directory_descriptor: int,
    files: Mapping[str, _RetainedEvaluationFile],
) -> None:
    """Require the exact retained evaluation inventory and bytes.

    Parameters
    ----------
    directory_descriptor : int
        Retained staged or installed directory descriptor.
    files : mapping of str to _RetainedEvaluationFile
        Exact expected file inventory.

    Returns
    -------
    None
    """
    if set(os.listdir(directory_descriptor)) != set(files):
        raise ValueError("evaluation artifact inventory changed during publication")
    for name, retained in files.items():
        entry = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        current = os.fstat(retained.descriptor)
        os.lseek(retained.descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(retained.descriptor, 1024 * 1024):
            chunks.append(chunk)
        identity = (
            retained.device,
            retained.inode,
            retained.size_bytes,
            retained.modified_ns,
            retained.changed_ns,
        )
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or (
                entry.st_dev,
                entry.st_ino,
                entry.st_size,
                entry.st_mtime_ns,
                entry.st_ctime_ns,
            )
            != identity
            or (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            != identity
            or b"".join(chunks) != retained.content
        ):
            raise ValueError("evaluation artifact changed during publication")


def _directory_entry_matches_descriptor(
    parent_descriptor: int, name: str, directory_descriptor: int
) -> bool:
    """Return whether a name still selects a retained directory descriptor.

    Parameters
    ----------
    parent_descriptor : int
        Owning parent directory descriptor.
    name : str
        Direct child directory name.
    directory_descriptor : int
        Retained directory descriptor.

    Returns
    -------
    bool
        Whether the directory entry and retained descriptor identities match.
    """
    return RetainedDirectoryChain.entry_matches_descriptor(
        parent_descriptor, name, directory_descriptor
    )


def _validate_row(
    text: object, label: object, *, split: str, index: int
) -> tuple[str, int]:
    """Validate one dataset row at a frozen split position.

    Parameters
    ----------
    text : object
        Candidate review text.
    label : object
        Candidate binary sentiment label.
    split : str
        Official split name used in validation errors.
    index : int
        Zero-based official split row index.

    Returns
    -------
    tuple of str and int
        Validated text and binary label.

    Raises
    ------
    ValueError
        If text is not a string or label is not a built-in binary integer.
    """
    if not isinstance(text, str):
        raise ValueError(f"{split} row {index} has a non-string text value")
    if type(label) is not int or label not in (0, 1):
        raise ValueError(f"{split} row {index} has an invalid binary label")
    return text, label


def _publish_evaluation_artifact_unlocked(
    rows: Iterable[Mapping[str, Any]],
    parent: _RetainedEvaluationParent,
    output_name: str,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    """Publish the official test split into a new evaluation-only directory.

    Parameters
    ----------
    rows : iterable of mappings
        Official test rows in ascending source order.
    parent : _RetainedEvaluationParent
        Retained visible parent chain for mutation and the returned path.
    output_name : str
        Direct child destination basename.
    protocol : mapping or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    pathlib.Path
        Published evaluation artifact directory.

    Raises
    ------
    FileExistsError
        If the destination already exists.
    ValueError
        If rows or publication paths violate the frozen artifact contract.
    """
    frozen = protocol or load_scientific_protocol()
    dataset_manifest = _evaluation_dataset_manifest(frozen)
    parent_descriptor = parent.descriptor
    output_path = parent.path / output_name
    _verify_evaluation_parent(parent)
    if _destination_exists(parent_descriptor, output_name):
        raise FileExistsError(
            f"evaluation artifact path already exists; refusing replacement: {output_path}"
        )
    nonce = uuid.uuid4().hex
    staging_name = f".{output_name}.{nonce}.staging"
    parent_stat = os.fstat(parent_descriptor)
    state = _EvaluationOwnershipState(
        "reserved",
        parent_stat.st_dev,
        parent_stat.st_ino,
        output_name,
        nonce,
        staging_name,
    )
    state_document = _write_evaluation_state(parent, state, None)
    staging_descriptor: int | None = None
    record_hash = hashlib.sha256()
    content_hash = hashlib.sha256()
    label_counts: Counter[int] = Counter()
    row_count = 0
    retained_files: dict[str, _RetainedEvaluationFile] = {}
    destination_owned = False
    try:
        _verify_evaluation_parent(parent)
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        _verify_evaluation_parent(parent)
        staging_descriptor = parent.chain.open_child(
            staging_name,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        staged_identity = os.fstat(staging_descriptor)
        state = _EvaluationOwnershipState(
            "build",
            state.parent_device,
            state.parent_inode,
            output_name,
            nonce,
            staging_name,
            staged_identity.st_dev,
            staged_identity.st_ino,
        )
        state_document = _write_evaluation_state(parent, state, state_document)
        records_descriptor = parent.chain.open_child(
            EVALUATION_RECORDS_FILENAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            parent_descriptor=staging_descriptor,
        )
        with os.fdopen(records_descriptor, "wb") as file:
            for index, row in enumerate(rows):
                text, label = _validate_row(
                    row.get("text"), row.get("label"), split="test", index=index
                )
                content_hash.update(canonical_source_row_bytes(text, label))
                record = canonical_evaluation_row_bytes(index, text, label)
                file.write(record)
                record_hash.update(record)
                label_counts[label] += 1
                row_count += 1
            file.flush()
            _fsync_evaluation_descriptor(parent, file.fileno())
        _verify_evaluation_parent(parent)
        os.chmod(
            EVALUATION_RECORDS_FILENAME,
            0o644,
            dir_fd=staging_descriptor,
            follow_symlinks=False,
        )
        _verify_evaluation_parent(parent)

        expected_counts = dataset_manifest["label_counts"]
        if row_count != dataset_manifest["rows"]:
            raise ValueError(
                f"test row count mismatch: expected {dataset_manifest['rows']}, got {row_count}"
            )
        if [label_counts[0], label_counts[1]] != expected_counts:
            raise ValueError("test label counts differ from the frozen protocol")
        if content_hash.hexdigest() != dataset_manifest["content_sha256"]:
            raise ValueError(
                "test canonical content SHA-256 differs from the frozen protocol"
            )

        records_checksum = f"sha256:{record_hash.hexdigest()}"
        manifest = {
            "schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "untouched_global_test_set",
            "lifecycle": "complete",
            "dataset": dataset_manifest,
            "records": {
                "filename": EVALUATION_RECORDS_FILENAME,
                "format": "canonical-jsonl",
                "encoding": "utf-8",
                "newline": "LF",
                "trailing_newline": True,
                "row_count": row_count,
                "fields": ["label", "row_id", "text"],
                "row_identity": "{split}:{zero_based_official_split_row_index}",
                "order": "ascending_zero_based_official_split_row_index",
            },
            "checksums": {EVALUATION_RECORDS_FILENAME: records_checksum},
        }
        manifest_descriptor = parent.chain.open_child(
            EVALUATION_MANIFEST_FILENAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
            parent_descriptor=staging_descriptor,
        )
        with os.fdopen(manifest_descriptor, "wb") as file:
            manifest_bytes = canonical_json_bytes(manifest)
            file.write(manifest_bytes)
            file.flush()
            _fsync_evaluation_descriptor(parent, file.fileno())
        for name in (EVALUATION_MANIFEST_FILENAME, EVALUATION_RECORDS_FILENAME):
            retained_files[name] = _read_retained_evaluation_file(
                staging_descriptor, name
            )
        if retained_files[EVALUATION_MANIFEST_FILENAME].content != manifest_bytes:
            raise ValueError("evaluation manifest changed during publication")
        if (
            sha256_bytes(retained_files[EVALUATION_RECORDS_FILENAME].content)
            != records_checksum
        ):
            raise ValueError("evaluation records changed during publication")
        _verify_evaluation_inventory(staging_descriptor, retained_files)
        _fsync_evaluation_descriptor(parent, staging_descriptor)
        staged_stat = os.stat(
            staging_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        opened_stat = os.fstat(staging_descriptor)
        if not stat.S_ISDIR(staged_stat.st_mode) or (
            staged_stat.st_dev,
            staged_stat.st_ino,
        ) != (opened_stat.st_dev, opened_stat.st_ino):
            raise ValueError("evaluation staging directory changed during publication")
        state = _EvaluationOwnershipState(
            "install",
            state.parent_device,
            state.parent_inode,
            output_name,
            nonce,
            staging_name,
            state.device,
            state.inode,
            staging_name,
        )
        state_document = _write_evaluation_state(parent, state, state_document)
        _verify_evaluation_parent(parent)
        os.rename(
            staging_name,
            output_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        destination_owned = True
        _verify_evaluation_parent(parent)
        if not _directory_entry_matches_descriptor(
            parent_descriptor, output_name, staging_descriptor
        ):
            raise ValueError("evaluation destination changed during publication")
        _fsync_evaluation_descriptor(parent, parent_descriptor)
        _verify_evaluation_parent(parent)
        if not _directory_entry_matches_descriptor(
            parent_descriptor, output_name, staging_descriptor
        ):
            raise ValueError("evaluation destination changed during publication")
        _verify_evaluation_inventory(staging_descriptor, retained_files)
        _verify_evaluation_parent(parent)
        if not _directory_entry_matches_descriptor(
            parent_descriptor, output_name, staging_descriptor
        ):
            raise ValueError("evaluation destination changed during publication")
        parent.chain.commit()
        _verify_evaluation_parent(parent)
        _remove_evaluation_state(parent, output_name, state_document)
        return output_path
    except BaseException:
        if staging_descriptor is not None:
            try:
                owned_names = (
                    (output_name, staging_name)
                    if destination_owned
                    else (staging_name, output_name)
                )
                for owned_name in owned_names:
                    if _directory_entry_matches_descriptor(
                        parent_descriptor, owned_name, staging_descriptor
                    ):
                        _delete_bound_evaluation_directory(
                            parent,
                            state,
                            state_document,
                            owned_name,
                            staging_descriptor,
                            require_visible_parent=False,
                        )
                        break
            except (FileNotFoundError, OSError, ValueError):
                pass
        raise
    finally:
        for retained in retained_files.values():
            os.close(retained.descriptor)
        if staging_descriptor is not None:
            os.close(staging_descriptor)


def publish_evaluation_artifact(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    """Publish one immutable evaluation artifact under an exclusive lock.

    Parameters
    ----------
    rows : iterable of mappings
        Official test rows in ascending source order.
    output_dir : str or pathlib.Path
        New dedicated artifact directory.
    protocol : mapping or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    pathlib.Path
        Published evaluation artifact directory.

    Raises
    ------
    FileExistsError
        If the immutable destination already exists.
    RuntimeError
        If another writer owns the same destination or direct publication runs
        outside the supported Linux platform.
    ValueError
        If owned residue, rows, or paths are invalid.
    """
    require_secure_artifact_platform()
    parent, output_name = _open_new_artifact_parent(output_dir)
    with parent.chain:
        _verify_evaluation_parent(parent)
        lock = _acquire_evaluation_lock(parent.descriptor, output_name)
        try:
            _verify_evaluation_parent(parent)
            _recover_evaluation_state(parent, output_name)
            _verify_evaluation_parent(parent)
            result = _publish_evaluation_artifact_unlocked(
                rows,
                parent,
                output_name,
                protocol=protocol,
            )
        finally:
            lock.release()
        _verify_evaluation_parent(parent)
        return result


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], name: str
) -> None:
    """Require an evaluation manifest object to contain exactly known fields.

    Parameters
    ----------
    payload : mapping of str to Any
        Manifest object whose keys are validated.
    expected : set of str
        Complete permitted field set.
    name : str
        Human-readable object name used in errors.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the object contains missing or additional fields.
    """
    if set(payload) != expected:
        raise ValueError(f"evaluation {name} has an invalid field set")


def _validate_manifest(
    payload: object, protocol: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate the exact untouched-evaluation manifest contract.

    Parameters
    ----------
    payload : object
        Decoded candidate manifest.
    protocol : mapping of str to Any
        Frozen protocol providing the authoritative test identity.

    Returns
    -------
    mapping of str to Any
        Validated manifest mapping.

    Raises
    ------
    ValueError
        If schema, fields, lifecycle, dataset, records, or checksums differ.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation manifest must be a JSON object")
    _require_exact_fields(payload, _MANIFEST_FIELDS, "manifest")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != EVALUATION_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("evaluation manifest has an unsupported schema_version")
    if payload["artifact_type"] != "untouched_global_test_set":
        raise ValueError("evaluation manifest has an invalid artifact_type")
    if payload["lifecycle"] != "complete":
        raise ValueError("evaluation manifest is not complete")

    dataset = payload["dataset"]
    records = payload["records"]
    checksums = payload["checksums"]
    if not isinstance(dataset, Mapping):
        raise ValueError("evaluation manifest dataset must be an object")
    if not isinstance(records, Mapping):
        raise ValueError("evaluation manifest records must be an object")
    if not isinstance(checksums, Mapping):
        raise ValueError("evaluation manifest checksums must be an object")
    _require_exact_fields(dataset, _DATASET_FIELDS, "dataset")
    _require_exact_fields(records, _RECORD_FIELDS, "records")
    if dict(dataset) != _evaluation_dataset_manifest(protocol):
        raise ValueError("evaluation dataset identity differs from the frozen protocol")
    expected_records = {
        "filename": EVALUATION_RECORDS_FILENAME,
        "format": "canonical-jsonl",
        "encoding": "utf-8",
        "newline": "LF",
        "trailing_newline": True,
        "row_count": dataset["rows"],
        "fields": ["label", "row_id", "text"],
        "row_identity": "{split}:{zero_based_official_split_row_index}",
        "order": "ascending_zero_based_official_split_row_index",
    }
    if dict(records) != expected_records:
        raise ValueError("evaluation record contract is invalid")
    if set(checksums) != {EVALUATION_RECORDS_FILENAME} or not isinstance(
        checksums[EVALUATION_RECORDS_FILENAME], str
    ):
        raise ValueError("evaluation manifest checksums are invalid")
    return payload


def load_evaluation_artifact_snapshot(
    artifact_dir: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> EvaluationArtifactSnapshot:
    """Read and fully verify an untouched evaluation artifact snapshot.

    Parameters
    ----------
    artifact_dir : str or pathlib.Path
        Directory containing the evaluation manifest and canonical records.
    protocol : mapping or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    EvaluationArtifactSnapshot
        Manifest and record bytes verified in one pass.

    Raises
    ------
    ValueError
        If any path, byte, identity, ordering, or checksum is invalid.
    """
    directory = resolve_prepared_artifact_dir(artifact_dir, "evaluation")
    frozen = protocol or load_scientific_protocol()
    chain = RetainedDirectoryChain.open(
        directory,
        error_message="evaluation artifact directory chain changed while loading",
    )
    with chain:
        canonical_dir = chain.path
        descriptor = chain.directory.descriptor
        expected_inventory = {
            EVALUATION_MANIFEST_FILENAME,
            EVALUATION_RECORDS_FILENAME,
        }
        with ExitStack() as retained_files:
            try:
                chain.verify()
                retained_manifest = retained_files.enter_context(
                    read_regular_file_snapshot_at(
                        descriptor,
                        EVALUATION_MANIFEST_FILENAME,
                        retain=True,
                    )
                )
                manifest_bytes = retained_manifest.snapshot.content
                chain.verify()
                decoded_manifest = json.loads(manifest_bytes.decode("utf-8"))
                payload = _validate_manifest(decoded_manifest, frozen)
                chain.verify()
                canonical_manifest = (
                    json.dumps(decoded_manifest, indent=2, allow_nan=False) + "\n"
                ).encode("utf-8")
                if manifest_bytes != canonical_manifest:
                    raise ValueError("evaluation manifest bytes are not canonical")
                chain.verify()
                if set(os.listdir(descriptor)) != expected_inventory:
                    raise ValueError("evaluation artifact contains unexpected files")
                chain.verify()
                retained_records = retained_files.enter_context(
                    read_regular_file_snapshot_at(
                        descriptor,
                        EVALUATION_RECORDS_FILENAME,
                        retain=True,
                    )
                )
                records = retained_records.snapshot.content
                chain.verify()
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
            ) as error:
                raise ValueError("invalid evaluation artifact") from error

            expected_checksum = payload["checksums"][EVALUATION_RECORDS_FILENAME]
            if sha256_bytes(records) != expected_checksum:
                raise ValueError("evaluation record checksum mismatch")
            chain.verify()

            content_hash = hashlib.sha256()
            label_counts: Counter[int] = Counter()
            lines = records.splitlines(keepends=True)
            if len(lines) != payload["records"]["row_count"]:
                raise ValueError("evaluation record row count mismatch")
            for index, line in enumerate(lines):
                if not line.endswith(b"\n"):
                    raise ValueError("evaluation records must end every row with LF")
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid evaluation record at row {index}"
                    ) from error
                if not isinstance(row, Mapping) or set(row) != {
                    "label",
                    "row_id",
                    "text",
                }:
                    raise ValueError(f"invalid evaluation record fields at row {index}")
                text, label = _validate_row(
                    row["text"], row["label"], split="test", index=index
                )
                chain.verify()
                if row["row_id"] != f"test:{index}":
                    raise ValueError(
                        f"evaluation row identity or order mismatch at row {index}"
                    )
                if line != canonical_evaluation_row_bytes(index, text, label):
                    raise ValueError(
                        f"evaluation record is not canonical at row {index}"
                    )
                content_hash.update(canonical_source_row_bytes(text, label))
                label_counts[label] += 1

            dataset = payload["dataset"]
            if [label_counts[0], label_counts[1]] != dataset["label_counts"]:
                raise ValueError("evaluation label counts differ from the manifest")
            if content_hash.hexdigest() != dataset["content_sha256"]:
                raise ValueError("evaluation content SHA-256 differs from the manifest")
            snapshot = EvaluationArtifactSnapshot(
                canonical_dir,
                deep_freeze(payload),
                records,
            )
            chain.verify()
            if set(os.listdir(descriptor)) != expected_inventory:
                raise ValueError("evaluation artifact inventory changed while loading")
            retained_manifest.verify(
                descriptor,
                EVALUATION_MANIFEST_FILENAME,
                expected_content=manifest_bytes,
            )
            retained_records.verify(
                descriptor,
                EVALUATION_RECORDS_FILENAME,
                expected_content=records,
            )
            chain.verify()
            return snapshot
