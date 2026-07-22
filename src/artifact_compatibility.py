"""Artifact schema compatibility checks shared by producers and consumers."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Never

if TYPE_CHECKING:
    from src.app_manifest import AppManifest

ARTIFACT_SCHEMA_VERSION = 1
PUBLIC_ARTIFACT_SCHEMA_VERSION = 2
CLIENT_SHARD_SCHEMA_VERSION = 2
SERVER_ARTIFACT_SCHEMA_VERSION = 4
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
    "manifest.json",
    "metrics.csv",
    "run_manifest.json",
    "vocab.txt",
}
_SERVER_BINDING_FIELDS = {
    "public_manifest_checksum",
    "vocabulary_checksum",
    "model_dimensions",
}
_MODEL_DIMENSION_FIELDS = {"vocabulary_size", "sequence_length", "embedding_dim"}


@dataclass(frozen=True)
class RetainedDirectory:
    """Retain one no-follow directory descriptor and filesystem identity.

    Parameters
    ----------
    descriptor : int
        Open directory descriptor owned by a retained chain.
    device : int
        Captured filesystem device.
    inode : int
        Captured filesystem inode.
    """

    descriptor: int
    device: int
    inode: int


class RetainedDirectoryChain(AbstractContextManager["RetainedDirectoryChain"]):
    """Own and verify every directory edge from ``/`` through one path."""

    def __init__(
        self,
        path: Path,
        directories: list[RetainedDirectory],
        names: list[str],
        created: list[bool],
        *,
        error_message: str,
    ) -> None:
        self.path = path
        self._directories = directories
        self._names = names
        self._created = created
        self._error_message = error_message
        self._committed = False
        self._closed = False

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        create: bool = False,
        mode: int = 0o755,
        error_message: str = "artifact directory chain changed",
        check_platform: bool = True,
    ) -> "RetainedDirectoryChain":
        """Open an absolute, symlink-free directory chain.

        Parameters
        ----------
        path : str or pathlib.Path
            Directory to retain. Relative paths are launch-directory relative.
        create : bool, optional
            Create missing components descriptor-relatively.
        mode : int, optional
            Permissions requested for newly created components.
        error_message : str, optional
            Fail-closed message used for unsafe or replaced edges.
        check_platform : bool, optional
            Enforce the Linux contract unless the caller already did so.

        Returns
        -------
        RetainedDirectoryChain
            Owned descriptors for the complete visible path.

        Raises
        ------
        ValueError
            If a component is missing, linked, replaced, or not a directory.
        """
        if check_platform:
            require_secure_artifact_platform()
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        absolute = Path(os.path.abspath(candidate))
        anchor = Path(absolute.anchor)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directories: list[RetainedDirectory] = []
        names: list[str] = []
        created: list[bool] = []
        chain = cls(
            absolute,
            directories,
            names,
            created,
            error_message=error_message,
        )
        try:
            anchor_descriptor = os.open(anchor, flags)
            directories.append(
                cls._capture(anchor_descriptor, error_message=error_message)
            )
            created.append(False)
            for component in absolute.parts[len(anchor.parts) :]:
                parent = directories[-1]
                made = False
                try:
                    entry = os.stat(
                        component,
                        dir_fd=parent.descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    if not create:
                        raise ValueError(error_message) from None
                    chain.verify()
                    os.mkdir(component, mode=mode, dir_fd=parent.descriptor)
                    made = True
                    entry = os.stat(
                        component,
                        dir_fd=parent.descriptor,
                        follow_symlinks=False,
                    )
                if not stat.S_ISDIR(entry.st_mode):
                    raise ValueError(error_message)
                descriptor = os.open(component, flags, dir_fd=parent.descriptor)
                retained = cls._capture(descriptor, error_message=error_message)
                directories.append(retained)
                names.append(component)
                created.append(made)
                chain.verify()
                if made:
                    os.fsync(parent.descriptor)
                    chain.verify()
            chain.verify()
            return chain
        except BaseException:
            chain._cleanup_created()
            chain.close()
            raise

    @staticmethod
    def _capture(descriptor: int, *, error_message: str) -> RetainedDirectory:
        """Capture one open directory descriptor.

        Parameters
        ----------
        descriptor : int
            Open descriptor whose ownership transfers to the returned value.
        error_message : str
            Fail-closed validation error.

        Returns
        -------
        RetainedDirectory
            Captured descriptor identity.
        """
        try:
            current = os.fstat(descriptor)
            if not stat.S_ISDIR(current.st_mode) or current.st_nlink < 1:
                raise ValueError(error_message)
            return RetainedDirectory(descriptor, current.st_dev, current.st_ino)
        except BaseException:
            os.close(descriptor)
            raise

    @property
    def directory(self) -> RetainedDirectory:
        """Return the retained target directory.

        Returns
        -------
        RetainedDirectory
            Final descriptor in the chain.
        """
        return self._directories[-1]

    @property
    def directories(self) -> tuple[RetainedDirectory, ...]:
        """Return retained directories from the anchor through the target.

        Returns
        -------
        tuple of RetainedDirectory
            Stable retained chain.
        """
        return tuple(self._directories)

    def at(self, path: str | Path) -> RetainedDirectory:
        """Return the retained directory corresponding to one chain path.

        Parameters
        ----------
        path : str or pathlib.Path
            Absolute directory already retained by this chain.

        Returns
        -------
        RetainedDirectory
            Descriptor for the requested path.

        Raises
        ------
        ValueError
            If the path is outside or below the retained target.
        """
        candidate = Path(os.path.abspath(Path(path)))
        anchor = Path(self.path.anchor)
        parts = candidate.parts[len(anchor.parts) :]
        if (
            candidate.anchor != self.path.anchor
            or self.path.parts[: len(candidate.parts)] != candidate.parts
        ):
            raise ValueError("directory is not part of the retained chain")
        return self._directories[len(parts)]

    def verify(self) -> None:
        """Require every retained descriptor and visible edge to still match.

        Returns
        -------
        None
        """
        if self._closed or not self._directories:
            raise ValueError(self._error_message)
        anchor = os.stat(self.path.anchor, follow_symlinks=False)
        self._verify_identity(anchor, self._directories[0])
        for index, name in enumerate(self._names, start=1):
            try:
                entry = os.stat(
                    name,
                    dir_fd=self._directories[index - 1].descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ValueError(self._error_message) from error
            self._verify_identity(entry, self._directories[index])

    def stat(self, name: str) -> os.stat_result:
        """Stat one direct child without following links under chain checks.

        Parameters
        ----------
        name : str
            Direct child name.

        Returns
        -------
        os.stat_result
            No-follow entry metadata.
        """
        self._require_child_name(name)
        self.verify()
        result = os.stat(name, dir_fd=self.directory.descriptor, follow_symlinks=False)
        self.verify()
        return result

    def open_child(
        self,
        name: str,
        flags: int,
        mode: int = 0o777,
        *,
        parent_descriptor: int | None = None,
    ) -> int:
        """Open one direct child without following it under chain checks.

        Parameters
        ----------
        name : str
            Direct child name.
        flags : int
            Linux ``open(2)`` flags.
        mode : int, optional
            Creation permissions when ``flags`` includes ``O_CREAT``.
        parent_descriptor : int or None, optional
            Retained descendant descriptor, or the chain target when omitted.

        Returns
        -------
        int
            Caller-owned file descriptor.
        """
        self._require_child_name(name)
        self.verify()
        owner = (
            self.directory.descriptor
            if parent_descriptor is None
            else parent_descriptor
        )
        try:
            descriptor = os.open(
                name,
                flags | os.O_NOFOLLOW,
                mode,
                dir_fd=owner,
            )
        except OSError:
            self.verify()
            raise
        try:
            self.verify()
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def fsync(self, directory: RetainedDirectory | None = None) -> None:
        """Flush one retained directory while preserving visible ownership.

        Parameters
        ----------
        directory : RetainedDirectory or None, optional
            Chain member to flush, or the target directory when omitted.

        Returns
        -------
        None
        """
        retained = self.directory if directory is None else directory
        if retained not in self._directories:
            raise ValueError("directory is not part of the retained chain")
        self.verify()
        os.fsync(retained.descriptor)
        self.verify()

    def commit(self) -> None:
        """Preserve invocation-created directories after a verified success.

        Returns
        -------
        None
        """
        self.verify()
        self._committed = True

    def remove_created_target(self) -> None:
        """Remove an invocation-created target tree while it remains visible.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the target was not created by this chain or is no longer visible.
        """
        if len(self._directories) < 2 or not self._created[-1]:
            raise ValueError("retained target is not invocation-owned")
        self.verify()
        child = self._directories[-1]
        parent = self._directories[-2]
        shutil.rmtree(self._names[-1], dir_fd=parent.descriptor)
        os.fsync(parent.descriptor)
        os.close(child.descriptor)
        self._directories.pop()
        self._created.pop()
        self._names.pop()
        self.verify()

    def _verify_identity(
        self, entry: os.stat_result, retained: RetainedDirectory
    ) -> None:
        """Require an edge and descriptor to name the same linked directory.

        Parameters
        ----------
        entry : os.stat_result
            No-follow metadata for the visible entry.
        retained : RetainedDirectory
            Captured descriptor identity.

        Returns
        -------
        None
        """
        current = os.fstat(retained.descriptor)
        if (
            not stat.S_ISDIR(entry.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or entry.st_nlink < 1
            or current.st_nlink < 1
            or (entry.st_dev, entry.st_ino) != (retained.device, retained.inode)
            or (current.st_dev, current.st_ino) != (retained.device, retained.inode)
        ):
            raise ValueError(self._error_message)

    @staticmethod
    def entry_matches_descriptor(
        parent_descriptor: int, name: str, descriptor: int
    ) -> bool:
        """Return whether one edge exactly selects a linked directory descriptor.

        Parameters
        ----------
        parent_descriptor : int
            Retained owning directory descriptor.
        name : str
            Direct child name.
        descriptor : int
            Retained child directory descriptor.

        Returns
        -------
        bool
            Whether type, link state, device, and inode all match.
        """
        try:
            RetainedDirectoryChain._require_child_name(name)
            entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            current = os.fstat(descriptor)
        except (OSError, ValueError):
            return False
        return (
            stat.S_ISDIR(entry.st_mode)
            and stat.S_ISDIR(current.st_mode)
            and entry.st_nlink >= 1
            and current.st_nlink >= 1
            and (entry.st_dev, entry.st_ino) == (current.st_dev, current.st_ino)
        )

    @staticmethod
    def _require_child_name(name: str) -> None:
        """Require one non-traversing direct child name.

        Parameters
        ----------
        name : str
            Candidate descriptor-relative child name.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the name is empty, special, or contains a path separator.
        """
        if name in {"", ".", ".."} or Path(name).name != name:
            raise ValueError("descriptor-relative name must be a direct child")

    def _cleanup_created(self) -> None:
        """Remove only still-visible, invocation-owned empty suffixes."""
        if self._committed:
            return
        while len(self._directories) > 1 and self._created[-1]:
            child = self._directories[-1]
            parent = self._directories[-2]
            name = self._names[-1]
            try:
                self.verify()
                if os.listdir(child.descriptor):
                    return
                os.rmdir(name, dir_fd=parent.descriptor)
                os.fsync(parent.descriptor)
            except (OSError, ValueError):
                return
            os.close(child.descriptor)
            self._directories.pop()
            self._created.pop()
            self._names.pop()
            try:
                self.verify()
            except ValueError:
                return

    def close(self) -> None:
        """Close every owned descriptor exactly once.

        Returns
        -------
        None
        """
        if self._closed:
            return
        for directory in reversed(self._directories):
            os.close(directory.descriptor)
        self._directories.clear()
        self._names.clear()
        self._created.clear()
        self._closed = True

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Clean failed creations and close the complete retained chain."""
        if exc_type is not None:
            self._cleanup_created()
        self.close()


def reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Decode one JSON object while rejecting duplicate member names.

    Parameters
    ----------
    pairs : list of tuple of str and object
        Decoded object members in source order.

    Returns
    -------
    dict of str to object
        Unique decoded members preserving their source order.

    Raises
    ------
    ValueError
        If a member name occurs more than once.
    """
    result: dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object member: {key}")
        result[key] = item
    return result


def _reject_non_finite_json_constant(constant: str) -> Never:
    """Reject one non-standard non-finite JSON numeric constant.

    Parameters
    ----------
    constant : str
        Decoder token, such as ``NaN`` or ``Infinity``.

    Raises
    ------
    ValueError
        Always, because JSON does not define non-finite numeric constants.
    """
    raise ValueError(f"non-finite JSON constant: {constant}")


def _parse_finite_json_float(value: str) -> float:
    """Decode one JSON float while rejecting overflow to infinity.

    Parameters
    ----------
    value : str
        Valid JSON numeric token supplied by the decoder.

    Returns
    -------
    float
        Finite decoded value with the standard JSON decoder's float semantics.

    Raises
    ------
    ValueError
        If the numeric token decodes to a non-finite float.
    """
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON float: {value}")
    return parsed


def strict_json_loads(document: str | bytes | bytearray, *, source: str) -> Any:
    """Decode strict JSON and normalize parser failures for one trust boundary.

    Parameters
    ----------
    document : str, bytes, or bytearray
        JSON document to decode.
    source : str
        Boundary-specific description included in the public error.

    Returns
    -------
    Any
        Decoded JSON value.

    Raises
    ------
    ValueError
        If UTF-8 or JSON syntax is invalid, an object member is duplicated, or a
        non-finite numeric constant is present.
    """
    try:
        return json.loads(
            document,
            object_pairs_hook=reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {source}") from error


def require_secure_artifact_platform() -> None:
    """Require the Linux filesystem primitives used by artifact boundaries.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the process cannot enforce descriptor-relative, no-follow artifact
        publication and validation.
    """
    required_dir_fd = (os.open, os.stat, os.mkdir, os.rename)
    if (
        sys.platform != "linux"
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or any(function not in os.supports_dir_fd for function in required_dir_fd)
        or os.stat not in os.supports_follow_symlinks
        or not shutil.rmtree.avoids_symlink_attacks
    ):
        raise RuntimeError(
            "secure artifact publication and validation require Linux; use the "
            "documented Linux-container workflow on Windows or macOS"
        )


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


@dataclass(frozen=True)
class RegularFileSnapshot:
    """Retain bytes and filesystem identity from one secure descriptor read.

    Parameters
    ----------
    content : bytes
        Exact bytes read from the opened file descriptor.
    device : int
        Device identity reported by ``fstat``.
    inode : int
        Inode identity reported by ``fstat``.
    size_bytes : int
        File size reported by ``fstat``.
    modified_ns : int
        Nanosecond modification timestamp reported by ``fstat``.
    changed_ns : int
        Nanosecond metadata-change timestamp reported by ``fstat``.
    """

    content: bytes
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class ServerFinalizationSnapshot:
    """Retain every finalizable server file from one secure capture pass.

    Parameters
    ----------
    files : mapping of str to RegularFileSnapshot
        Immutable filename-to-descriptor snapshot mapping.
    """

    files: Mapping[str, RegularFileSnapshot]


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
    RuntimeError
        If the process cannot enforce the Linux artifact filesystem contract.
    ValueError
        If the path escapes its parent or is not a regular file.
    """
    return read_regular_file_snapshot(path, parent=parent).content


def read_regular_file_snapshot(
    path: Path, *, parent: Path, sync: bool = False
) -> RegularFileSnapshot:
    """Read one regular file while retaining descriptor identity metadata.

    Parameters
    ----------
    path : pathlib.Path
        File to read.
    parent : pathlib.Path
        Canonical directory that must directly contain the file.
    sync : bool, optional
        Flush the opened regular file before returning its stable snapshot.

    Returns
    -------
    RegularFileSnapshot
        Exact bytes and stable descriptor identity from the same read.

    Raises
    ------
    RuntimeError
        If the process cannot enforce the Linux artifact filesystem contract.
    ValueError
        If the path escapes its parent, is not regular, or changes during the read.
    """
    require_secure_artifact_platform()
    canonical_parent = parent.resolve(strict=True)
    if path.parent.resolve(strict=True) != canonical_parent or path.is_symlink():
        raise ValueError(f"artifact must be a contained regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    parent_descriptor = -1
    try:
        parent_descriptor = os.open(
            canonical_parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        raise ValueError(
            f"artifact must be a contained regular file: {path.name}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"artifact must be a contained regular file: {path.name}")
        with os.fdopen(descriptor, "rb") as file:
            descriptor = -1
            content = file.read()
            if sync:
                os.fsync(file.fileno())
            after = os.fstat(file.fileno())
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
                raise ValueError(f"artifact changed while reading: {path.name}")
            return RegularFileSnapshot(content, *identity)
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
    return write_bytes_atomically(
        path, canonical_json_bytes(payload), overwrite=overwrite
    )


def write_bytes_atomically(
    path: Path, content: bytes, *, overwrite: bool = True
) -> Path:
    """Persist exact bytes atomically.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    content : bytes
        Exact bytes to persist.
    overwrite : bool, optional
        Replace an existing destination when ``True``. When ``False``, publish by
        atomic hard link so an existing immutable file cannot be replaced.

    Returns
    -------
    pathlib.Path
        Destination path.

    Raises
    ------
    FileExistsError
        If ``overwrite`` is ``False`` and the destination already exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            os.fchmod(file.fileno(), 0o644)
            file.flush()
            os.fsync(file.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
            temporary_path.unlink()
        sync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def sync_directory(path: Path) -> None:
    """Flush one directory entry set through a no-follow descriptor.

    Parameters
    ----------
    path : pathlib.Path
        Existing regular directory to synchronize.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the Linux filesystem contract is unavailable.
    ValueError
        If the path is not a no-follow regular directory.
    OSError
        If the durability barrier fails.
    """
    require_secure_artifact_platform()
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise ValueError(f"artifact directory is unsafe: {path}") from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"artifact directory is unsafe: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    artifact_snapshot: ServerFinalizationSnapshot | None = None,
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
    artifact_snapshot : ServerFinalizationSnapshot or None, optional
        Securely retained artifact snapshot used for finalization.

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
            document = read_regular_file(path, parent=canonical_dir)
        except ValueError as error:
            raise ValueError("invalid existing server artifact manifest") from error
        decoded = strict_json_loads(
            document,
            source="existing server artifact manifest",
        )
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
        retained = (
            capture_server_artifact_files(artifact_dir)
            if artifact_snapshot is None
            else artifact_snapshot
        )
        missing = sorted(REQUIRED_COMPLETED_ARTIFACTS - retained.files.keys())
        if missing:
            raise ValueError(
                "cannot finalize run with missing artifacts: "
                + ", ".join(sorted(missing))
            )
        verify_server_artifact_files(artifact_dir, retained)
        payload["lifecycle"] = "complete"
        payload["sizes"] = {
            name: snapshot.size_bytes for name, snapshot in retained.files.items()
        }
        payload["checksums"] = {
            name: sha256_bytes(snapshot.content)
            for name, snapshot in retained.files.items()
        }
    return write_json_atomically(path, payload)


def capture_server_artifact_files(artifact_dir: Path) -> ServerFinalizationSnapshot:
    """Retain one secure snapshot of every finalizable server artifact.

    Parameters
    ----------
    artifact_dir : pathlib.Path
        Direct server run directory whose regular files are retained.

    Returns
    -------
    ServerFinalizationSnapshot
        Immutable descriptor snapshots excluding the artifact manifest.

    Raises
    ------
    ValueError
        If inventory entries are missing, linked, non-regular, or unsafe.
    """
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError("server artifact directory must be a regular directory")
    canonical_dir = artifact_dir.resolve(strict=True)
    files: dict[str, RegularFileSnapshot] = {}
    for artifact_path in sorted(artifact_dir.iterdir(), key=lambda path: path.name):
        if artifact_path.name == SERVER_ARTIFACT_MANIFEST_FILENAME:
            continue
        files[artifact_path.name] = read_regular_file_snapshot(
            artifact_path, parent=canonical_dir
        )
    missing = sorted(REQUIRED_COMPLETED_ARTIFACTS - files.keys())
    if missing:
        raise ValueError(
            "cannot finalize run with missing artifacts: " + ", ".join(missing)
        )
    return ServerFinalizationSnapshot(MappingProxyType(files))


def verify_server_artifact_files(
    artifact_dir: Path,
    artifact_snapshot: ServerFinalizationSnapshot,
    *,
    sync: bool = False,
) -> None:
    """Require current paths to remain identical to retained artifact bytes.

    Parameters
    ----------
    artifact_dir : pathlib.Path
        Direct server run directory owning the retained files.
    artifact_snapshot : ServerFinalizationSnapshot
        Previously retained secure snapshot.
    sync : bool, optional
        Flush every retained regular file before returning.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the inventory, path type, inode target, or content changed.
    """
    files = artifact_snapshot.files
    if any(
        not isinstance(name, str)
        or Path(name).name != name
        or not isinstance(snapshot, RegularFileSnapshot)
        for name, snapshot in files.items()
    ):
        raise ValueError("retained server artifact snapshot is invalid")
    current_names = {
        path.name
        for path in artifact_dir.iterdir()
        if path.name != SERVER_ARTIFACT_MANIFEST_FILENAME
    }
    if current_names != set(files):
        raise ValueError("server artifact inventory changed during finalization")
    canonical_dir = artifact_dir.resolve(strict=True)
    for name, retained in files.items():
        current = read_regular_file_snapshot(
            artifact_dir / name,
            parent=canonical_dir,
            sync=sync,
        )
        if current != retained:
            raise ValueError(f"server artifact changed during finalization: {name}")


def sync_server_artifact_files(
    artifact_dir: Path, artifact_snapshot: ServerFinalizationSnapshot
) -> None:
    """Flush every securely retained completed-run regular file.

    Parameters
    ----------
    artifact_dir : pathlib.Path
        Direct server run directory owning the retained files.
    artifact_snapshot : ServerFinalizationSnapshot
        Previously retained secure snapshot.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an artifact changed or became unsafe before its durability barrier.
    OSError
        If any file durability barrier fails.
    """
    verify_server_artifact_files(artifact_dir, artifact_snapshot, sync=True)


def validate_run_provenance_evidence(
    provenance: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    retained_manifest: AppManifest,
) -> None:
    """Validate run provenance and server binding against retained public bytes.

    Parameters
    ----------
    provenance : mapping of str to Any
        Validated run provenance manifest.
    artifact_manifest : mapping of str to Any
        Validated server artifact manifest.
    retained_manifest : src.app_manifest.AppManifest
        Public snapshot reconstructed from the retained bytes.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If provenance or server binding differs from retained public bytes.
    """
    if dict(artifact_manifest["binding"]) != server_artifact_binding(retained_manifest):
        raise ValueError(
            "server artifact binding does not match retained public evidence"
        )
    manifest_checksum = sha256_bytes(retained_manifest.manifest_bytes)
    expected_checksums = {
        "manifest.json": manifest_checksum,
        "vocab.txt": sha256_bytes(retained_manifest.vocabulary_bytes),
    }
    expected_public_manifest = {
        "filename": "manifest.json",
        "size_bytes": len(retained_manifest.manifest_bytes),
        "checksum": manifest_checksum,
    }
    expected_identity = json.dumps(
        dict(retained_manifest.payload["dataset"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    dataset = provenance["dataset"]
    if (
        dataset["status"] != "available"
        or dict(dataset["checksums"]) != expected_checksums
        or dict(dataset["public_manifest"]) != expected_public_manifest
        or dataset["identity"] != expected_identity
    ):
        raise ValueError("run provenance does not match retained public evidence")


def _validate_completed_run_provenance(
    artifact_dir: Path,
    artifact_manifest: Mapping[str, Any],
    files: Mapping[str, bytes],
    retained_manifest: AppManifest,
) -> None:
    """Validate completed provenance bytes and directory identity.

    Parameters
    ----------
    artifact_dir : pathlib.Path
        Canonical completed run directory.
    artifact_manifest : mapping of str to Any
        Validated completed server artifact manifest.
    files : mapping of str to bytes
        Exact checksummed completed artifact bytes.
    retained_manifest : src.app_manifest.AppManifest
        Public snapshot reconstructed from the retained bytes.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If provenance identity or public evidence differs from retained bytes.
    """
    from src.run_provenance import load_run_provenance_manifest

    provenance = load_run_provenance_manifest(
        artifact_dir / "run_manifest.json",
        manifest_bytes=files["run_manifest.json"],
    )
    if provenance["run_id"] != artifact_dir.name:
        raise ValueError("completed run directory does not match its provenance run_id")
    validate_run_provenance_evidence(
        provenance,
        artifact_manifest,
        retained_manifest,
    )


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
    payload = validate_artifact_schema(
        strict_json_loads(
            manifest_bytes,
            source=f"server artifact manifest: {path}",
        ),
        "server artifact manifest",
        supported_version=SERVER_ARTIFACT_SCHEMA_VERSION,
    )
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
    sizes = payload.get("sizes")
    if lifecycle is None and checksums is None and sizes is None:
        filenames = {
            str(layout["filename"])
            for layout in SERVER_ARTIFACTS.values()
            if (artifact_dir / str(layout["filename"])).exists()
        }
    else:
        if (
            lifecycle != "complete"
            or not isinstance(checksums, Mapping)
            or not isinstance(sizes, Mapping)
            or set(sizes) != set(checksums)
        ):
            raise ValueError("server artifact manifest has invalid completion metadata")
        if not REQUIRED_COMPLETED_ARTIFACTS <= checksums.keys():
            raise ValueError("server artifact manifest is missing required checksums")
        filenames = set(checksums)
        inventory = {entry.name for entry in artifact_dir.iterdir()}
        expected_inventory = {SERVER_ARTIFACT_MANIFEST_FILENAME, *filenames}
        if inventory != expected_inventory:
            raise ValueError("completed server artifact inventory does not match")

    files: dict[str, bytes] = {}
    for filename in filenames:
        expected = checksums.get(filename) if isinstance(checksums, Mapping) else None
        expected_size = sizes.get(filename) if isinstance(sizes, Mapping) else None
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
            or (expected_size is not None and type(expected_size) is not int)
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
        if expected_size is not None and len(content) != expected_size:
            raise ValueError(f"server artifact size does not match: {filename}")
        files[filename] = content
    if lifecycle == "complete":
        from src.app_manifest import validate_app_manifest_bytes

        retained_manifest = validate_app_manifest_bytes(
            files["manifest.json"],
            files["vocab.txt"],
            vocabulary_path=artifact_dir / "vocab.txt",
        )
        _validate_server_binding(payload.get("binding"), app_manifest=retained_manifest)
        if {entry.name for entry in artifact_dir.iterdir()} != expected_inventory:
            raise ValueError(
                "completed server artifact inventory changed while loading"
            )
        _validate_completed_run_provenance(
            canonical_dir,
            payload,
            files,
            retained_manifest,
        )
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
