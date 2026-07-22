"""Manage immutable run directories and deterministic artifact retention."""

from __future__ import annotations

import hmac
import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

if TYPE_CHECKING:
    from src.app_manifest import AppManifest

from src.artifact_compatibility import (
    ARTIFACT_SCHEMA_VERSION,
    SERVER_ARTIFACT_MANIFEST_FILENAME,
    ServerArtifactSnapshot,
    load_server_artifact_snapshot,
    read_regular_file,
    sha256_bytes,
    write_json_atomically,
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


@contextmanager
def _finalization_lock(artifact_root: Path, run_id: str) -> Iterator[None]:
    """Serialize finalization attempts for one run across threads and processes.

    Parameters
    ----------
    artifact_root : pathlib.Path
        Canonical history root that owns the lock file.
    run_id : str
        Run identity used to isolate unrelated finalizations.

    Yields
    ------
    None
        Control while the exclusive finalization lock is held.

    Raises
    ------
    ValueError
        If the lock path is not a contained single-link regular file.
    """
    path = artifact_root / f".{run_id}.finalize.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
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
        yield
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
            flower_run_id=flower_run_id,
            run_id=run_id,
        )
    except BaseException:
        shutil.rmtree(run_dir)
        raise
    return run_dir


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
    if not path.exists():
        return None
    try:
        payload = json.loads(read_regular_file(path, parent=root).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid current-run index: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("current-run index must be a JSON object")
    version = payload.get("schema_version")
    if type(version) is not int or version != ARTIFACT_SCHEMA_VERSION:
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
    return load_server_artifact_snapshot(
        canonical_run_dir,
        manifest_bytes=manifest_bytes,
        app_manifest=app_manifest,
    )


def publish_completed_run(artifact_root: str | Path, run_dir: str | Path) -> Path:
    """Finalize checksums and atomically select one completed run.

    Parameters
    ----------
    artifact_root : str or pathlib.Path
        Root containing run history.
    run_dir : str or pathlib.Path
        Completed run directory below the root.

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
    with _finalization_lock(root, resolved_run_dir.name):
        current = _load_current_index(root)
        if current is not None and current["run_id"] == resolved_run_dir.name:
            raise ValueError("completed run cannot be finalized again")

        provenance_path = run_manifest_path(resolved_run_dir)
        provenance_bytes = read_regular_file(provenance_path, parent=resolved_run_dir)
        provenance = load_run_provenance_manifest(
            provenance_path, manifest_bytes=provenance_bytes
        )
        if provenance["run_id"] != resolved_run_dir.name:
            raise ValueError("run directory does not match its provenance run_id")
        manifest_path = resolved_run_dir / SERVER_ARTIFACT_MANIFEST_FILENAME
        if manifest_path.exists():
            manifest_bytes = read_regular_file(manifest_path, parent=resolved_run_dir)
            snapshot = load_server_artifact_snapshot(
                resolved_run_dir, manifest_bytes=manifest_bytes
            )
            if snapshot.manifest.get("lifecycle") != "complete":
                manifest_path = write_server_artifact_manifest(
                    resolved_run_dir, finalized=True
                )
                manifest_bytes = read_regular_file(
                    manifest_path, parent=resolved_run_dir
                )
        else:
            manifest_path = write_server_artifact_manifest(
                resolved_run_dir, finalized=True
            )
            manifest_bytes = read_regular_file(manifest_path, parent=resolved_run_dir)
        index = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": resolved_run_dir.name,
            "artifact_manifest_checksum": sha256_bytes(manifest_bytes),
        }
        return write_json_atomically(root / CURRENT_RUN_FILENAME, index)


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
