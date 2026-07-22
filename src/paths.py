from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import BinaryIO

PUBLIC_ARTIFACT_DIR_ENV = "FML_PUBLIC_ARTIFACT_DIR"
SERVER_ARTIFACT_DIR_ENV = "FML_SERVER_ARTIFACT_DIR"
EVALUATION_ARTIFACT_DIR_ENV = "FML_EVALUATION_ARTIFACT_DIR"
DEFAULT_PUBLIC_ARTIFACT_DIR = Path("artifacts/public")
DEFAULT_SERVER_ARTIFACT_DIR = Path("artifacts/server")
DEFAULT_EVALUATION_ARTIFACT_DIR = Path("artifacts/evaluation")


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
