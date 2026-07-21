from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import BinaryIO, Iterable

PUBLIC_ARTIFACT_DIR_ENV = "FML_PUBLIC_ARTIFACT_DIR"
SERVER_ARTIFACT_DIR_ENV = "FML_SERVER_ARTIFACT_DIR"
DEFAULT_PUBLIC_ARTIFACT_DIR = Path("artifacts/public")
DEFAULT_SERVER_ARTIFACT_DIR = Path("artifacts/server")


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
    lock_file = lock_path.open("a+b")

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


def clear_artifact_dir(
    artifact_dir: str | Path | None = None,
    protected_paths: Iterable[str | Path] = (),
) -> Path:
    """Remove stale run artifacts and return the resolved artifact directory."""
    artifact_path = resolve_dir(artifact_dir or default_server_artifact_dir())
    if artifact_path.is_symlink():
        raise ValueError(
            f"Refusing to clear symlinked artifact directory: {artifact_path}"
        )

    absolute_artifact_dir = artifact_path.absolute()
    resolved_artifact_dir = artifact_path.resolve()
    current_dir = Path.cwd().resolve()
    protected_dirs = {current_dir, *current_dir.parents}

    if (
        resolved_artifact_dir == resolved_artifact_dir.parent
        or resolved_artifact_dir in protected_dirs
    ):
        raise ValueError(
            f"Refusing to clear unsafe artifact directory: {resolved_artifact_dir}"
        )

    for protected_path in protected_paths:
        absolute_protected_path = resolve_dir(protected_path).absolute()
        resolved_protected_path = absolute_protected_path.resolve()
        if (
            absolute_artifact_dir == absolute_protected_path
            or resolved_artifact_dir == resolved_protected_path
        ):
            raise ValueError(
                "Refusing to clear artifact directory because it matches protected "
                f"path: {resolved_artifact_dir}"
            )
        if absolute_protected_path.is_relative_to(
            absolute_artifact_dir
        ) or resolved_protected_path.is_relative_to(resolved_artifact_dir):
            raise ValueError(
                "Refusing to clear artifact directory because it contains protected "
                f"path: {resolved_protected_path}"
            )

    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    for child in resolved_artifact_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()

    return resolved_artifact_dir


def global_model_path(artifact_dir: str | Path) -> Path:
    return resolve_dir(artifact_dir) / "global_model.keras"


def metrics_path(artifact_dir: str | Path) -> Path:
    return resolve_dir(artifact_dir) / "metrics.csv"


def client_metrics_path(artifact_dir: str | Path) -> Path:
    return resolve_dir(artifact_dir) / "client_metrics.csv"
