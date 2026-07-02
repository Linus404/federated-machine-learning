from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

DATA_DIR_ENV = "FML_DATA_DIR"
ARTIFACT_DIR_ENV = "FML_ARTIFACT_DIR"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_ARTIFACT_DIR = Path("artifacts")


def resolve_dir(value: str | Path) -> Path:
    """Return an absolute path, treating relative paths as launch-dir relative."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def default_data_dir() -> Path:
    """Return the configured data directory, defaulting to the repo's data folder."""
    return resolve_dir(os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR))


def default_artifact_dir() -> Path:
    """Return the configured artifact directory, defaulting to the repo's artifacts folder."""
    return resolve_dir(os.environ.get(ARTIFACT_DIR_ENV, DEFAULT_ARTIFACT_DIR))


def data_dir_path(data_dir: str | Path | None = None) -> Path:
    return default_data_dir() if data_dir is None else resolve_dir(data_dir)


def artifact_dir_path(artifact_dir: str | Path | None = None) -> Path:
    return default_artifact_dir() if artifact_dir is None else resolve_dir(artifact_dir)


def clear_artifact_dir(
    artifact_dir: str | Path | None = None,
    protected_paths: Iterable[str | Path] = (),
) -> Path:
    """Remove stale run artifacts and return the resolved artifact directory."""
    artifact_path = artifact_dir_path(artifact_dir)
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


def partition_paths(data_dir: str | Path, partition: int) -> tuple[Path, Path]:
    base = data_dir_path(data_dir)
    return base / f"partition_{partition}_x.npy", base / f"partition_{partition}_y.npy"


def vocab_path(data_dir: str | Path) -> Path:
    return data_dir_path(data_dir) / "vocab.txt"


def global_model_path(artifact_dir: str | Path) -> Path:
    return artifact_dir_path(artifact_dir) / "global_model.keras"


def metrics_path(artifact_dir: str | Path) -> Path:
    return artifact_dir_path(artifact_dir) / "metrics.csv"
