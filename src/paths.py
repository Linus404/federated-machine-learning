from __future__ import annotations

import os
from pathlib import Path

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


def partition_paths(data_dir: str | Path, partition: int) -> tuple[Path, Path]:
    base = data_dir_path(data_dir)
    return base / f"partition_{partition}_x.npy", base / f"partition_{partition}_y.npy"


def vocab_path(data_dir: str | Path) -> Path:
    return data_dir_path(data_dir) / "vocab.txt"


def global_model_path(artifact_dir: str | Path) -> Path:
    return artifact_dir_path(artifact_dir) / "global_model.keras"


def metrics_path(artifact_dir: str | Path) -> Path:
    return artifact_dir_path(artifact_dir) / "metrics.csv"
