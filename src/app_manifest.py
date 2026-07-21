"""Load the public vocabulary manifest used by all app entrypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifact_compatibility import validate_artifact_schema
from src.paths import default_public_artifact_dir, resolve_dir


@dataclass(frozen=True)
class AppManifest:
    payload: dict[str, Any]
    vocabulary_path: Path


def configured_value(config, key):
    value = config.get(key) if config else None
    return (
        value
        if value is not None and (not isinstance(value, str) or value.strip())
        else None
    )


def resolve_public_artifact_dir(config=None, *, public_artifact_dir=None) -> Path:
    value = public_artifact_dir or configured_value(config, "public-artifact-dir")
    return resolve_dir(value) if value else default_public_artifact_dir()


def load_app_manifest(*, public_artifact_dir=None) -> AppManifest:
    """Load the public model and vocabulary manifest.

    Parameters
    ----------
    public_artifact_dir : str or pathlib.Path, optional
        Directory containing ``manifest.json`` and the shared vocabulary.

    Returns
    -------
    AppManifest
        Validated manifest payload and vocabulary path.
    """
    public_dir = resolve_public_artifact_dir(public_artifact_dir=public_artifact_dir)
    path = public_dir / "manifest.json"
    payload = dict(
        validate_artifact_schema(
            json.loads(path.read_text(encoding="utf-8")), "public manifest"
        )
    )

    # These are the only manifest values the product consumes.
    required = {"embedding_dim", "sequence_length", "vocabulary_size", "vocabulary"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Manifest missing: {', '.join(sorted(missing))}")
    vocabulary_path = public_dir / payload["vocabulary"]["filename"]
    return AppManifest(payload, vocabulary_path)
