"""Load the public vocabulary manifest used by all app entrypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.paths import default_public_artifact_dir, resolve_dir


@dataclass(frozen=True)
class AppManifest:
    path: Path
    public_artifact_dir: Path
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


def load_app_manifest(
    config=None,
    *,
    public_manifest_path=None,
    public_artifact_dir=None,
    config_embedding_dim=None,
) -> AppManifest:
    public_dir = resolve_public_artifact_dir(
        config, public_artifact_dir=public_artifact_dir
    )
    path = resolve_dir(
        public_manifest_path
        or configured_value(config, "public-manifest-path")
        or public_dir / "manifest.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    # These are the only manifest values the product consumes.
    required = {"embedding_dim", "sequence_length", "vocabulary_size", "vocabulary"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Manifest missing: {', '.join(sorted(missing))}")
    if config_embedding_dim is not None and int(payload["embedding_dim"]) != int(
        config_embedding_dim
    ):
        raise ValueError("Manifest and run config use different embedding dimensions")

    vocabulary_path = public_dir / payload["vocabulary"]["filename"]
    return AppManifest(path, public_dir, payload, vocabulary_path)
