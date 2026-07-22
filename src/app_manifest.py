"""Load the public vocabulary manifest used by all app entrypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from src.artifact_compatibility import (
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    read_regular_file,
    sha256_bytes,
    validate_artifact_schema,
)
from src.evaluation_artifact import load_scientific_protocol
from src.paths import default_public_artifact_dir, resolve_dir


@dataclass(frozen=True)
class AppManifest:
    """Validated immutable snapshot of one public application manifest.

    Parameters
    ----------
    payload : mapping of str to Any
        Validated decoded manifest payload.
    vocabulary_path : pathlib.Path
        Original verified vocabulary path, retained for diagnostics only.
    manifest_bytes : bytes
        Exact manifest bytes read during validation.
    vocabulary_bytes : bytes
        Exact vocabulary bytes read during validation.
    vocabulary_terms : tuple of str
        Parsed vocabulary terms including the two reserved tokens.
    """

    payload: Mapping[str, Any]
    vocabulary_path: Path
    manifest_bytes: bytes
    vocabulary_bytes: bytes
    vocabulary_terms: tuple[str, ...]


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


def _expected_train_dataset(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact public train-dataset identity from the frozen protocol.

    Parameters
    ----------
    protocol : mapping of str to Any
        Parsed frozen scientific protocol.

    Returns
    -------
    dict of str to Any
        Exact dataset identity permitted at the public artifact boundary.
    """
    dataset = protocol["dataset"]
    train = dataset["splits"]["train"]
    return {
        "id": dataset["id"],
        "config": dataset["config"],
        "revision": dataset["revision"],
        "datasets_version": dataset["datasets_version"],
        "split": "train",
        "rows": train["rows"],
        "raw_parquet_sha256": train["raw_parquet_sha256"],
        "content_sha256": train["content_sha256"],
    }


def load_app_manifest(
    *, public_artifact_dir=None, protocol: Mapping[str, Any] | None = None
) -> AppManifest:
    """Load the public model and vocabulary manifest.

    Parameters
    ----------
    public_artifact_dir : str or pathlib.Path, optional
        Directory containing ``manifest.json`` and the shared vocabulary.
    protocol : mapping of str to Any or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    AppManifest
        Validated manifest, vocabulary bytes, and parsed-term snapshot.

    Raises
    ------
    ValueError
        If the public artifact path, schema, dataset, or vocabulary is invalid.
    """
    public_dir = resolve_public_artifact_dir(public_artifact_dir=public_artifact_dir)
    if public_dir.is_symlink() or not public_dir.is_dir():
        raise ValueError("public artifact directory must be a regular directory")
    canonical_dir = public_dir.resolve(strict=True)
    path = public_dir / "manifest.json"
    try:
        manifest_bytes = read_regular_file(path, parent=canonical_dir)
        payload = dict(
            validate_artifact_schema(
                json.loads(manifest_bytes.decode("utf-8")),
                "public manifest",
                supported_version=PUBLIC_ARTIFACT_SCHEMA_VERSION,
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid public manifest") from error

    # These are the only manifest values the product consumes.
    required = {
        "dataset",
        "embedding_dim",
        "sequence_length",
        "vocabulary_size",
        "vocabulary",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Manifest missing: {', '.join(sorted(missing))}")

    frozen = protocol or load_scientific_protocol()
    if payload["dataset"] != _expected_train_dataset(frozen):
        raise ValueError("public dataset identity differs from the frozen protocol")

    preprocessing = frozen["preprocessing"]
    if payload["vocabulary_size"] != preprocessing["vocabulary_size"]:
        raise ValueError("public vocabulary size differs from the frozen protocol")
    vocabulary = payload["vocabulary"]
    if not isinstance(vocabulary, Mapping) or set(vocabulary) != {
        "filename",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("public vocabulary contract is invalid")
    filename = vocabulary["filename"]
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).is_absolute()
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise ValueError("public vocabulary path must be a safe relative filename")
    if vocabulary["sha256"] != preprocessing["vocabulary_sha256"]:
        raise ValueError("public vocabulary SHA-256 differs from the frozen protocol")
    vocabulary_path = public_dir / filename
    vocabulary_bytes = read_regular_file(vocabulary_path, parent=canonical_dir)
    if type(vocabulary["size_bytes"]) is not int or vocabulary["size_bytes"] != len(
        vocabulary_bytes
    ):
        raise ValueError("public vocabulary byte length is invalid")
    if sha256_bytes(vocabulary_bytes) != f"sha256:{vocabulary['sha256']}":
        raise ValueError("public vocabulary checksum mismatch")
    try:
        vocabulary_text = vocabulary_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("public vocabulary is not valid UTF-8") from error
    if not vocabulary_text.endswith("\n"):
        raise ValueError("public vocabulary must end with an LF byte")
    vocabulary_terms = tuple(vocabulary_text[:-1].split("\n"))
    if len(vocabulary_terms) != payload["vocabulary_size"]:
        raise ValueError("public vocabulary term count is invalid")
    if vocabulary_terms[:2] != ("", "[UNK]"):
        raise ValueError("public vocabulary reserved tokens are invalid")
    return AppManifest(
        MappingProxyType(payload),
        vocabulary_path,
        manifest_bytes,
        vocabulary_bytes,
        vocabulary_terms,
    )
