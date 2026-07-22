"""Load the public vocabulary manifest used by all app entrypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.artifact_compatibility import (
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    canonical_json_bytes,
    deep_freeze,
    read_regular_file,
    sha256_bytes,
    validate_artifact_schema,
)
from src.evaluation_artifact import load_scientific_protocol
from src.paths import (
    default_public_artifact_dir,
    resolve_dir,
    resolve_prepared_artifact_dir,
)

PUBLIC_ARTIFACT_FILENAMES = frozenset({"manifest.json", "vocab.txt"})


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
    """Resolve configured public artifacts through the selected data generation.

    Parameters
    ----------
    config : mapping or None, optional
        Flower run configuration containing ``public-artifact-dir``.
    public_artifact_dir : str or pathlib.Path or None, optional
        Explicit logical public artifact root.

    Returns
    -------
    pathlib.Path
        Selected immutable public directory, or the legacy configured directory.

    Raises
    ------
    ValueError
        If the prepared-generation pointer or selected public directory is unsafe.
    """
    value = public_artifact_dir or configured_value(config, "public-artifact-dir")
    logical_dir = resolve_dir(value) if value else default_public_artifact_dir()
    return resolve_prepared_artifact_dir(logical_dir, "public")


def expected_train_dataset(
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact public train-dataset identity from the frozen protocol.

    Parameters
    ----------
    protocol : mapping of str to Any or None, optional
        Parsed frozen scientific protocol, primarily for deterministic tests.

    Returns
    -------
    dict of str to Any
        Exact dataset identity permitted at the public artifact boundary.
    """
    frozen = protocol or load_scientific_protocol()
    dataset = frozen["dataset"]
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


def protocol_model_dimensions(protocol: Mapping[str, Any]) -> dict[str, int]:
    """Return internally consistent positive dimensions from the frozen protocol.

    Parameters
    ----------
    protocol : mapping of str to Any
        Parsed frozen scientific protocol.

    Returns
    -------
    dict of str to int
        Public manifest field names mapped to registered model dimensions.

    Raises
    ------
    ValueError
        If dimensions are not built-in positive integers or model and
        preprocessing values disagree.
    """
    model = protocol["model"]
    preprocessing = protocol["preprocessing"]
    dimensions = {
        "vocabulary_size": model["vocabulary_size"],
        "sequence_length": model["sequence_length"],
        "embedding_dim": model["embedding_dimension"],
    }
    if any(type(value) is not int or value <= 0 for value in dimensions.values()):
        raise ValueError("frozen protocol model dimensions must be positive integers")
    if (
        dimensions["vocabulary_size"] != preprocessing["vocabulary_size"]
        or dimensions["vocabulary_size"] != preprocessing["max_tokens"]
        or dimensions["sequence_length"] != preprocessing["output_sequence_length"]
    ):
        raise ValueError("frozen model and preprocessing dimensions differ")
    return dimensions


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
    manifest_bytes = read_regular_file(path, parent=canonical_dir)
    try:
        validate_artifact_schema(
            json.loads(manifest_bytes.decode("utf-8")),
            "public manifest",
            supported_version=PUBLIC_ARTIFACT_SCHEMA_VERSION,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid public manifest") from error
    if {entry.name for entry in public_dir.iterdir()} != PUBLIC_ARTIFACT_FILENAMES:
        raise ValueError("public artifact contains unexpected files")
    vocabulary_path = public_dir / "vocab.txt"
    vocabulary_bytes = read_regular_file(vocabulary_path, parent=canonical_dir)
    return validate_app_manifest_bytes(
        manifest_bytes,
        vocabulary_bytes,
        vocabulary_path=vocabulary_path,
        protocol=protocol,
    )


def validate_app_manifest_bytes(
    manifest_bytes: bytes,
    vocabulary_bytes: bytes,
    *,
    vocabulary_path: Path,
    protocol: Mapping[str, Any] | None = None,
) -> AppManifest:
    """Validate retained public manifest and vocabulary bytes.

    Parameters
    ----------
    manifest_bytes : bytes
        Exact canonical ``manifest.json`` bytes.
    vocabulary_bytes : bytes
        Exact ``vocab.txt`` bytes bound by the manifest.
    vocabulary_path : pathlib.Path
        Original vocabulary path retained for diagnostics only.
    protocol : mapping of str to Any or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    AppManifest
        Validated immutable public artifact snapshot.

    Raises
    ------
    ValueError
        If the manifest, frozen protocol binding, or vocabulary is invalid.
    """
    try:
        payload = dict(
            validate_artifact_schema(
                json.loads(manifest_bytes.decode("utf-8")),
                "public manifest",
                supported_version=PUBLIC_ARTIFACT_SCHEMA_VERSION,
            )
        )
        canonical_manifest = canonical_json_bytes(payload)
        if manifest_bytes != canonical_manifest:
            raise ValueError("public manifest bytes are not canonical")
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
    if payload["dataset"] != expected_train_dataset(frozen):
        raise ValueError("public dataset identity differs from the frozen protocol")

    preprocessing = frozen["preprocessing"]
    dimensions = protocol_model_dimensions(frozen)
    for field, expected in dimensions.items():
        value = payload[field]
        if type(value) is not int or value <= 0 or value != expected:
            raise ValueError(
                f"public {field} differs from the frozen protocol dimensions"
            )
    vocabulary = payload["vocabulary"]
    if not isinstance(vocabulary, Mapping) or set(vocabulary) != {
        "filename",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("public vocabulary contract is invalid")
    filename = vocabulary["filename"]
    if not isinstance(filename, str) or filename != "vocab.txt":
        raise ValueError("public vocabulary filename must be vocab.txt")
    if vocabulary_path.name != filename:
        raise ValueError("public vocabulary path must end with vocab.txt")
    if vocabulary["sha256"] != preprocessing["vocabulary_sha256"]:
        raise ValueError("public vocabulary SHA-256 differs from the frozen protocol")
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
        deep_freeze(payload),
        vocabulary_path,
        manifest_bytes,
        vocabulary_bytes,
        vocabulary_terms,
    )
