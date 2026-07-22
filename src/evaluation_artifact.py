"""Publish and validate the fixed untouched global evaluation dataset."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from src.artifact_compatibility import (
    read_regular_file,
    sha256_bytes,
    write_json_atomically,
)
from src.paths import resolve_dir

EVALUATION_ARTIFACT_SCHEMA_VERSION = 1
EVALUATION_MANIFEST_FILENAME = "manifest.json"
EVALUATION_RECORDS_FILENAME = "test.jsonl"
SCIENTIFIC_PROTOCOL_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "scientific-protocol-v1.toml"
)
_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_type",
    "lifecycle",
    "dataset",
    "records",
    "checksums",
}
_DATASET_FIELDS = {
    "id",
    "config",
    "revision",
    "datasets_version",
    "split",
    "rows",
    "label_counts",
    "raw_parquet_sha256",
    "content_sha256",
}
_RECORD_FIELDS = {
    "filename",
    "format",
    "encoding",
    "newline",
    "trailing_newline",
    "row_count",
    "fields",
    "row_identity",
    "order",
}


@dataclass(frozen=True)
class EvaluationArtifactSnapshot:
    """Validated immutable bytes for the untouched evaluation dataset.

    Parameters
    ----------
    directory : pathlib.Path
        Canonical evaluation artifact directory.
    manifest : mapping of str to Any
        Strictly validated manifest payload.
    records : bytes
        Canonical JSONL bytes verified against the manifest and frozen protocol.
    """

    directory: Path
    manifest: Mapping[str, Any]
    records: bytes


def load_scientific_protocol() -> Mapping[str, Any]:
    """Load the frozen scientific protocol from the repository.

    Returns
    -------
    collections.abc.Mapping
        Parsed frozen protocol.

    Raises
    ------
    ValueError
        If the protocol is not the frozen version used by this artifact schema.
    """
    protocol = tomllib.loads(SCIENTIFIC_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != 1 or protocol.get("status") != "frozen":
        raise ValueError("scientific protocol must be frozen protocol version 1")
    return protocol


def canonical_source_row_bytes(text: str, label: int) -> bytes:
    """Serialize one upstream row under the frozen content-hash contract.

    Parameters
    ----------
    text : str
        Review text preserved exactly as supplied by the dataset.
    label : int
        Dataset label.

    Returns
    -------
    bytes
        Compact UTF-8 JSON followed by one LF byte.
    """
    return (
        json.dumps(
            {"label": label, "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_evaluation_row_bytes(index: int, text: str, label: int) -> bytes:
    """Serialize one split-qualified untouched evaluation record.

    Parameters
    ----------
    index : int
        Zero-based official test-split row index.
    text : str
        Review text preserved exactly as supplied by the dataset.
    label : int
        Binary sentiment label.

    Returns
    -------
    bytes
        Compact UTF-8 JSON followed by one LF byte.
    """
    return (
        json.dumps(
            {"label": label, "row_id": f"test:{index}", "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _evaluation_dataset_manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen test-dataset identity for an evaluation manifest.

    Parameters
    ----------
    protocol : mapping of str to Any
        Parsed frozen scientific protocol.

    Returns
    -------
    dict of str to Any
        Exact test-split identity permitted at the evaluation boundary.
    """
    dataset = protocol["dataset"]
    split = dataset["splits"]["test"]
    return {
        "id": dataset["id"],
        "config": dataset["config"],
        "revision": dataset["revision"],
        "datasets_version": dataset["datasets_version"],
        "split": "test",
        "rows": split["rows"],
        "label_counts": split["label_counts"],
        "raw_parquet_sha256": split["raw_parquet_sha256"],
        "content_sha256": split["content_sha256"],
    }


def _validate_new_artifact_path(output_dir: str | Path) -> Path:
    """Validate and prepare a new immutable evaluation artifact path.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        New evaluation artifact directory.

    Returns
    -------
    pathlib.Path
        Canonical destination beneath a real parent directory.

    Raises
    ------
    FileExistsError
        If the destination already exists or is a symlink.
    ValueError
        If the destination parent is not a regular directory.
    """
    output_path = resolve_dir(output_dir)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(
            f"evaluation artifact path already exists; refusing replacement: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parent = output_path.parent.resolve(strict=True)
    if output_path.parent.is_symlink() or not parent.is_dir():
        raise ValueError("evaluation artifact parent must be a regular directory")
    return parent / output_path.name


def _validate_row(
    text: object, label: object, *, split: str, index: int
) -> tuple[str, int]:
    if not isinstance(text, str):
        raise ValueError(f"{split} row {index} has a non-string text value")
    if type(label) is not int or label not in (0, 1):
        raise ValueError(f"{split} row {index} has an invalid binary label")
    return text, label


def publish_evaluation_artifact(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    """Publish the official test split into a new evaluation-only directory.

    Parameters
    ----------
    rows : iterable of mappings
        Official test rows in ascending source order.
    output_dir : str or pathlib.Path
        New dedicated artifact directory. Existing paths are never replaced.
    protocol : mapping or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    pathlib.Path
        Published evaluation artifact directory.

    Raises
    ------
    FileExistsError
        If the destination already exists.
    ValueError
        If rows or publication paths violate the frozen artifact contract.
    """
    frozen = protocol or load_scientific_protocol()
    dataset_manifest = _evaluation_dataset_manifest(frozen)
    output_path = _validate_new_artifact_path(output_dir)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".jsonl.tmp",
    )
    temporary_path = Path(temporary_name)
    record_hash = hashlib.sha256()
    content_hash = hashlib.sha256()
    label_counts: Counter[int] = Counter()
    row_count = 0
    created_directory = False
    try:
        with os.fdopen(descriptor, "wb") as file:
            for index, row in enumerate(rows):
                text, label = _validate_row(
                    row.get("text"), row.get("label"), split="test", index=index
                )
                content_hash.update(canonical_source_row_bytes(text, label))
                record = canonical_evaluation_row_bytes(index, text, label)
                file.write(record)
                record_hash.update(record)
                label_counts[label] += 1
                row_count += 1
            file.flush()
            os.fsync(file.fileno())
        temporary_path.chmod(0o644)

        expected_counts = dataset_manifest["label_counts"]
        if row_count != dataset_manifest["rows"]:
            raise ValueError(
                f"test row count mismatch: expected {dataset_manifest['rows']}, got {row_count}"
            )
        if [label_counts[0], label_counts[1]] != expected_counts:
            raise ValueError("test label counts differ from the frozen protocol")
        if content_hash.hexdigest() != dataset_manifest["content_sha256"]:
            raise ValueError(
                "test canonical content SHA-256 differs from the frozen protocol"
            )

        # The manifest is the atomic completion marker; consumers reject the
        # directory until the already-fsynced records and this manifest both exist.
        output_path.mkdir(mode=0o755)
        created_directory = True
        os.replace(temporary_path, output_path / EVALUATION_RECORDS_FILENAME)
        manifest = {
            "schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "untouched_global_test_set",
            "lifecycle": "complete",
            "dataset": dataset_manifest,
            "records": {
                "filename": EVALUATION_RECORDS_FILENAME,
                "format": "canonical-jsonl",
                "encoding": "utf-8",
                "newline": "LF",
                "trailing_newline": True,
                "row_count": row_count,
                "fields": ["label", "row_id", "text"],
                "row_identity": "{split}:{zero_based_official_split_row_index}",
                "order": "ascending_zero_based_official_split_row_index",
            },
            "checksums": {
                EVALUATION_RECORDS_FILENAME: f"sha256:{record_hash.hexdigest()}"
            },
        }
        write_json_atomically(
            output_path / EVALUATION_MANIFEST_FILENAME,
            manifest,
            overwrite=False,
        )
        return output_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        if created_directory:
            for filename in (EVALUATION_MANIFEST_FILENAME, EVALUATION_RECORDS_FILENAME):
                (output_path / filename).unlink(missing_ok=True)
            output_path.rmdir()
        raise


def _require_exact_fields(
    payload: Mapping[str, Any], expected: set[str], name: str
) -> None:
    """Require an evaluation manifest object to contain exactly known fields.

    Parameters
    ----------
    payload : mapping of str to Any
        Manifest object whose keys are validated.
    expected : set of str
        Complete permitted field set.
    name : str
        Human-readable object name used in errors.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the object contains missing or additional fields.
    """
    if set(payload) != expected:
        raise ValueError(f"evaluation {name} has an invalid field set")


def _validate_manifest(
    payload: object, protocol: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation manifest must be a JSON object")
    _require_exact_fields(payload, _MANIFEST_FIELDS, "manifest")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != EVALUATION_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("evaluation manifest has an unsupported schema_version")
    if payload["artifact_type"] != "untouched_global_test_set":
        raise ValueError("evaluation manifest has an invalid artifact_type")
    if payload["lifecycle"] != "complete":
        raise ValueError("evaluation manifest is not complete")

    dataset = payload["dataset"]
    records = payload["records"]
    checksums = payload["checksums"]
    if not isinstance(dataset, Mapping):
        raise ValueError("evaluation manifest dataset must be an object")
    if not isinstance(records, Mapping):
        raise ValueError("evaluation manifest records must be an object")
    if not isinstance(checksums, Mapping):
        raise ValueError("evaluation manifest checksums must be an object")
    _require_exact_fields(dataset, _DATASET_FIELDS, "dataset")
    _require_exact_fields(records, _RECORD_FIELDS, "records")
    if dict(dataset) != _evaluation_dataset_manifest(protocol):
        raise ValueError("evaluation dataset identity differs from the frozen protocol")
    expected_records = {
        "filename": EVALUATION_RECORDS_FILENAME,
        "format": "canonical-jsonl",
        "encoding": "utf-8",
        "newline": "LF",
        "trailing_newline": True,
        "row_count": dataset["rows"],
        "fields": ["label", "row_id", "text"],
        "row_identity": "{split}:{zero_based_official_split_row_index}",
        "order": "ascending_zero_based_official_split_row_index",
    }
    if dict(records) != expected_records:
        raise ValueError("evaluation record contract is invalid")
    if set(checksums) != {EVALUATION_RECORDS_FILENAME} or not isinstance(
        checksums[EVALUATION_RECORDS_FILENAME], str
    ):
        raise ValueError("evaluation manifest checksums are invalid")
    return payload


def load_evaluation_artifact_snapshot(
    artifact_dir: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> EvaluationArtifactSnapshot:
    """Read and fully verify an untouched evaluation artifact snapshot.

    Parameters
    ----------
    artifact_dir : str or pathlib.Path
        Directory containing the evaluation manifest and canonical records.
    protocol : mapping or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    EvaluationArtifactSnapshot
        Manifest and record bytes verified in one pass.

    Raises
    ------
    ValueError
        If any path, byte, identity, ordering, or checksum is invalid.
    """
    directory = resolve_dir(artifact_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("evaluation artifact directory must be a regular directory")
    canonical_dir = directory.resolve(strict=True)
    frozen = protocol or load_scientific_protocol()
    try:
        manifest_bytes = read_regular_file(
            directory / EVALUATION_MANIFEST_FILENAME, parent=canonical_dir
        )
        decoded_manifest = json.loads(manifest_bytes.decode("utf-8"))
        payload = _validate_manifest(decoded_manifest, frozen)
        canonical_manifest = (
            json.dumps(decoded_manifest, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        if manifest_bytes != canonical_manifest:
            raise ValueError("evaluation manifest bytes are not canonical")
        if {path.name for path in directory.iterdir()} != {
            EVALUATION_MANIFEST_FILENAME,
            EVALUATION_RECORDS_FILENAME,
        }:
            raise ValueError("evaluation artifact contains unexpected files")
        records = read_regular_file(
            directory / EVALUATION_RECORDS_FILENAME, parent=canonical_dir
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("invalid evaluation artifact") from error

    expected_checksum = payload["checksums"][EVALUATION_RECORDS_FILENAME]
    if sha256_bytes(records) != expected_checksum:
        raise ValueError("evaluation record checksum mismatch")

    content_hash = hashlib.sha256()
    label_counts: Counter[int] = Counter()
    lines = records.splitlines(keepends=True)
    if len(lines) != payload["records"]["row_count"]:
        raise ValueError("evaluation record row count mismatch")
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            raise ValueError("evaluation records must end every row with LF")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid evaluation record at row {index}") from error
        if not isinstance(row, Mapping) or set(row) != {"label", "row_id", "text"}:
            raise ValueError(f"invalid evaluation record fields at row {index}")
        text, label = _validate_row(
            row["text"], row["label"], split="test", index=index
        )
        if row["row_id"] != f"test:{index}":
            raise ValueError(
                f"evaluation row identity or order mismatch at row {index}"
            )
        if line != canonical_evaluation_row_bytes(index, text, label):
            raise ValueError(f"evaluation record is not canonical at row {index}")
        content_hash.update(canonical_source_row_bytes(text, label))
        label_counts[label] += 1

    dataset = payload["dataset"]
    if [label_counts[0], label_counts[1]] != dataset["label_counts"]:
        raise ValueError("evaluation label counts differ from the manifest")
    if content_hash.hexdigest() != dataset["content_sha256"]:
        raise ValueError("evaluation content SHA-256 differs from the manifest")
    return EvaluationArtifactSnapshot(
        canonical_dir,
        MappingProxyType(dict(payload)),
        records,
    )
