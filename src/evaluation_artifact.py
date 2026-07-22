"""Publish and validate the fixed untouched global evaluation dataset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tomllib
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.artifact_compatibility import (
    canonical_json_bytes,
    deep_freeze,
    read_regular_file,
    require_secure_artifact_platform,
    sha256_bytes,
)
from src.paths import RunArtifactLock, resolve_prepared_artifact_dir

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


def _open_new_artifact_parent(output_dir: str | Path) -> tuple[Path, str, int]:
    """Open a symlink-free parent chain for a new evaluation artifact.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        New evaluation artifact directory.

    Returns
    -------
    tuple of pathlib.Path, str, and int
        Lexical absolute parent, destination basename, and held directory descriptor.

    Raises
    ------
    ValueError
        If any existing component is a symlink or unsafe path type.
    """
    candidate = Path(output_dir).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    output_path = Path(os.path.abspath(candidate))
    if output_path.name in {"", ".", ".."}:
        raise ValueError("evaluation artifact path must name a child directory")

    anchor = Path(output_path.anchor)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(anchor, flags)
    current = anchor
    try:
        relative_parts = output_path.parent.parts[len(anchor.parts) :]
        for component in relative_parts:
            try:
                component_stat = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                component_stat = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
            if not stat.S_ISDIR(component_stat.st_mode):
                raise ValueError(
                    "every existing evaluation artifact path component must be a "
                    "regular directory"
                )
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            current /= component
        return current, output_path.name, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _acquire_evaluation_lock(
    parent_descriptor: int, output_name: str
) -> RunArtifactLock:
    """Lock one evaluation destination through its validated parent descriptor.

    Parameters
    ----------
    parent_descriptor : int
        Held descriptor for the validated destination parent.
    output_name : str
        Direct child destination basename.

    Returns
    -------
    RunArtifactLock
        Exclusive nonblocking publication lock.

    Raises
    ------
    RuntimeError
        If another process owns the destination lock.
    ValueError
        If the lock entry is not a single-link regular file.
    """
    lock_name = f".{output_name}.run.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_name, flags, 0o600, dir_fd=parent_descriptor)
    except OSError as error:
        raise ValueError("evaluation artifact lock path is unsafe") from error
    lock_stat = os.fstat(descriptor)
    if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
        os.close(descriptor)
        raise ValueError("evaluation artifact lock path is unsafe")
    file = os.fdopen(descriptor, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if file.seek(0, os.SEEK_END) == 0:
                file.write(b"\0")
                file.flush()
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        file.close()
        raise RuntimeError(
            "Another evaluation artifact publication is already in progress"
        ) from error
    return RunArtifactLock(Path(lock_name), file)


def _destination_exists(parent_descriptor: int, output_name: str) -> bool:
    """Return whether a destination entry exists without following it.

    Parameters
    ----------
    parent_descriptor : int
        Held descriptor for the validated parent.
    output_name : str
        Direct child basename.

    Returns
    -------
    bool
        Whether any filesystem entry owns the name.
    """
    try:
        os.stat(output_name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _validate_row(
    text: object, label: object, *, split: str, index: int
) -> tuple[str, int]:
    """Validate one dataset row at a frozen split position.

    Parameters
    ----------
    text : object
        Candidate review text.
    label : object
        Candidate binary sentiment label.
    split : str
        Official split name used in validation errors.
    index : int
        Zero-based official split row index.

    Returns
    -------
    tuple of str and int
        Validated text and binary label.

    Raises
    ------
    ValueError
        If text is not a string or label is not a built-in binary integer.
    """
    if not isinstance(text, str):
        raise ValueError(f"{split} row {index} has a non-string text value")
    if type(label) is not int or label not in (0, 1):
        raise ValueError(f"{split} row {index} has an invalid binary label")
    return text, label


def _publish_evaluation_artifact_unlocked(
    rows: Iterable[Mapping[str, Any]],
    parent: Path,
    output_name: str,
    parent_descriptor: int,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    """Publish the official test split into a new evaluation-only directory.

    Parameters
    ----------
    rows : iterable of mappings
        Official test rows in ascending source order.
    parent : pathlib.Path
        Validated lexical parent retained for the returned diagnostic path.
    output_name : str
        Direct child destination basename.
    parent_descriptor : int
        Held descriptor anchoring all mutation to the validated parent.
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
    output_path = parent / output_name
    if _destination_exists(parent_descriptor, output_name):
        raise FileExistsError(
            f"evaluation artifact path already exists; refusing replacement: {output_path}"
        )
    staging_name = f".{output_name}.{uuid.uuid4().hex}.staging"
    os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
    staging_descriptor = os.open(
        staging_name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    record_hash = hashlib.sha256()
    content_hash = hashlib.sha256()
    label_counts: Counter[int] = Counter()
    row_count = 0
    try:
        records_descriptor = os.open(
            EVALUATION_RECORDS_FILENAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=staging_descriptor,
        )
        with os.fdopen(records_descriptor, "wb") as file:
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
        os.chmod(
            EVALUATION_RECORDS_FILENAME,
            0o644,
            dir_fd=staging_descriptor,
            follow_symlinks=False,
        )

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
        manifest_descriptor = os.open(
            EVALUATION_MANIFEST_FILENAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=staging_descriptor,
        )
        with os.fdopen(manifest_descriptor, "wb") as file:
            file.write(canonical_json_bytes(manifest))
            file.flush()
            os.fsync(file.fileno())
        os.fsync(staging_descriptor)
        staged_stat = os.stat(
            staging_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        opened_stat = os.fstat(staging_descriptor)
        if not stat.S_ISDIR(staged_stat.st_mode) or (
            staged_stat.st_dev,
            staged_stat.st_ino,
        ) != (opened_stat.st_dev, opened_stat.st_ino):
            raise ValueError("evaluation staging directory changed during publication")
        os.rename(
            staging_name,
            output_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
        return output_path
    except BaseException:
        try:
            shutil.rmtree(staging_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(staging_descriptor)


def publish_evaluation_artifact(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    """Publish one immutable evaluation artifact under an exclusive lock.

    Parameters
    ----------
    rows : iterable of mappings
        Official test rows in ascending source order.
    output_dir : str or pathlib.Path
        New dedicated artifact directory.
    protocol : mapping or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    pathlib.Path
        Published evaluation artifact directory.

    Raises
    ------
    FileExistsError
        If the immutable destination already exists.
    RuntimeError
        If another writer owns the same destination or direct publication runs
        outside the supported Linux platform.
    ValueError
        If owned residue, rows, or paths are invalid.
    """
    require_secure_artifact_platform()
    parent, output_name, parent_descriptor = _open_new_artifact_parent(output_dir)
    lock = _acquire_evaluation_lock(parent_descriptor, output_name)
    try:
        prefix = f".{output_name}."
        for residue_name in os.listdir(parent_descriptor):
            if not residue_name.startswith(prefix) or not residue_name.endswith(
                ".staging"
            ):
                continue
            residue_stat = os.stat(
                residue_name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if not stat.S_ISDIR(residue_stat.st_mode):
                raise ValueError("evaluation staging residue is unsafe")
            shutil.rmtree(residue_name, dir_fd=parent_descriptor)
        return _publish_evaluation_artifact_unlocked(
            rows,
            parent,
            output_name,
            parent_descriptor,
            protocol=protocol,
        )
    finally:
        lock.release()
        os.close(parent_descriptor)


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
    """Validate the exact untouched-evaluation manifest contract.

    Parameters
    ----------
    payload : object
        Decoded candidate manifest.
    protocol : mapping of str to Any
        Frozen protocol providing the authoritative test identity.

    Returns
    -------
    mapping of str to Any
        Validated manifest mapping.

    Raises
    ------
    ValueError
        If schema, fields, lifecycle, dataset, records, or checksums differ.
    """
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
    directory = resolve_prepared_artifact_dir(artifact_dir, "evaluation")
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
        deep_freeze(payload),
        records,
    )
