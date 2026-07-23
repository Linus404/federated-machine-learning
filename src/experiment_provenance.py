"""Finalize compact provenance for completed experiment cells."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.artifact_compatibility import read_regular_file, sha256_bytes
from src.evaluation_artifact import (
    EVALUATION_MANIFEST_FILENAME,
    EVALUATION_RECORDS_FILENAME,
    SCIENTIFIC_PROTOCOL_PATH,
    load_scientific_protocol,
)
from src.run_provenance import _code_revision

CLIENT_METADATA_FILENAME = "client_metadata.json"
CLIENT_REVIEWS_FILENAME = "reviews.jsonl"
PROVENANCE_FILENAME = "provenance.json"
PROVENANCE_CHECKSUM_FILENAME = "provenance.json.sha256"


def _canonical_bytes(value: Mapping[str, Any], protocol: Mapping[str, Any]) -> bytes:
    """Serialize a value with the frozen provenance JSON contract.

    Parameters
    ----------
    value : mapping of str to Any
        JSON-compatible provenance payload.
    protocol : mapping of str to Any
        Frozen scientific protocol.

    Returns
    -------
    bytes
        Canonical UTF-8 JSON followed by one newline.
    """
    config = protocol["provenance"]["canonical_serialization"]
    return (
        json.dumps(
            value,
            ensure_ascii=bool(config["ensure_ascii"]),
            sort_keys=bool(config["sort_keys"]),
            separators=tuple(config["separators"]),
            allow_nan=bool(config["allow_nan"]),
        ).encode(str(config["encoding"]))
        + b"\n"
    )


def _write_bytes_once(path: Path, content: bytes) -> None:
    """Publish bytes atomically without replacing an existing artifact.

    Parameters
    ----------
    path : pathlib.Path
        New artifact path.
    content : bytes
        Exact bytes to publish.

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If the destination already exists.
    """
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.chmod(0o644)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _verified_bytes(path: Path, expected: bytes | None = None) -> bytes:
    """Read contained regular bytes and optionally match a prior snapshot.

    Parameters
    ----------
    path : pathlib.Path
        Artifact path whose immediate parent is its containment boundary.
    expected : bytes or None, optional
        Previously validated immutable bytes.

    Returns
    -------
    bytes
        Exact artifact bytes.

    Raises
    ------
    ValueError
        If the file is unsafe or changed after validation.
    """
    content = read_regular_file(path, parent=path.parent)
    if expected is not None and content != expected:
        raise ValueError(f"input artifact changed after validation: {path.name}")
    return content


def _input_checksums(
    public_manifest: Any,
    evaluation_snapshot: Any,
    client_snapshots: Sequence[Any],
) -> dict[str, str]:
    """Return checksums for the exact validated experiment inputs.

    Parameters
    ----------
    public_manifest : Any
        Validated public application-manifest snapshot.
    evaluation_snapshot : Any
        Validated untouched-evaluation snapshot.
    client_snapshots : sequence of Any
        Validated client shard snapshots.

    Returns
    -------
    dict of str to str
        Sorted logical artifact paths mapped to prefixed SHA-256 digests.
    """
    vocabulary_path = Path(public_manifest.vocabulary_path)
    public_dir = vocabulary_path.parent
    evaluation_dir = Path(evaluation_snapshot.directory)
    checksums = {
        "public/manifest.json": sha256_bytes(
            _verified_bytes(
                public_dir / "manifest.json", public_manifest.manifest_bytes
            )
        ),
        f"public/{vocabulary_path.name}": sha256_bytes(
            _verified_bytes(vocabulary_path, public_manifest.vocabulary_bytes)
        ),
        f"evaluation/{EVALUATION_MANIFEST_FILENAME}": sha256_bytes(
            _verified_bytes(evaluation_dir / EVALUATION_MANIFEST_FILENAME)
        ),
        f"evaluation/{EVALUATION_RECORDS_FILENAME}": sha256_bytes(
            _verified_bytes(
                evaluation_dir / EVALUATION_RECORDS_FILENAME,
                evaluation_snapshot.records,
            )
        ),
    }
    for snapshot in client_snapshots:
        client_id = snapshot.metadata["client_id"]
        if type(client_id) is not int or client_id < 0:
            raise ValueError("client snapshot has an invalid client ID")
        directory = Path(snapshot.directory)
        prefix = f"clients/client-{client_id}"
        for filename, expected in (
            (CLIENT_METADATA_FILENAME, snapshot.metadata_bytes),
            (CLIENT_REVIEWS_FILENAME, snapshot.records_bytes),
        ):
            key = f"{prefix}/{filename}"
            if key in checksums:
                raise ValueError("client snapshot IDs must be unique")
            checksums[key] = sha256_bytes(
                _verified_bytes(directory / filename, expected)
            )
    return dict(sorted(checksums.items()))


def _output_checksums(output_dir: Path) -> dict[str, str]:
    """Hash every regular output while rejecting links and path escapes.

    Parameters
    ----------
    output_dir : pathlib.Path
        Canonical experiment-cell directory.

    Returns
    -------
    dict of str to str
        Sorted relative POSIX paths mapped to prefixed SHA-256 digests.

    Raises
    ------
    ValueError
        If an output is not a contained regular file or directory.
    """
    checksums: dict[str, str] = {}
    for root, directories, filenames in os.walk(output_dir, followlinks=False):
        root_path = Path(root)
        for name in directories:
            path = root_path / name
            if path.is_symlink() or not stat.S_ISDIR(
                path.stat(follow_symlinks=False).st_mode
            ):
                raise ValueError(f"output artifact directory is unsafe: {path}")
        for name in filenames:
            path = root_path / name
            relative = path.relative_to(output_dir).as_posix()
            if relative in {PROVENANCE_FILENAME, PROVENANCE_CHECKSUM_FILENAME}:
                continue
            checksums[relative] = sha256_bytes(_verified_bytes(path))
    return dict(sorted(checksums.items()))


def _derived_seeds(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect every derived seed object already exposed by a cell result.

    Parameters
    ----------
    result : mapping of str to Any
        Completed strategy-runner result.

    Returns
    -------
    list of dict of str to Any
        Seed records identified by their result-object paths.
    """
    records: list[dict[str, Any]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = (*path, str(key))
                if key == "seeds" and isinstance(item, Mapping):
                    for name, seed in item.items():
                        if type(seed) is not int or seed < 0:
                            raise ValueError("result contains an invalid derived seed")
                        records.append(
                            {"path": ".".join((*child, str(name))), "value": seed}
                        )
                else:
                    visit(item, child)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))

    visit(result, ())
    return sorted(records, key=lambda record: record["path"].encode("utf-8"))


def write_experiment_provenance(
    output_dir: str | Path,
    *,
    result: Mapping[str, Any],
    public_manifest: Any,
    evaluation_snapshot: Any,
    client_snapshots: Sequence[Any],
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    """Finalize provenance and its checksum for a completed strategy cell.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Completed cell output directory.
    result : mapping of str to Any
        Effective result returned by ``strategy_runner.run``.
    public_manifest : Any
        Validated public application-manifest snapshot used by the cell.
    evaluation_snapshot : Any
        Validated untouched-evaluation snapshot used by the cell.
    client_snapshots : sequence of Any
        Validated client shard snapshots used by the cell.
    protocol : mapping of str to Any or None, optional
        Frozen protocol already loaded by the runner.

    Returns
    -------
    pathlib.Path
        Newly published ``provenance.json`` path.

    Raises
    ------
    FileExistsError
        If provenance was already finalized.
    ValueError
        If result or artifact evidence is incomplete or unsafe.
    """
    directory = Path(output_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("experiment output directory must be a regular directory")
    directory = directory.resolve(strict=True)
    provenance_path = directory / PROVENANCE_FILENAME
    checksum_path = directory / PROVENANCE_CHECKSUM_FILENAME
    if provenance_path.exists() or provenance_path.is_symlink():
        raise FileExistsError(provenance_path)
    if checksum_path.exists() or checksum_path.is_symlink():
        raise FileExistsError(checksum_path)

    frozen = protocol or load_scientific_protocol()
    config = result.get("config")
    strategy = result.get("strategy")
    if not isinstance(config, Mapping) or not isinstance(strategy, str) or not strategy:
        raise ValueError("completed result must contain strategy and effective config")
    master_seed = config.get("seed")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("completed result must contain a non-negative master seed")

    protocol_bytes = _verified_bytes(SCIENTIFIC_PROTOCOL_PATH)
    dataset = frozen["dataset"]
    payload = {
        "schema": {
            "name": "federated-imdb-compact-experiment-provenance",
            "version": 1,
        },
        "code": _code_revision(),
        "config": {**dict(config), "strategy": strategy},
        "dataset": {
            "id": dataset["id"],
            "config": dataset["config"],
            "revision": dataset["revision"],
            "datasets_version": dataset["datasets_version"],
            "splits": {
                name: {
                    "raw_parquet_sha256": split["raw_parquet_sha256"],
                    "content_sha256": split["content_sha256"],
                }
                for name, split in sorted(dataset["splits"].items())
            },
        },
        "seeds": {"master": master_seed, "derived": _derived_seeds(result)},
        "protocol": {
            "path": SCIENTIFIC_PROTOCOL_PATH.relative_to(
                SCIENTIFIC_PROTOCOL_PATH.parents[1]
            ).as_posix(),
            "version": frozen["protocol_version"],
            "sha256": sha256_bytes(protocol_bytes),
        },
        "inputs": _input_checksums(
            public_manifest, evaluation_snapshot, client_snapshots
        ),
        "outputs": _output_checksums(directory),
    }
    content = _canonical_bytes(payload, frozen)
    checksum = hashlib.sha256(content).hexdigest()
    provenance_written = False
    checksum_written = False
    try:
        _write_bytes_once(provenance_path, content)
        provenance_written = True
        _write_bytes_once(
            checksum_path,
            f"{checksum}  {PROVENANCE_FILENAME}\n".encode("ascii"),
        )
        checksum_written = True
    except BaseException:
        if provenance_written:
            provenance_path.unlink(missing_ok=True)
        if checksum_written:
            checksum_path.unlink(missing_ok=True)
        raise
    return provenance_path
