"""Small helpers for splitting and packaging client data."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from src.artifact_compatibility import (
    CLIENT_SHARD_SCHEMA_VERSION,
    sha256_bytes,
)

DEFAULT_SPLIT_SEED = 67
DEFAULT_VALIDATION_SEED = 67
DEFAULT_DIRICHLET_ALPHA = 0.5


def label_histogram(labels: Any) -> dict[str, int]:
    """Count labels using stable string keys.

    Parameters
    ----------
    labels : Any
        Array-like label values.

    Returns
    -------
    dict of str to int
        Counts keyed by each label's string representation.
    """
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def canonical_client_row_bytes(row_id: str, text: str, label: int) -> bytes:
    """Serialize one client record under the shard checksum contract.

    Parameters
    ----------
    row_id : str
        Split-qualified official source-row identity.
    text : str
        Review text preserved exactly.
    label : int
        Binary sentiment label.

    Returns
    -------
    bytes
        Compact canonical UTF-8 JSON followed by one LF byte.
    """
    return (
        json.dumps(
            {"label": label, "row_id": row_id, "text": text},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def dirichlet_split(
    labels: Any,
    num_clients: int,
    alpha: float = DEFAULT_DIRICHLET_ALPHA,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[int, list[int]]:
    """Return deterministic, non-IID sample indices for each client."""
    rng = np.random.default_rng(seed)
    label_array = np.asarray(labels)
    partitions: dict[int, list[int]] = {
        client_id: [] for client_id in range(num_clients)
    }

    for label in np.unique(label_array):
        indices = np.flatnonzero(label_array == label)
        probabilities = rng.dirichlet(np.full(num_clients, alpha))
        clients = rng.choice(num_clients, len(indices), p=probabilities)
        for index, client_id in zip(indices, clients):
            partitions[int(client_id)].append(int(index))

    # Do not leak the label-loop ordering into packaged shard order. Client-side
    # splitting is stratified as a second line of defense.
    for shard_indices in partitions.values():
        rng.shuffle(shard_indices)

    return partitions


def client_shard_metadata(
    client_id: int,
    labels: Any,
    *,
    records_bytes: bytes,
    public_manifest_bytes: bytes,
    dataset: Mapping[str, Any],
    source_split: str = "train",
    row_identity: str = "train:{zero_based_official_split_row_index}",
    split_seed: int = DEFAULT_SPLIT_SEED,
    alpha: float = DEFAULT_DIRICHLET_ALPHA,
) -> dict[str, Any]:
    """Build authoritative metadata for one client-scoped shard.

    Parameters
    ----------
    client_id : int
        Client partition identifier.
    labels : Any
        Labels stored in the shard.
    records_bytes : bytes
        Exact canonical record bytes stored in ``reviews.jsonl``.
    public_manifest_bytes : bytes
        Exact canonical public manifest bytes used by the shard.
    dataset : mapping of str to Any
        Frozen official training-dataset identity.
    source_split : str, optional
        Official source split for every row.
    row_identity : str, optional
        Exact qualified row-ID contract.
    split_seed : int, optional
        Random seed used to produce the partition.
    alpha : float, optional
        Dirichlet concentration used to produce the partition.
    Returns
    -------
    dict of str to Any
        Complete client shard metadata.

    """
    return {
        "schema_version": CLIENT_SHARD_SCHEMA_VERSION,
        "artifact_type": "private_client_train_shard",
        "client_id": client_id,
        "dataset": dict(dataset),
        "source_split": source_split,
        "row_identity": row_identity,
        "split_seed": split_seed,
        "alpha": alpha,
        "sample_count": len(labels),
        "label_histogram": label_histogram(labels),
        "records": {
            "filename": "reviews.jsonl",
            "format": "canonical-jsonl",
            "encoding": "utf-8",
            "newline": "LF",
            "trailing_newline": True,
            "checksum": sha256_bytes(records_bytes),
        },
        "public_manifest": {
            "filename": "manifest.json",
            "size_bytes": len(public_manifest_bytes),
            "checksum": sha256_bytes(public_manifest_bytes),
        },
    }
