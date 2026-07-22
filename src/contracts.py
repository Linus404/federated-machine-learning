"""Small helpers for splitting and packaging client data."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.artifact_compatibility import ARTIFACT_SCHEMA_VERSION

DEFAULT_SPLIT_SEED = 67
DEFAULT_VALIDATION_SEED = 67
DEFAULT_DIRICHLET_ALPHA = 0.5


def label_histogram(labels: Any) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


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
    split_seed: int = DEFAULT_SPLIT_SEED,
    alpha: float = DEFAULT_DIRICHLET_ALPHA,
    manifest_checksum: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build authoritative metadata for one client-scoped shard.

    Parameters
    ----------
    client_id : int
        Client partition identifier.
    labels : Any
        Labels stored in the shard.
    split_seed : int, optional
        Random seed used to produce the partition.
    alpha : float, optional
        Dirichlet concentration used to produce the partition.
    manifest_checksum : str or None, optional
        Checksum tying the shard to its public manifest.
    extra_metadata : mapping or None, optional
        Additional fields that do not replace schema-owned fields.

    Returns
    -------
    dict of str to Any
        Complete client shard metadata.

    Raises
    ------
    ValueError
        If additional metadata tries to replace a schema-owned field.
    """
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "client_id": client_id,
        "split_seed": split_seed,
        "alpha": alpha,
        "sample_count": len(labels),
        "label_histogram": label_histogram(labels),
        "manifest_checksum": manifest_checksum,
    }
    collisions = metadata.keys() & (extra_metadata or {}).keys()
    if collisions:
        raise ValueError(
            "extra client shard metadata cannot replace reserved fields: "
            + ", ".join(sorted(collisions))
        )
    metadata.update(extra_metadata or {})
    return metadata
