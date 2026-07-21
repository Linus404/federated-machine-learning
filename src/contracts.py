"""Small helpers for splitting and packaging client data."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

DEFAULT_SPLIT_SEED = 67
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
    metadata = {
        "client_id": client_id,
        "split_seed": split_seed,
        "alpha": alpha,
        "sample_count": len(labels),
        "label_histogram": label_histogram(labels),
        "manifest_checksum": manifest_checksum,
    }
    metadata.update(extra_metadata or {})
    return metadata
