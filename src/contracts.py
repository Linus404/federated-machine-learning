"""Small helpers for splitting and packaging client data."""

from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

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
    labels = np.asarray(labels)
    partitions = {client_id: [] for client_id in range(num_clients)}

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        probabilities = rng.dirichlet(np.full(num_clients, alpha))
        clients = rng.choice(num_clients, len(indices), p=probabilities)
        for index, client_id in zip(indices, clients):
            partitions[int(client_id)].append(int(index))

    # Do not leak the label-loop ordering into packaged shard order. Client-side
    # splitting is stratified as a second line of defense.
    for indices in partitions.values():
        rng.shuffle(indices)

    return partitions


def client_shard_metadata(
    client_id: int,
    labels: Any,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    alpha: float = DEFAULT_DIRICHLET_ALPHA,
    manifest_checksum: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
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


def create_client_shard_archive(
    client_id: int,
    source_dir: str | Path,
    output_dir: str | Path,
    manifest: dict[str, Any],
) -> Path:
    """Archive one client's raw reviews and metadata."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"client-{client_id}.tar.gz"
    metadata = json.dumps(manifest, indent=2).encode()

    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_dir / "reviews.jsonl", arcname="reviews.jsonl")
        info = tarfile.TarInfo("client_metadata.json")
        info.size = len(metadata)
        archive.addfile(info, BytesIO(metadata))

    return archive_path
