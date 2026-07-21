from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

# TensorFlow/Keras read these before import; set them before Keras loads.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np

from src.artifact_compatibility import ARTIFACT_SCHEMA_VERSION
from src.contracts import (
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_SPLIT_SEED,
    client_shard_metadata,
    dirichlet_split,
    label_histogram,
)
from src.paths import (
    default_public_artifact_dir,
    resolve_dir,
)

DEFAULT_MAX_TOKENS = 20_000
DEFAULT_SEQUENCE_LENGTH = 500
DEFAULT_EMBEDDING_DIM = 100


def load_imdb_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the public IMDB training split once."""
    from datasets import load_dataset

    dataset = load_dataset("stanfordnlp/imdb")
    texts = np.asarray(dataset["train"]["text"])
    labels = np.asarray(dataset["train"]["label"], dtype="int32")
    return texts, labels


def build_vectorizer(texts: Any) -> Any:
    """Build the shared public token-ID contract."""
    import keras

    # Strip markup before vectorization so tokens represent review text, not HTML tags.
    clean_texts = np.asarray([re.sub(r"<[^>]+>", " ", text) for text in texts])

    vectorizer = keras.layers.TextVectorization(
        max_tokens=DEFAULT_MAX_TOKENS,
        output_sequence_length=DEFAULT_SEQUENCE_LENGTH,
        dtype="int32",
    )
    vectorizer.adapt(clean_texts)
    return vectorizer


def _checksum(payload: Mapping[str, Any]) -> str:
    content = json.dumps(payload, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(content).hexdigest()


def package_raw_client_shards(
    texts: Any,
    labels: Any,
    output_dir: str | Path,
    num_clients: int,
    *,
    alpha: float = DEFAULT_DIRICHLET_ALPHA,
    seed: int = DEFAULT_SPLIT_SEED,
    manifest: Mapping[str, Any] | None = None,
    manifest_checksum: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write raw review text and labels into one directory per client.

    Parameters
    ----------
    texts : Any
        Review texts to partition.
    labels : Any
        Labels corresponding to ``texts``.
    output_dir : str or pathlib.Path
        Parent directory for client shards.
    num_clients : int
        Number of client shards to create.
    alpha : float, optional
        Dirichlet concentration parameter.
    seed : int, optional
        Random seed for deterministic partitioning.
    manifest : mapping, optional
        Public manifest used to derive the shard checksum.
    manifest_checksum : str, optional
        Precomputed public manifest checksum.
    metadata : mapping, optional
        Additional metadata copied into every shard.

    Returns
    -------
    list of pathlib.Path
        Created client shard directories.
    """
    text_array = np.asarray(texts)
    label_array = np.asarray(labels)
    if len(text_array) != len(label_array):
        raise ValueError(
            "text/label sample count mismatch: "
            f"len(texts)={len(text_array)} len(labels)={len(label_array)}"
        )

    output_path = resolve_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for legacy_archive in output_path.glob("client-*.tar.gz"):
        legacy_archive.unlink()
    checksum = manifest_checksum or _checksum(
        manifest
        or {
            "labels": label_histogram(label_array),
            "clients": num_clients,
            "seed": seed,
        }
    )

    split = dirichlet_split(
        label_array,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    shard_paths: list[Path] = []
    for client_id in range(num_clients):
        indices = split[client_id]
        client_texts = text_array[indices]
        client_labels = label_array[indices]
        shard_dir = output_path / f"client-{client_id}"
        shard_dir.mkdir(exist_ok=True)
        with (shard_dir / "reviews.jsonl").open("w", encoding="utf-8") as file:
            for text, label in zip(client_texts, client_labels):
                file.write(
                    json.dumps(
                        {"text": str(text), "label": int(label)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        shard_metadata = client_shard_metadata(
            client_id,
            client_labels,
            split_seed=seed,
            alpha=alpha,
            manifest_checksum=checksum,
            extra_metadata=metadata,
        )
        (shard_dir / "client_metadata.json").write_text(
            json.dumps(shard_metadata, indent=2), encoding="utf-8"
        )
        shard_paths.append(shard_dir)

    return shard_paths


def publish_public_artifacts(vectorizer: Any, output_dir: str | Path) -> dict[str, Any]:
    """Publish the vocabulary and small model/tokenizer manifest."""
    output_path = resolve_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vocabulary = vectorizer.get_vocabulary()
    (output_path / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "vocabulary_size": len(vocabulary),
        "vocabulary": {"filename": "vocab.txt"},
        "provenance": "Keras TextVectorization adapted on public IMDB training data",
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def prepare_all(
    partitions: int,
    client_shard_dir: str | Path,
    public_artifact_dir: str | Path,
) -> None:
    """Create raw private client shards and shared public artifacts."""
    texts, labels = load_imdb_training_data()
    vectorizer = build_vectorizer(texts)
    manifest = publish_public_artifacts(vectorizer, public_artifact_dir)
    package_raw_client_shards(
        texts,
        labels,
        client_shard_dir,
        num_clients=partitions,
        manifest=manifest,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare sentiment data partitions.")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument(
        "--client-shard-dir", type=Path, default=Path("artifacts/clients")
    )
    parser.add_argument(
        "--public-artifact-dir", type=Path, default=default_public_artifact_dir()
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prepare_all(
        partitions=args.partitions,
        client_shard_dir=args.client_shard_dir,
        public_artifact_dir=args.public_artifact_dir,
    )


if __name__ == "__main__":
    main()
