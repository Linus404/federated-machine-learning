from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

# TensorFlow/Keras read these before import; set them before Keras loads.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np

from src.artifact_compatibility import ARTIFACT_SCHEMA_VERSION, sha256_file
from src.contracts import (
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_SPLIT_SEED,
    client_shard_metadata,
    dirichlet_split,
    label_histogram,
)
from src.evaluation_artifact import (
    canonical_source_row_bytes,
    load_scientific_protocol,
    publish_evaluation_artifact,
)
from src.paths import (
    default_evaluation_artifact_dir,
    default_public_artifact_dir,
    resolve_dir,
)

DEFAULT_MAX_TOKENS = 20_000
DEFAULT_SEQUENCE_LENGTH = 500
DEFAULT_EMBEDDING_DIM = 100


def _raw_split_path(
    dataset_id: str,
    config: str,
    revision: str,
    split: str,
    *,
    local_files_only: bool,
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=dataset_id,
            filename=f"{config}/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def _validate_loaded_dataset(
    dataset: Any,
    protocol: Mapping[str, Any],
    raw_paths: Mapping[str, Path],
) -> None:
    dataset_spec = protocol["dataset"]
    split_specs = dataset_spec["splits"]
    if set(dataset) != set(split_specs):
        raise ValueError("loaded dataset splits differ from the frozen protocol")

    for split, split_spec in split_specs.items():
        raw_digest = sha256_file(raw_paths[split]).removeprefix("sha256:")
        if raw_digest != split_spec["raw_parquet_sha256"]:
            raise ValueError(f"{split} raw Parquet SHA-256 differs from the protocol")

        loaded_split = dataset[split]
        info = loaded_split.info
        if (
            info.dataset_name != dataset_spec["id"].split("/", maxsplit=1)[1]
            or info.config_name != dataset_spec["config"]
            or loaded_split.column_names != ["text", "label"]
        ):
            raise ValueError("loaded dataset identity differs from the frozen protocol")
        if len(loaded_split) != split_spec["rows"]:
            raise ValueError(f"{split} row count differs from the frozen protocol")

        content_hash = hashlib.sha256()
        label_counts: Counter[int] = Counter()
        label_ranges = split_spec.get("label_index_ranges")
        for index, row in enumerate(loaded_split):
            text = row.get("text")
            label = row.get("label")
            if not isinstance(text, str) or type(label) is not int:
                raise ValueError(f"{split} row {index} has invalid text or label data")
            if label_ranges is not None:
                expected_label = next(
                    (
                        range_label
                        for range_label, (start, end) in enumerate(label_ranges)
                        if start <= index <= end
                    ),
                    None,
                )
                if label != expected_label:
                    raise ValueError(
                        f"{split} labels differ from the frozen official row ranges"
                    )
            content_hash.update(canonical_source_row_bytes(text, label))
            label_counts[label] += 1

        expected_counts = split_spec.get("label_counts")
        if (
            expected_counts is not None
            and [label_counts[label] for label in range(len(expected_counts))]
            != expected_counts
        ):
            raise ValueError(f"{split} label counts differ from the frozen protocol")
        if content_hash.hexdigest() != split_spec["content_sha256"]:
            raise ValueError(
                f"{split} canonical content SHA-256 differs from the frozen protocol"
            )


def load_verified_imdb_dataset(*, local_files_only: bool = False) -> Any:
    """Load and verify every frozen IMDB split before artifact preparation.

    Parameters
    ----------
    local_files_only : bool, optional
        Require all dataset metadata and bytes to be available in local caches.

    Returns
    -------
    datasets.DatasetDict
        Verified official train, test, and unsupervised splits.

    Raises
    ------
    ValueError
        If package version, identity, rows, labels, or checksums differ from the
        frozen scientific protocol.
    """
    from datasets import DownloadConfig, load_dataset

    protocol = load_scientific_protocol()
    dataset_spec = protocol["dataset"]
    installed_version = importlib.metadata.version("datasets")
    if installed_version != dataset_spec["datasets_version"]:
        raise ValueError(
            "datasets version differs from the frozen protocol: "
            f"expected {dataset_spec['datasets_version']}, got {installed_version}"
        )

    raw_paths = {
        split: _raw_split_path(
            dataset_spec["id"],
            dataset_spec["config"],
            dataset_spec["revision"],
            split,
            local_files_only=local_files_only,
        )
        for split in dataset_spec["splits"]
    }
    dataset = load_dataset(
        dataset_spec["id"],
        dataset_spec["config"],
        revision=dataset_spec["revision"],
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    _validate_loaded_dataset(dataset, protocol, raw_paths)
    return dataset


def _validated_frameworks() -> tuple[Any, Any, Mapping[str, Any]]:
    """Load frameworks only when their versions match the frozen protocol.

    Returns
    -------
    tuple
        Keras, TensorFlow, and the parsed frozen protocol.

    Raises
    ------
    ValueError
        If any registered framework version differs from the runtime.
    """
    import keras
    import tensorflow as tf

    protocol = load_scientific_protocol()
    framework = protocol["framework"]
    installed_versions = {
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
    }
    expected_versions = {
        "tensorflow": framework["tensorflow_version"],
        "keras": framework["keras_version"],
        "numpy": framework["numpy_version"],
    }
    mismatches = [
        f"{name}: expected {expected_versions[name]}, got {installed_versions[name]}"
        for name in expected_versions
        if installed_versions[name] != expected_versions[name]
    ]
    if mismatches:
        raise ValueError(
            "framework versions differ from the frozen protocol: "
            + "; ".join(mismatches)
        )
    return keras, tf, protocol


def build_vectorizer(texts: Any) -> Any:
    """Build and verify the shared public token-ID contract.

    Parameters
    ----------
    texts : Any
        Complete official training texts in ascending row-index order.

    Returns
    -------
    keras.layers.TextVectorization
        Adapted vectorizer matching the frozen vocabulary contract.
    """
    keras, tf, protocol = _validated_frameworks()

    def protocol_standardize(text):
        text = tf.strings.regex_replace(text, "<[^>]+>", " ")
        text = tf.strings.lower(text)
        return tf.strings.regex_replace(
            text, "[" + re.escape(string.punctuation) + "]", ""
        )

    vectorizer = keras.layers.TextVectorization(
        max_tokens=DEFAULT_MAX_TOKENS,
        standardize=protocol_standardize,
        split="whitespace",
        output_mode="int",
        output_sequence_length=DEFAULT_SEQUENCE_LENGTH,
        pad_to_max_tokens=False,
    )
    vectorizer.adapt(
        tf.data.Dataset.from_tensor_slices(list(texts)).batch(256, drop_remainder=False)
    )
    preprocessing = protocol["preprocessing"]
    vocabulary = vectorizer.get_vocabulary()
    vocabulary_bytes = b"".join(item.encode("utf-8") + b"\n" for item in vocabulary)
    if len(vocabulary) != preprocessing["vocabulary_size"]:
        raise ValueError("vocabulary size differs from the frozen protocol")
    if (
        hashlib.sha256(vocabulary_bytes).hexdigest()
        != preprocessing["vocabulary_sha256"]
    ):
        raise ValueError("vocabulary SHA-256 differs from the frozen protocol")
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
    source_split: str = "train",
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
    source_split : str, optional
        Official split name used to qualify stable row identities.

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
            for source_index, text, label in zip(indices, client_texts, client_labels):
                file.write(
                    json.dumps(
                        {
                            "label": int(label),
                            "row_id": f"{source_split}:{source_index}",
                            "text": str(text),
                        },
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


def publish_public_artifacts(
    vectorizer: Any,
    output_dir: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the vocabulary and its train-only provenance contract.

    Parameters
    ----------
    vectorizer : Any
        Verified training-split vectorizer.
    output_dir : str or pathlib.Path
        Public artifact directory.
    protocol : mapping or None, optional
        Parsed frozen protocol, primarily for deterministic tests.

    Returns
    -------
    dict of str to Any
        Published public manifest.
    """
    output_path = resolve_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vocabulary = vectorizer.get_vocabulary()
    vocabulary_bytes = b"".join(item.encode("utf-8") + b"\n" for item in vocabulary)
    frozen = protocol or load_scientific_protocol()
    preprocessing = frozen["preprocessing"]
    vocabulary_sha256 = hashlib.sha256(vocabulary_bytes).hexdigest()
    if len(vocabulary) != preprocessing["vocabulary_size"]:
        raise ValueError("vocabulary size differs from the frozen protocol")
    if vocabulary_sha256 != preprocessing["vocabulary_sha256"]:
        raise ValueError("vocabulary SHA-256 differs from the frozen protocol")
    (output_path / "vocab.txt").write_bytes(vocabulary_bytes)
    dataset = frozen["dataset"]
    train = dataset["splits"]["train"]
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "vocabulary_size": len(vocabulary),
        "vocabulary": {
            "filename": "vocab.txt",
            "sha256": vocabulary_sha256,
            "size_bytes": len(vocabulary_bytes),
        },
        "dataset": {
            "id": dataset["id"],
            "config": dataset["config"],
            "revision": dataset["revision"],
            "datasets_version": dataset["datasets_version"],
            "split": "train",
            "rows": train["rows"],
            "raw_parquet_sha256": train["raw_parquet_sha256"],
            "content_sha256": train["content_sha256"],
        },
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def prepare_all(
    partitions: int,
    client_shard_dir: str | Path,
    public_artifact_dir: str | Path,
    evaluation_artifact_dir: str | Path,
) -> None:
    """Create train-only client/public artifacts and the separate test artifact.

    Parameters
    ----------
    partitions : int
        Number of client training shards.
    client_shard_dir : str or pathlib.Path
        Parent directory for train-only client shards.
    public_artifact_dir : str or pathlib.Path
        Directory for the shared train-derived vocabulary.
    evaluation_artifact_dir : str or pathlib.Path
        New immutable directory for the untouched official test split.

    Returns
    -------
    None
    """
    client_dir = resolve_dir(client_shard_dir)
    public_dir = resolve_dir(public_artifact_dir)
    evaluation_dir = resolve_dir(evaluation_artifact_dir)
    roots = {
        "client": client_dir.resolve(strict=False),
        "public": public_dir.resolve(strict=False),
        "evaluation": evaluation_dir.resolve(strict=False),
    }
    for first_name, first in roots.items():
        for second_name, second in roots.items():
            if first_name >= second_name:
                continue
            if (
                first == second
                or first.is_relative_to(second)
                or second.is_relative_to(first)
            ):
                raise ValueError(
                    f"{first_name} and {second_name} artifact roots must be separate"
                )
    if evaluation_dir.exists() or evaluation_dir.is_symlink():
        raise FileExistsError(
            "evaluation artifact path already exists; use a new path to preserve "
            "the immutable test set"
        )

    _validated_frameworks()
    dataset = load_verified_imdb_dataset()
    train = dataset["train"]
    texts = np.asarray(train["text"])
    labels = np.asarray(train["label"], dtype="int32")
    vectorizer = build_vectorizer(texts)
    protocol = load_scientific_protocol()
    manifest = publish_public_artifacts(vectorizer, public_dir, protocol=protocol)
    dataset_spec = protocol["dataset"]
    train_spec = dataset_spec["splits"]["train"]
    package_raw_client_shards(
        texts,
        labels,
        client_dir,
        num_clients=partitions,
        manifest=manifest,
        metadata={
            "dataset": {
                "id": dataset_spec["id"],
                "config": dataset_spec["config"],
                "revision": dataset_spec["revision"],
                "datasets_version": dataset_spec["datasets_version"],
                "split": "train",
                "content_sha256": train_spec["content_sha256"],
            },
            "source_split": "train",
            "row_identity": "train:{zero_based_official_split_row_index}",
        },
    )
    publish_evaluation_artifact(dataset["test"], evaluation_dir, protocol=protocol)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse data-preparation command-line arguments.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    argparse.Namespace
        Validated preparation options.
    """
    parser = argparse.ArgumentParser(description="Prepare sentiment data partitions.")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument(
        "--client-shard-dir", type=Path, default=Path("artifacts/clients")
    )
    parser.add_argument(
        "--public-artifact-dir", type=Path, default=default_public_artifact_dir()
    )
    parser.add_argument(
        "--evaluation-artifact-dir",
        type=Path,
        default=default_evaluation_artifact_dir(),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run verified train/test artifact preparation.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit command-line arguments, or process arguments when omitted.

    Returns
    -------
    None
    """
    args = parse_args(argv)
    prepare_all(
        partitions=args.partitions,
        client_shard_dir=args.client_shard_dir,
        public_artifact_dir=args.public_artifact_dir,
        evaluation_artifact_dir=args.evaluation_artifact_dir,
    )


if __name__ == "__main__":
    main()
