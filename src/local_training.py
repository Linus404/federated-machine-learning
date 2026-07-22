from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from src.protocol_runtime import validate_protocol_runtime

import keras
import numpy as np

from src.app_manifest import AppManifest, load_app_manifest
from src.artifact_compatibility import (
    CLIENT_SHARD_SCHEMA_VERSION,
    canonical_json_bytes,
    deep_freeze,
    read_regular_file,
    sha256_bytes,
    validate_artifact_schema,
    write_server_artifact_manifest,
)
from src.artifact_history import (
    DEFAULT_ARTIFACT_RETENTION_RUNS,
    create_run_artifact_dir,
    prune_run_history,
    publish_completed_run,
)
from src.contracts import (
    DEFAULT_VALIDATION_SEED,
    canonical_client_row_bytes,
)
from src.paths import (
    acquire_run_artifact_lock,
    default_public_artifact_dir,
    resolve_dir,
    resolve_prepared_artifact_dir,
)
from src.text_preprocessing import create_text_vectorizer

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"keras\..*")

ArrayPair: TypeAlias = tuple[np.ndarray, np.ndarray]
PartitionSplit: TypeAlias = tuple[ArrayPair, ArrayPair]
DEFAULT_LOCAL_EPOCHS = 1
CLIENT_REVIEWS_FILENAME = "reviews.jsonl"
CLIENT_METADATA_FILENAME = "client_metadata.json"
CLIENT_ROW_IDENTITY = "train:{zero_based_official_split_row_index}"
_CLIENT_METADATA_FIELDS = {
    "schema_version",
    "artifact_type",
    "client_id",
    "dataset",
    "source_split",
    "row_identity",
    "split_seed",
    "alpha",
    "sample_count",
    "label_histogram",
    "records",
    "public_manifest",
}


@dataclass(frozen=True)
class ClientShardSnapshot:
    """Validated immutable snapshot of one private client training shard.

    Parameters
    ----------
    directory : pathlib.Path
        Canonical source directory used only for diagnostics.
    metadata : mapping of str to Any
        Recursively frozen validated metadata.
    metadata_bytes : bytes
        Exact canonical metadata bytes.
    records_bytes : bytes
        Exact canonical record bytes.
    rows : tuple of tuple
        Immutable ``(row_id, text, label)`` records.
    """

    directory: Path
    metadata: Mapping[str, Any]
    metadata_bytes: bytes
    records_bytes: bytes
    rows: tuple[tuple[str, str, int], ...]

    def provenance(self) -> Mapping[str, Any]:
        """Return shard identity and checksums suitable for local provenance.

        Returns
        -------
        mapping of str to Any
            Immutable validated shard evidence without review text.
        """
        identity_fields = (
            "client_id",
            "dataset",
            "source_split",
            "row_identity",
            "sample_count",
            "label_histogram",
            "public_manifest",
        )
        return deep_freeze(
            {
                "identity": {field: self.metadata[field] for field in identity_fields},
                "checksums": {
                    CLIENT_METADATA_FILENAME: sha256_bytes(self.metadata_bytes),
                    CLIENT_REVIEWS_FILENAME: sha256_bytes(self.records_bytes),
                },
            }
        )


def _require_exact_mapping_fields(
    value: object, expected: set[str], name: str
) -> Mapping[str, Any]:
    """Require one client-shard object to have an exact field set.

    Parameters
    ----------
    value : object
        Candidate decoded JSON value.
    expected : set of str
        Complete permitted keys.
    name : str
        Object name used in errors.

    Returns
    -------
    mapping of str to Any
        Validated mapping.

    Raises
    ------
    ValueError
        If the value is not a mapping or its fields differ.
    """
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"client shard {name} has an invalid field set")
    return value


def load_client_shard_snapshot(
    client_data_dir: str | Path,
    manifest: AppManifest,
    expected_client_id: int,
) -> ClientShardSnapshot:
    """Securely snapshot and validate one exact private training shard.

    Parameters
    ----------
    client_data_dir : str or pathlib.Path
        Directory containing metadata and canonical training records.
    manifest : AppManifest
        Validated public artifact snapshot bound by shard metadata.
    expected_client_id : int
        Client identity assigned by the caller.

    Returns
    -------
    ClientShardSnapshot
        Fully validated immutable shard bytes, metadata, and rows.

    Raises
    ------
    ValueError
        If paths, schema, identity, fields, rows, counts, or checksums differ.
    """
    if type(expected_client_id) is not int or expected_client_id < 0:
        raise ValueError("expected client ID must be a non-negative integer")
    logical_directory = resolve_dir(client_data_dir)
    client_root = resolve_prepared_artifact_dir(logical_directory.parent, "client")
    directory = client_root / logical_directory.name
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("client shard directory must be a regular directory")
    canonical_dir = directory.resolve(strict=True)
    if {path.name for path in directory.iterdir()} != {
        CLIENT_METADATA_FILENAME,
        CLIENT_REVIEWS_FILENAME,
    }:
        raise ValueError("client shard contains unexpected files")
    try:
        metadata_bytes = read_regular_file(
            directory / CLIENT_METADATA_FILENAME, parent=canonical_dir
        )
        records_bytes = read_regular_file(
            directory / CLIENT_REVIEWS_FILENAME, parent=canonical_dir
        )
        decoded = json.loads(metadata_bytes.decode("utf-8"))
        metadata = _require_exact_mapping_fields(
            validate_artifact_schema(
                decoded,
                "client shard metadata",
                supported_version=CLIENT_SHARD_SCHEMA_VERSION,
            ),
            _CLIENT_METADATA_FIELDS,
            "metadata",
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError("invalid client shard") from error
    if metadata_bytes != canonical_json_bytes(metadata):
        raise ValueError("client shard metadata bytes are not canonical")
    if metadata["artifact_type"] != "private_client_train_shard":
        raise ValueError("client shard artifact type is invalid")
    if (
        type(metadata["client_id"]) is not int
        or metadata["client_id"] != expected_client_id
    ):
        raise ValueError("client shard ID does not match the expected client ID")
    if metadata["dataset"] != manifest.payload["dataset"]:
        raise ValueError("client shard dataset differs from the public train dataset")
    if (
        metadata["source_split"] != "train"
        or metadata["row_identity"] != CLIENT_ROW_IDENTITY
    ):
        raise ValueError("client shard row identity contract is invalid")
    if (
        type(metadata["split_seed"]) is not int
        or metadata["split_seed"] < 0
        or type(metadata["alpha"]) is not float
        or not math.isfinite(metadata["alpha"])
        or metadata["alpha"] <= 0
    ):
        raise ValueError("client shard partition metadata is invalid")
    if type(metadata["sample_count"]) is not int or metadata["sample_count"] < 1:
        raise ValueError("client shard sample count is invalid")
    histogram = metadata["label_histogram"]
    if not isinstance(histogram, Mapping):
        raise ValueError("client shard label histogram is invalid")
    if not histogram or any(
        label not in {"0", "1"} or type(count) is not int or count < 1
        for label, count in histogram.items()
    ):
        raise ValueError("client shard label histogram is invalid")
    records_contract = _require_exact_mapping_fields(
        metadata["records"],
        {"filename", "format", "encoding", "newline", "trailing_newline", "checksum"},
        "records contract",
    )
    if dict(records_contract) != {
        "filename": CLIENT_REVIEWS_FILENAME,
        "format": "canonical-jsonl",
        "encoding": "utf-8",
        "newline": "LF",
        "trailing_newline": True,
        "checksum": sha256_bytes(records_bytes),
    }:
        raise ValueError("client shard record contract or checksum is invalid")
    public_contract = _require_exact_mapping_fields(
        metadata["public_manifest"],
        {"filename", "size_bytes", "checksum"},
        "public manifest contract",
    )
    if dict(public_contract) != {
        "filename": "manifest.json",
        "size_bytes": len(manifest.manifest_bytes),
        "checksum": sha256_bytes(manifest.manifest_bytes),
    }:
        raise ValueError("client shard public manifest binding is invalid")

    rows: list[tuple[str, str, int]] = []
    counts: Counter[int] = Counter()
    identities: set[str] = set()
    for index, line in enumerate(records_bytes.splitlines(keepends=True)):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid client shard record at row {index}") from error
        if not isinstance(row, Mapping) or set(row) != {"label", "row_id", "text"}:
            raise ValueError(f"invalid client shard record fields at row {index}")
        row_id, text, label = row["row_id"], row["text"], row["label"]
        if not isinstance(text, str) or type(label) is not int or label not in (0, 1):
            raise ValueError(
                f"invalid client shard text or binary label at row {index}"
            )
        if not isinstance(row_id, str) or not row_id.startswith("train:"):
            raise ValueError(f"invalid client shard row identity at row {index}")
        source_index_text = row_id.removeprefix("train:")
        if not source_index_text.isascii() or not source_index_text.isdigit():
            raise ValueError(f"invalid client shard row identity at row {index}")
        source_index = int(source_index_text)
        if (
            str(source_index) != source_index_text
            or source_index >= metadata["dataset"]["rows"]
        ):
            raise ValueError(f"invalid client shard row identity at row {index}")
        if row_id in identities:
            raise ValueError("client shard row identities must be unique")
        if line != canonical_client_row_bytes(row_id, text, label):
            raise ValueError(f"client shard record is not canonical at row {index}")
        identities.add(row_id)
        counts[label] += 1
        rows.append((row_id, text, label))
    if len(rows) != metadata["sample_count"]:
        raise ValueError("client shard sample count differs from its records")
    actual_histogram = {str(label): count for label, count in sorted(counts.items())}
    if actual_histogram != dict(histogram):
        raise ValueError("client shard label histogram differs from its records")
    return ClientShardSnapshot(
        canonical_dir,
        deep_freeze(metadata),
        metadata_bytes,
        records_bytes,
        tuple(rows),
    )


def _stratified_split_indices(
    labels: np.ndarray,
    validation_split: float,
    seed: int = DEFAULT_VALIDATION_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic train/validation indices without label-order bias."""
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        rng.shuffle(label_indices)
        validation_count = int(round(len(label_indices) * validation_split))
        if len(label_indices) > 1:
            validation_count = min(max(validation_count, 1), len(label_indices) - 1)
        else:
            validation_count = 0
        validation_indices.extend(label_indices[:validation_count].tolist())
        train_indices.extend(label_indices[validation_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    if not validation_indices and len(labels) > 1:
        all_indices = np.arange(len(labels))
        rng.shuffle(all_indices)
        validation_count = min(
            max(int(round(len(labels) * validation_split)), 1), len(labels) - 1
        )
        validation_indices = all_indices[:validation_count].tolist()
        train_indices = all_indices[validation_count:].tolist()
    if not train_indices or not validation_indices:
        raise ValueError("client shard is too small for a train/validation split")

    return np.asarray(train_indices), np.asarray(validation_indices)


def load_client_shard(
    client_data_dir: str | Path,
    manifest: AppManifest,
    expected_client_id: int,
    validation_split: float = 0.2,
) -> PartitionSplit:
    """Load client-scoped raw reviews and tokenize them in the client process.

    Parameters
    ----------
    client_data_dir : str or pathlib.Path
        Directory containing one versioned client shard.
    manifest : AppManifest
        Public vocabulary and model manifest.
    expected_client_id : int
        Client identity assigned by the caller.
    validation_split : float, optional
        Fraction of each label assigned to validation.

    Returns
    -------
    PartitionSplit
        Tokenized training and validation arrays.

    Raises
    ------
    ValueError
        If the shard metadata schema is unsupported.
    """
    snapshot = load_client_shard_snapshot(client_data_dir, manifest, expected_client_id)
    return _tokenize_client_shard(snapshot, manifest, validation_split)


def _tokenize_client_shard(
    snapshot: ClientShardSnapshot,
    manifest: AppManifest,
    validation_split: float,
) -> PartitionSplit:
    """Tokenize and deterministically split a validated client snapshot.

    Parameters
    ----------
    snapshot : ClientShardSnapshot
        Fully validated private shard snapshot.
    manifest : AppManifest
        Public vocabulary and model contract.
    validation_split : float
        Fraction of each label assigned to validation.

    Returns
    -------
    PartitionSplit
        Tokenized training and validation arrays.
    """
    vectorizer = create_text_vectorizer(
        sequence_length=manifest.payload["sequence_length"],
        vocabulary=manifest.vocabulary_terms[2:],
    )
    x = np.asarray(vectorizer([row[1] for row in snapshot.rows]), dtype="int32")
    y = np.asarray([row[2] for row in snapshot.rows], dtype="float32")

    train_indices, validation_indices = _stratified_split_indices(y, validation_split)

    return (
        (x[train_indices], y[train_indices]),
        (x[validation_indices], y[validation_indices]),
    )


def build_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int,
) -> Any:
    """Build the sentiment model reused by local and federated training.

    Parameters
    ----------
    vocab_size : int
        Exact frozen vocabulary size.
    sequence_length : int
        Exact frozen token sequence length.
    embedding_dim : int
        Exact frozen embedding dimension.

    Returns
    -------
    Any
        Compiled Keras sentiment model.

    Raises
    ------
    ValueError
        If the runtime differs from the frozen protocol.
    """
    validate_protocol_runtime()
    inputs = keras.Input(shape=(sequence_length,), dtype="int32")

    x = keras.layers.Embedding(vocab_size, embedding_dim, name="token_embedding")(
        inputs
    )
    padding_mask = keras.ops.cast(keras.ops.not_equal(inputs, 0), x.dtype)
    x = x * keras.ops.expand_dims(padding_mask, axis=-1)
    x = keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        use_bias=False,
        name="padding_safe_conv",
    )(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train(args: argparse.Namespace) -> tuple[Any, Any]:
    """Train one model from a client-scoped raw shard.

    Parameters
    ----------
    args : argparse.Namespace
        Validated local-training command arguments.

    Returns
    -------
    tuple of Any
        Trained model and its Keras training history.
    """
    artifact_root = resolve_dir(args.run_artifact_dir)
    retention_runs = int(
        getattr(args, "artifact_retention_runs", DEFAULT_ARTIFACT_RETENTION_RUNS)
    )
    if retention_runs < 1:
        raise ValueError("artifact retention must be a positive integer")
    run_config = {
        "artifact-retention-runs": retention_runs,
        "batch-size": args.batch_size,
        "client-id": getattr(args, "client_id", 0),
        "client-data-dir": args.client_data_dir,
        "epochs": args.epochs,
        "public-artifact-dir": args.public_artifact_dir,
        "quiet": args.quiet,
        "run-artifact-dir": args.run_artifact_dir,
        "validation-seed": DEFAULT_VALIDATION_SEED,
        "validation-split": args.validation_split,
    }
    lock = acquire_run_artifact_lock(artifact_root)
    try:
        manifest = load_app_manifest(public_artifact_dir=args.public_artifact_dir)
        shard_snapshot = load_client_shard_snapshot(
            args.client_data_dir,
            manifest,
            getattr(args, "client_id", 0),
        )
        run_dir = create_run_artifact_dir(
            artifact_root,
            run_config,
            public_artifact_dir=args.public_artifact_dir,
            client_shard=shard_snapshot.provenance(),
        )
        prune_run_history(artifact_root, retention_runs, active_run_dir=run_dir)
        write_server_artifact_manifest(run_dir, app_manifest=manifest)
        train_data, val_data = _tokenize_client_shard(
            shard_snapshot, manifest, args.validation_split
        )
        model = build_model_from_manifest(manifest)

        history = model.fit(
            *train_data,
            validation_data=val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0 if args.quiet else 1,
        )

        loss, accuracy = model.evaluate(*val_data, verbose=0)
        final = {name: float(values[-1]) for name, values in history.history.items()}
        model.save(str(run_dir / "global_model.keras"))
        with (run_dir / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["round", "loss", "accuracy"])
            writer.writeheader()
            for epoch, (epoch_loss, epoch_accuracy) in enumerate(
                zip(history.history["val_loss"], history.history["val_accuracy"]),
                start=1,
            ):
                writer.writerow(
                    {"round": epoch, "loss": epoch_loss, "accuracy": epoch_accuracy}
                )
        publish_completed_run(artifact_root, run_dir)
        prune_run_history(artifact_root, retention_runs, active_run_dir=run_dir)
    finally:
        lock.release()

    print(
        f"train_loss={final['loss']:.4f} "
        f"train_accuracy={final['accuracy']:.4f} "
        f"val_loss={final['val_loss']:.4f} "
        f"val_accuracy={final['val_accuracy']:.4f} "
        f"eval_loss={loss:.4f} "
        f"eval_accuracy={accuracy:.4f}"
    )

    return model, history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one local sentiment model.")
    parser.add_argument("--client-data-dir", type=Path, required=True)
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument(
        "--public-artifact-dir", type=Path, default=default_public_artifact_dir()
    )
    parser.add_argument("--run-artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-retention-runs",
        type=int,
        default=DEFAULT_ARTIFACT_RETENTION_RUNS,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def build_model_from_manifest(manifest: AppManifest) -> Any:
    """Build the sentiment model from public manifest metadata.

    Parameters
    ----------
    manifest : AppManifest
        Manifest containing the model dimensions.

    Returns
    -------
    Any
        Compiled Keras model.
    """
    payload = manifest.payload

    return build_model(
        payload["vocabulary_size"],
        payload["sequence_length"],
        payload["embedding_dim"],
    )


if __name__ == "__main__":
    train(parse_args())
