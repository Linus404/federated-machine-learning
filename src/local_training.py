from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path
from typing import Any, TypeAlias

# TensorFlow/Keras read these before import; set them before Keras loads.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np

from src.app_manifest import AppManifest, load_app_manifest
from src.artifact_compatibility import (
    validate_artifact_schema,
    write_server_artifact_manifest,
)
from src.artifact_history import (
    DEFAULT_ARTIFACT_RETENTION_RUNS,
    create_run_artifact_dir,
    prune_run_history,
    publish_completed_run,
)
from src.contracts import DEFAULT_VALIDATION_SEED
from src.paths import (
    acquire_run_artifact_lock,
    default_public_artifact_dir,
    resolve_dir,
)

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"keras\..*")

ArrayPair: TypeAlias = tuple[np.ndarray, np.ndarray]
PartitionSplit: TypeAlias = tuple[ArrayPair, ArrayPair]
DEFAULT_LOCAL_EPOCHS = 1
CLIENT_REVIEWS_FILENAME = "reviews.jsonl"


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
    validation_split: float = 0.2,
) -> PartitionSplit:
    """Load raw private reviews and tokenize them inside the client process.

    Parameters
    ----------
    client_data_dir : str or pathlib.Path
        Directory containing one versioned client shard.
    manifest : AppManifest
        Public vocabulary and model manifest.
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
    resolved_client_dir = resolve_dir(client_data_dir)
    metadata_path = resolved_client_dir / "client_metadata.json"
    if not metadata_path.exists():
        raise ValueError(
            "client shard metadata has no valid schema_version; regenerate its "
            "artifacts"
        )
    validate_artifact_schema(
        json.loads(metadata_path.read_text(encoding="utf-8")), "client shard metadata"
    )
    with (resolved_client_dir / CLIENT_REVIEWS_FILENAME).open(
        encoding="utf-8"
    ) as review_file:
        records = [json.loads(line) for line in review_file]
    with manifest.vocabulary_path.open(encoding="utf-8") as vocabulary_file:
        saved_vocab = [
            line.removesuffix("\n").removesuffix("\r") for line in vocabulary_file
        ]
    vectorizer = keras.layers.TextVectorization(
        output_mode="int",
        output_sequence_length=int(manifest.payload["sequence_length"]),
        vocabulary=saved_vocab[2:],
        dtype="int32",
    )
    x = np.asarray(vectorizer([record["text"] for record in records]), dtype="int32")
    y = np.asarray([record["label"] for record in records], dtype="float32")

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
    """Build the sentiment model reused by local and federated training."""
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
    """Train one model from a client's raw private shard.

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
        run_dir = create_run_artifact_dir(
            artifact_root,
            run_config,
            public_artifact_dir=args.public_artifact_dir,
        )
        prune_run_history(artifact_root, retention_runs, active_run_dir=run_dir)
        write_server_artifact_manifest(run_dir)
        train_data, val_data = load_client_shard(
            args.client_data_dir,
            manifest,
            args.validation_split,
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
        int(payload["vocabulary_size"]),
        int(payload["sequence_length"]),
        int(payload["embedding_dim"]),
    )


if __name__ == "__main__":
    train(parse_args())
