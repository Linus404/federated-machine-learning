from __future__ import annotations

import argparse
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

from src.paths import data_dir_path, default_data_dir, partition_paths, vocab_path

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"keras\..*")

ArrayPair: TypeAlias = tuple[np.ndarray, np.ndarray]
PartitionSplit: TypeAlias = tuple[ArrayPair, ArrayPair]
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_LOCAL_EPOCHS = 2


def load_partition(
    data_dir: str | Path | None = None,
    partition: int = 0,
    validation_split: float = 0.2,
) -> PartitionSplit:
    """Load one client partition and keep the last rows for validation."""
    resolved_data_dir = data_dir_path(data_dir)
    x_path, y_path = partition_paths(resolved_data_dir, partition)
    x = np.load(x_path).astype("int32", copy=False)
    y = np.load(y_path).astype("float32", copy=False)

    split = int(len(x) * (1 - validation_split))
    return (x[:split], y[:split]), (x[split:], y[split:])


def vocab_size(data_dir: str | Path | None = None) -> int:
    """Count the shared Stage 1 vocabulary entries."""
    resolved_data_dir = data_dir_path(data_dir)
    with vocab_path(resolved_data_dir).open() as file:
        return sum(1 for _ in file)


def sequence_length(data_dir: str | Path | None = None, partition: int = 0) -> int:
    """Read the saved token sequence length without loading all samples."""
    resolved_data_dir = data_dir_path(data_dir)
    x_path, _ = partition_paths(resolved_data_dir, partition)
    x = np.load(x_path, mmap_mode="r")
    return int(x.shape[1])


def build_model(
    vocab_size: int, sequence_length: int, embedding_dim: int = DEFAULT_EMBEDDING_DIM
) -> Any:
    """Build the small sentiment model reused by local and federated training."""
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
    """Train one local baseline model for Stage 2."""
    train_data, val_data = load_partition(
        data_dir=args.data_dir,
        partition=args.partition,
        validation_split=args.validation_split,
    )

    model = build_model(
        vocab_size(args.data_dir), train_data[0].shape[1], args.embedding_dim
    )

    history = model.fit(
        *train_data,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0 if args.quiet else 1,
    )

    loss, accuracy = model.evaluate(*val_data, verbose=0)
    final = {name: float(values[-1]) for name, values in history.history.items()}

    print(
        f"train_loss={final['loss']:.4f} "
        f"train_accuracy={final['accuracy']:.4f} "
        f"val_loss={final['val_loss']:.4f} "
        f"val_accuracy={final['val_accuracy']:.4f} "
        f"eval_loss={loss:.4f} "
        f"eval_accuracy={accuracy:.4f}"
    )
    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one local sentiment model.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
