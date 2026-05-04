from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeAlias

# TensorFlow/Keras read these before import; set them before Keras loads.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np

ArrayPair: TypeAlias = tuple[np.ndarray, np.ndarray]
PartitionSplit: TypeAlias = tuple[ArrayPair, ArrayPair]


def load_partition(
    partition: int = 0, validation_split: float = 0.2
) -> PartitionSplit:
    """Load one client partition and keep the last rows for validation."""

    x = np.load(f"data/partition_{partition}_x.npy").astype(
        "int32", copy=False
    )
    y = np.load(f"data/partition_{partition}_y.npy").astype(
        "float32", copy=False
    )

    split = int(len(x) * (1 - validation_split))
    return (x[:split], y[:split]), (x[split:], y[split:])


@lru_cache(maxsize=None)
def vocab_size() -> int:
    """Count the shared Stage 1 vocabulary entries."""
    vocab_path = Path("data/vocab.txt")
    return sum(1 for _ in vocab_path.open())


@lru_cache(maxsize=None)
def sequence_length(partition: int = 0) -> int:
    """Read the saved token sequence length without loading all samples."""
    x = np.load(f"data/partition_{partition}_x.npy", mmap_mode="r")
    return int(x.shape[1])


def build_model(vocab_size: int, sequence_length: int, embedding_dim: int = 16) -> Any:
    """Build the small sentiment model reused by local and federated training."""
    inputs = keras.Input(shape=(sequence_length,), dtype="int32")

    # Average word embeddings keep the baseline compact and quick to train.
    x = keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    x = keras.layers.GlobalAveragePooling1D()(x)

    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train(args: argparse.Namespace) -> tuple[Any, Any]:
    """Train one local baseline model for Stage 2."""
    train_data: ArrayPair
    val_data: ArrayPair
    train_data, val_data = load_partition(
        args.partition, args.validation_split
    )

    model = build_model(
        vocab_size(), train_data[0].shape[1], args.embedding_dim
    )

    history = model.fit(
        *train_data,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0 if args.quiet else 1,
    )

    loss: float
    accuracy: float
    loss, accuracy = model.evaluate(*val_data, verbose=0)
    final: dict[str, float] = {
        name: float(values[-1]) for name, values in history.history.items()
    }

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
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
