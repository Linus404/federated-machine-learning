from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

# TensorFlow/Keras read these before import; set them before Keras loads.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
from datasets import load_dataset

from src.paths import data_dir_path, default_data_dir, partition_paths, vocab_path


def prepare_partitions(data_dir: str | Path, partitions: int = 4) -> None:
    """Prepare sentiment partitions and the shared vocabulary."""
    output_dir = data_dir_path(data_dir)
    dataset = load_dataset("stanfordnlp/imdb")

    # Strip markup before vectorization so tokens represent review text, not HTML tags.
    texts = np.asarray([re.sub(r"<[^>]+>", " ", x) for x in dataset["train"]["text"]])
    labels = np.asarray(dataset["train"]["label"], dtype="int32")

    vectorizer = keras.layers.TextVectorization(
        max_tokens=20_000,
        output_sequence_length=500,
        dtype="int32",
    )
    vectorizer.adapt(texts)

    rng = np.random.default_rng(67)
    idx = rng.permutation(len(texts))

    x = keras.ops.convert_to_numpy(vectorizer(texts[idx]))
    y = labels[idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path(output_dir).write_text(
        "\n".join(vectorizer.get_vocabulary()),
        encoding="utf-8",
    )

    samples_per_partition = len(x) // partitions
    for partition in range(partitions):
        start = partition * samples_per_partition
        stop = start + samples_per_partition
        x_path, y_path = partition_paths(output_dir, partition)
        np.save(x_path, x[start:stop])
        np.save(y_path, y[start:stop])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare sentiment data partitions.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--partitions", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_partitions(data_dir=args.data_dir, partitions=args.partitions)


if __name__ == "__main__":
    main()
