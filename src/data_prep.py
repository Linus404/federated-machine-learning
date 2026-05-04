from __future__ import annotations

import os
import re

# TensorFlow/Keras read these before import; set them before Keras loads.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
from datasets import load_dataset


def main() -> None:
    """Prepare four sentiment partitions and the shared vocabulary."""
    dataset = load_dataset("stanfordnlp/imdb")
    
    # Remove HTML markup so the vocabulary stores review words, not tags.
    texts = np.asarray([re.sub(r"<[^>]+>", " ", x) for x in dataset["train"]["text"]])
    labels = np.asarray(dataset["train"]["label"], dtype="int32")

    vectorizer = keras.layers.TextVectorization(
        max_tokens=20_000, output_sequence_length=500, dtype="int32"
    )
    vectorizer.adapt(texts)

    rng = np.random.default_rng(67)
    idx = rng.permutation(len(texts))

    x = keras.ops.convert_to_numpy(vectorizer(texts[idx]))
    y = labels[idx]

    os.makedirs("data", exist_ok=True)
    with open("data/vocab.txt", "w") as file:
        file.write("\n".join(vectorizer.get_vocabulary()))

    samples_per_partition: int = 6_250
    for partition in range(4):
        start = partition * samples_per_partition
        stop = start + samples_per_partition
        np.save(f"data/partition_{partition}_x.npy", x[start:stop])
        np.save(f"data/partition_{partition}_y.npy", y[start:stop])

if __name__ == "__main__":
    main()
