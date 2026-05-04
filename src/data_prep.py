import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress CUDA warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import numpy as np
from datasets import load_dataset
import keras


def main():
    dataset = load_dataset("stanfordnlp/imdb")

    # Clean dataset of HTML artifacts
    texts = [re.sub(r"<[^>]+>", " ", x) for x in dataset["train"]["text"]]
    labels = np.array(dataset["train"]["label"])

    # TODO: Try different values
    vectorizer = keras.layers.TextVectorization(
        max_tokens=20_000, output_sequence_length=500, dtype="int32"
    )
    vectorizer.adapt(texts)

    # Set seed
    rng = np.random.default_rng(67)

    # Generate random sequence of indices of the entire test set
    idx = rng.permutation(25_000)

    # Pass the randomly sorted texts to the vectorizer
    x = keras.ops.convert_to_numpy(vectorizer([texts[i] for i in idx]))
    y = labels[idx]  # Contains the corresponding labels

    os.makedirs("data", exist_ok=True)
    with open("data/vocab.txt", "w") as f:
        f.write("\n".join(vectorizer.get_vocabulary()))

    for i in range(4):
        np.save(f"data/partition_{i}_x.npy", x[i * 6250 : (i + 1) * 6250])
        np.save(f"data/partition_{i}_y.npy", y[i * 6250 : (i + 1) * 6250])


if __name__ == "__main__":
    main()
