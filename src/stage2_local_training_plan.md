# Stage 2: Local Model Training Plan

## Overview

The goal of Stage 2 is to train **one sentiment model on one prepared client partition**.
This stage proves the machine-learning path works before adding federated orchestration in Stage 3.
No client-server communication, aggregation, or Flower code is needed yet.

The workflow consists of four steps:

1. Load one saved data partition
2. Build a small Keras text classifier
3. Train and evaluate locally
4. Keep metrics for comparison with later stages

---

## Step 1: Load one partition

### Why
Stage 1 already produced four independent client partitions. Stage 2 should consume those files directly
instead of downloading or preprocessing the dataset again. This keeps the boundary between data preparation
and model training clear.

### Options

| Option | Pros | Cons |
|---|---|---|
| **NumPy `.npy` loading** | Matches the Stage 1 output exactly and has no extra runtime dependency | Requires Stage 1 files to exist |
| **Re-run preprocessing** | Useful for experimentation | Couples Stage 2 to Stage 1 internals |
| **Load all partitions together** | Higher accuracy may be possible | Hides the local-client setup that Stage 3 will federate |

### Recommendation: NumPy `.npy` loading
Load one pair of files, for example:

```python
x = np.load("data/partition_0_x.npy")
y = np.load("data/partition_0_y.npy")
```

Use a small validation split from the same partition to check whether training behaves sensibly.

---

## Step 2: Build a small Keras model

### Why
The model should be simple enough to train quickly but still match the vectorized text format from Stage 1:
integer token IDs with shape `(samples, sequence_length)`.

### Options
| Option | Pros | Cons |
|---|---|---|
| **Embedding + global average pooling** | Small, fast, backend-portable Keras layers | Ignores word order |
| **Embedding + LSTM/GRU** | Uses word order | Slower and more code |
| **Transformer block** | Stronger text model | Overkill for proving local training |

### Recommendation: Embedding + global average pooling
Use only built-in Keras layers (`Input`, `Embedding`, `GlobalAveragePooling1D`, `Dense`) plus `model.fit()`
and `model.evaluate()`. Keras 3 models that stay in Keras APIs and built-in layers are portable across the
TensorFlow, JAX, and PyTorch backends.

---

## Step 3: Train locally

### Why
The Stage 2 success signal is not final accuracy; it is that a local model can learn from one client's data.
The PDF asks us to verify that loss decreases and accuracy improves before adding federation.

### Training defaults

| Setting | Value |
|---|---|
| Partition | `0` |
| Epochs | `2` |
| Batch size | `64` |
| Validation split | Last 20 % of the selected partition |
| Loss | Binary cross-entropy |
| Metric | Accuracy |

Example command:

```bash
uv run python -m src.local_training --partition 0 --epochs 2
```

---

## Step 4: Record metrics

### Why
Stage 3 will compare federated training against local training. Printing final loss and accuracy now gives a
small baseline and confirms the training loop works.

### Recommendation
Keep Stage 2 output simple: print final training, validation, and evaluation metrics. Avoid checkpointing,
experiment tracking, and federated callbacks until later stages need them.

---

## Validation checklist

After implementing Stage 2, verify:

- [ ] The training script loads one partition without re-running preprocessing
- [ ] The model uses Keras APIs and no backend-native TensorFlow/PyTorch operations
- [ ] A smoke training run exits successfully and prints loss/accuracy
- [ ] The walkthrough notebook opens as valid notebook JSON
- [ ] `ruff check` and `ruff format --check` pass
