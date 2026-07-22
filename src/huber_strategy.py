from __future__ import annotations

from numbers import Integral

import numpy as np
from flwr.common import NDArrays

DEFAULT_HUBER_THRESHOLD = 10.0
WEISZFELD_ITERS = 10
WEISZFELD_EPS = 1e-8


def _flatten(weights: NDArrays) -> np.ndarray:
    """Flatten a list of weight arrays into one long vector."""
    return np.concatenate([w.flatten() for w in weights])


def _unflatten(vector: np.ndarray, reference: NDArrays) -> NDArrays:
    """Reshape a flat vector back into the original list-of-arrays shape."""
    out: NDArrays = []
    idx = 0
    for ref in reference:
        size = ref.size
        out.append(vector[idx : idx + size].reshape(ref.shape))
        idx += size

    return out


def huber_aggregate(
    client_vectors: list[np.ndarray],
    sample_counts: list[int],
    threshold: float,
) -> np.ndarray:
    """Aggregate post-fit client vectors by minimizing multidimensional Huber loss.

    Args:
        client_vectors: Flattened post-fit model vectors in ascending client ID order.
        sample_counts: Positive fitted-row counts in the same client order.
        threshold: Positive Huber residual threshold.

    Returns:
        The float64 aggregate after the fixed number of reweighting iterations.

    Raises:
        ValueError: If inputs violate the frozen aggregation contract.
    """
    if not client_vectors:
        raise ValueError("Huber aggregation requires at least one client vector")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Huber threshold must be finite and positive")

    flattened_vectors = [np.asarray(vector) for vector in client_vectors]
    vector_length = flattened_vectors[0].size
    if vector_length == 0 or any(
        vector.ndim != 1 or vector.size != vector_length for vector in flattened_vectors
    ):
        raise ValueError("Huber client vectors must be nonempty one-dimensional peers")
    vectors = np.stack(flattened_vectors).astype(np.float32, copy=False)
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Huber client vectors must contain only finite values")
    if len(sample_counts) != len(vectors) or any(
        isinstance(count, bool) or not isinstance(count, Integral) or count <= 0
        for count in sample_counts
    ):
        raise ValueError("Huber sample counts must be positive integers per client")

    weights = np.asarray(sample_counts, dtype=np.float64)
    weights = weights / weights.sum()

    centre = np.average(vectors, axis=0, weights=weights)

    for _ in range(WEISZFELD_ITERS):
        coeffs: list[float] = []
        for vec, weight in zip(vectors, weights, strict=True):
            dist = np.linalg.norm(centre - vec)
            scale = 1.0 if dist <= threshold else threshold / (dist + WEISZFELD_EPS)
            coeffs.append(weight * scale)
        coefficient_array = np.asarray(coeffs, dtype=np.float64)
        coefficient_sum = coefficient_array.sum()
        if not np.isfinite(coefficient_sum) or coefficient_sum <= 0:
            raise ValueError("Huber effective weights must have a positive finite sum")
        coefficient_array = coefficient_array / coefficient_sum
        centre = np.average(vectors, axis=0, weights=coefficient_array)
        if not np.all(np.isfinite(centre)):
            raise ValueError("Huber aggregate became non-finite")

    return centre
