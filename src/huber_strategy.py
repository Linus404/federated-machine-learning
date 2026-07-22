from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from flwr.common import NDArrays

DEFAULT_HUBER_THRESHOLD = 10.0
WEISZFELD_ITERS = 10
WEISZFELD_EPS = 1e-8
_SUPPORTED_DTYPES = frozenset((np.dtype(np.float32), np.dtype(np.float64)))


def _flatten(weights: NDArrays) -> np.ndarray:
    """Flatten supported model weights into one C-order vector.

    Args:
        weights: Nonempty float32 or float64 model weights in model order.

    Returns:
        The concatenated model vector.

    Raises:
        ValueError: If no weights are provided or a weight has an unsupported dtype.
    """
    arrays = [np.asarray(weight) for weight in weights]
    if not arrays:
        raise ValueError("Huber model weights must be nonempty")
    if any(array.dtype not in _SUPPORTED_DTYPES for array in arrays):
        raise ValueError("Huber model weights must use float32 or float64")
    return np.concatenate([array.flatten() for array in arrays])


def _unflatten(vector: np.ndarray, reference: NDArrays) -> NDArrays:
    """Reshape a flat vector to reference model shapes in C order.

    Args:
        vector: One-dimensional aggregate vector.
        reference: Nonempty model weights whose shapes define reconstruction.

    Returns:
        Aggregate arrays in reference model-weight order.

    Raises:
        ValueError: If the vector or reference cannot define an exact reconstruction.
    """
    vector = np.asarray(vector)
    if vector.ndim != 1 or not reference:
        raise ValueError("Huber reconstruction requires a vector and model reference")
    expected_size = sum(weight.size for weight in reference)
    if vector.size != expected_size:
        raise ValueError("Huber vector length must match the reference model")

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
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("Huber threshold must be finite and positive")

    flattened_vectors = [np.asarray(vector) for vector in client_vectors]
    if any(vector.dtype not in _SUPPORTED_DTYPES for vector in flattened_vectors):
        raise ValueError("Huber client vectors must use float32 or float64")
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
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Huber sample counts must have a positive finite sum")
    weights = weights / weight_sum

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
