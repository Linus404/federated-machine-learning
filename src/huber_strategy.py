from __future__ import annotations

from numbers import Real

import numpy as np
from flwr.common import NDArrays

DEFAULT_HUBER_THRESHOLD = 10.0
WEISZFELD_ITERS = 10
WEISZFELD_EPS = 1e-8
_PROTOCOL_DTYPE = np.dtype(np.float32)


def _flatten(weights: NDArrays) -> np.ndarray:
    """Flatten protocol model weights into one C-order vector.

    Parameters
    ----------
    weights : NDArrays
        Nonempty finite float32 model weights in model order.

    Returns
    -------
    numpy.ndarray
        The concatenated float32 model vector.

    Raises
    ------
    ValueError
        If the weights violate the frozen aggregation contract.
    """
    arrays = [np.asarray(weight) for weight in weights]
    if not arrays:
        raise ValueError("Huber model weights must be nonempty")
    if any(array.dtype != _PROTOCOL_DTYPE for array in arrays):
        raise ValueError("Huber model weights must use exactly float32")
    if any(array.ndim == 0 or array.size == 0 for array in arrays):
        raise ValueError("Huber model weights must be non-scalar and nonempty")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Huber model weights must contain only finite values")
    return np.concatenate([array.flatten(order="C") for array in arrays])


def _unflatten(vector: np.ndarray, reference: NDArrays) -> NDArrays:
    """Reshape a flat vector to reference model shapes in C order.

    Parameters
    ----------
    vector : numpy.ndarray
        One-dimensional finite float32 aggregate vector.
    reference : NDArrays
        Nonempty float32 model weights whose shapes define reconstruction.

    Returns
    -------
    NDArrays
        Float32 aggregate arrays in reference model-weight order.

    Raises
    ------
    ValueError
        If the vector or reference cannot define an exact reconstruction.
    """
    vector = np.asarray(vector)
    if vector.dtype != _PROTOCOL_DTYPE or vector.ndim != 1 or vector.size == 0:
        raise ValueError("Huber reconstruction requires a nonempty float32 vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Huber reconstruction vector must contain only finite values")
    if not reference:
        raise ValueError("Huber reconstruction requires a vector and model reference")
    reference_arrays = [np.asarray(weight) for weight in reference]
    if any(array.dtype != _PROTOCOL_DTYPE for array in reference_arrays):
        raise ValueError("Huber reconstruction reference must use exactly float32")
    if any(array.ndim == 0 or array.size == 0 for array in reference_arrays):
        raise ValueError(
            "Huber reconstruction reference must be non-scalar and nonempty"
        )
    if any(not np.all(np.isfinite(array)) for array in reference_arrays):
        raise ValueError("Huber reconstruction reference must contain finite values")
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

    Parameters
    ----------
    client_vectors : list[numpy.ndarray]
        Finite, nonempty float32 post-fit model vectors in ascending client ID order.
    sample_counts : list[int]
        Positive built-in Python integer fitted-row counts in the same client order.
    threshold : float
        Positive finite Huber residual threshold.

    Returns
    -------
    numpy.ndarray
        The aggregate explicitly rounded to float32 after the fixed float64
        reweighting iterations.

    Raises
    ------
    ValueError
        If inputs violate the frozen aggregation contract.
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
    if any(vector.dtype != _PROTOCOL_DTYPE for vector in flattened_vectors):
        raise ValueError("Huber client vectors must use exactly float32")
    vector_length = flattened_vectors[0].size
    if vector_length == 0 or any(
        vector.ndim != 1 or vector.size != vector_length for vector in flattened_vectors
    ):
        raise ValueError("Huber client vectors must be nonempty one-dimensional peers")
    vectors = np.stack(flattened_vectors)
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Huber client vectors must contain only finite values")
    if len(sample_counts) != len(vectors) or any(
        type(count) is not int or count <= 0 for count in sample_counts
    ):
        raise ValueError(
            "Huber sample counts must be positive built-in integers per client"
        )

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

    aggregate = centre.astype(np.float32)
    if not np.all(np.isfinite(aggregate)):
        raise ValueError("Huber float32 aggregate became non-finite")
    return aggregate
