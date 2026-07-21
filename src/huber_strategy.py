from __future__ import annotations

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
    """Aggregate client update vectors by minimizing the multi-dim Huber loss."""
    weights = np.asarray(sample_counts, dtype="float64")
    weights = weights / weights.sum()

    centre = np.average(client_vectors, axis=0, weights=weights)

    for _ in range(WEISZFELD_ITERS):
        coeffs: list[float] = []
        for vec, w in zip(client_vectors, weights):
            dist = np.linalg.norm(centre - vec)
            scale = 1.0 if dist <= threshold else threshold / (dist + WEISZFELD_EPS)
            coeffs.append(w * scale)
        coefficient_array = np.asarray(coeffs)
        coefficient_array = coefficient_array / coefficient_array.sum()
        centre = np.average(client_vectors, axis=0, weights=coefficient_array)

    return centre
