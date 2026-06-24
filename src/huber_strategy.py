from __future__ import annotations

from typing import Any

import numpy as np
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

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
        coeffs = []
        for vec, w in zip(client_vectors, weights):
            dist = np.linalg.norm(centre - vec)
            scale = 1.0 if dist <= threshold else threshold / (dist + WEISZFELD_EPS)
            coeffs.append(w * scale)
        coeffs = np.asarray(coeffs)
        coeffs = coeffs / coeffs.sum()
        centre = np.average(client_vectors, axis=0, weights=coeffs)

    return centre


class HuberRobustFedAvg(FedAvg):
    """FedAvg variant that aggregates with multi-dimensional Huber loss."""

    def __init__(
        self, *args: Any, huber_threshold: float = DEFAULT_HUBER_THRESHOLD, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.huber_threshold = huber_threshold

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Any]]:
        if not results:
            return None, {}

        reference = parameters_to_ndarrays(results[0][1].parameters)
        vectors = [
            _flatten(parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results
        ]
        counts = [fit_res.num_examples for _, fit_res in results]

        aggregated_vector = huber_aggregate(vectors, counts, self.huber_threshold)
        aggregated_ndarrays = _unflatten(aggregated_vector, reference)
        parameters = ndarrays_to_parameters(aggregated_ndarrays)

        metrics: dict[str, Any] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters, metrics
