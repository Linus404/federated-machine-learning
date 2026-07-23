"""Run the registered local-only and federated strategy contracts."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from flwr.server.strategy.aggregate import (
    aggregate_inplace,
    aggregate_median,
    aggregate_trimmed_avg,
)

from src.huber_strategy import _flatten, _unflatten, huber_aggregate

NDArrays: TypeAlias = list[np.ndarray]
FEDERATED_STRATEGIES = (
    "fedavg",
    "fedprox",
    "fedprox_huber",
    "fedmedian",
    "fedtrimmedavg",
)
REGISTERED_STRATEGIES = ("local_only", *FEDERATED_STRATEGIES)


def _validate_client_weights(
    client_weights: list[NDArrays], sample_counts: list[int]
) -> None:
    """Validate complete ordered client models before aggregation.

    Parameters
    ----------
    client_weights : list of list of numpy.ndarray
        Complete post-fit model weights in ascending client ID order.
    sample_counts : list of int
        Registered fitted-row counts in the same client order.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If counts or model tensors violate the registered contract.
    """
    if not client_weights or len(client_weights) != len(sample_counts):
        raise ValueError("aggregation requires one sample count per client model")
    if any(type(count) is not int or count <= 0 for count in sample_counts):
        raise ValueError("sample counts must be positive built-in integers")
    reference_shapes = tuple(array.shape for array in client_weights[0])
    if not reference_shapes or any(not shape for shape in reference_shapes):
        raise ValueError("client models must contain non-scalar weight tensors")
    for weights in client_weights:
        if tuple(array.shape for array in weights) != reference_shapes:
            raise ValueError("client model weight shapes must match")
        if any(
            array.dtype != np.dtype(np.float32)
            or array.size == 0
            or not np.all(np.isfinite(array))
            for array in weights
        ):
            raise ValueError(
                "client model weights must be nonempty finite float32 tensors"
            )


def aggregate_model_weights(
    strategy: str,
    client_weights: list[NDArrays],
    sample_counts: list[int],
    *,
    huber_threshold: float = 10.0,
    trimmed_fraction: float = 0.25,
) -> NDArrays:
    """Aggregate complete client models with one registered strategy.

    Parameters
    ----------
    strategy : str
        One of the registered federated strategy identifiers.
    client_weights : list of list of numpy.ndarray
        Complete post-fit model weights in ascending client ID order.
    sample_counts : list of int
        Registered fitted-row counts in ascending client ID order.
    huber_threshold : float, optional
        Registered multidimensional Huber threshold.
    trimmed_fraction : float, optional
        Registered coordinate-wise trimming proportion.

    Returns
    -------
    list of numpy.ndarray
        Aggregated model weights in model order.

    Raises
    ------
    ValueError
        If the strategy or aggregation inputs are not registered and valid.
    """
    if strategy not in FEDERATED_STRATEGIES:
        raise ValueError(f"unsupported federated strategy: {strategy}")
    _validate_client_weights(client_weights, sample_counts)

    if strategy == "fedprox_huber":
        aggregate = huber_aggregate(
            [_flatten(weights) for weights in client_weights],
            sample_counts,
            huber_threshold,
        )
        return _unflatten(aggregate, client_weights[0])

    weighted_models = list(zip(client_weights, sample_counts, strict=True))
    if strategy == "fedmedian":
        return aggregate_median(weighted_models)
    if strategy == "fedtrimmedavg":
        return aggregate_trimmed_avg(
            weighted_models, proportiontocut=trimmed_fraction
        )

    results: list[tuple[Any, FitRes]] = [
        (
            None,
            FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(weights),
                num_examples=count,
                metrics={"client_id": client_id},
            ),
        )
        for client_id, (weights, count) in enumerate(
            zip(client_weights, sample_counts, strict=True)
        )
    ]
    return aggregate_inplace(results)
