"""Benchmark aggregation scaling and controlled malicious-update scenarios."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns
from typing import Any

import numpy as np

from src.artifact_compatibility import write_json_atomically
from src.strategy_runner import _parameter_payload_bytes, aggregate_model_weights

STRATEGIES = ("fedavg", "fedprox_huber", "fedmedian", "fedtrimmedavg")
ATTACK_SCALES = {"outlier": 10.0, "sign_flip": -10.0}


def _updates(
    client_count: int, dimensions: int, seed: int
) -> tuple[list[list[np.ndarray]], list[int]]:
    """Return deterministic finite float32 client models and equal sample counts.

    Parameters
    ----------
    client_count : int
        Number of simulated clients.
    dimensions : int
        Parameters in the single benchmark tensor.
    seed : int
        NumPy PCG64 seed.

    Returns
    -------
    tuple of list and list
        Client model tensors and equal positive sample counts.
    """
    generator = np.random.Generator(np.random.PCG64(seed))
    values = generator.normal(0.0, 0.1, size=(client_count, dimensions)).astype(
        np.float32
    )
    return [[row] for row in values], [100] * client_count


def _aggregate(
    strategy: str, client_models: list[list[np.ndarray]], sample_counts: list[int]
) -> list[np.ndarray]:
    """Aggregate one benchmark update set with registered strategy settings.

    Parameters
    ----------
    strategy : str
        Registered aggregation strategy.
    client_models : list of list of numpy.ndarray
        Simulated client models.
    sample_counts : list of int
        Positive client sample counts.

    Returns
    -------
    list of numpy.ndarray
        Validated aggregate tensors.
    """
    return aggregate_model_weights(
        strategy,
        client_models,
        sample_counts,
        huber_threshold=10.0,
        trimmed_fraction=0.25,
    )


def run_benchmark(
    *,
    dimensions: int,
    repeats: int,
    seed: int,
    client_counts: tuple[int, ...] = (4, 16, 64),
) -> dict[str, Any]:
    """Run deterministic scaling and controlled-attack microbenchmarks.

    Parameters
    ----------
    dimensions : int
        Parameters in each simulated client update.
    repeats : int
        Timed aggregation repetitions per scale and strategy.
    seed : int
        NumPy PCG64 seed.
    client_counts : tuple of int, optional
        Simulated client scales.

    Returns
    -------
    dict of str to Any
        JSON-compatible benchmark results.

    Raises
    ------
    ValueError
        If dimensions, repeats, seed, or client counts are invalid.
    """
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("dimensions must be a positive integer")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not client_counts or any(
        type(client_count) is not int or client_count < 4
        for client_count in client_counts
    ):
        raise ValueError("client counts must contain integers of at least four")

    scalability = []
    for client_count in client_counts:
        models, counts = _updates(client_count, dimensions, seed + client_count)
        one_model_bytes = _parameter_payload_bytes(models[0])
        for strategy in STRATEGIES:
            durations = []
            for _ in range(repeats):
                started_at = perf_counter_ns()
                _aggregate(strategy, models, counts)
                durations.append(perf_counter_ns() - started_at)
            scalability.append(
                {
                    "client_count": client_count,
                    "strategy": strategy,
                    "mean_aggregation_time_ns": fmean(durations),
                    "durations_ns": durations,
                    "fit_parameter_bytes": one_model_bytes * client_count * 2,
                }
            )

    honest_models, counts = _updates(4, dimensions, seed)
    honest_target = _aggregate("fedavg", honest_models, counts)[0]
    robustness = []
    for attack, scale in ATTACK_SCALES.items():
        for malicious_clients in (1, 2):
            attacked = [[tensor.copy() for tensor in model] for model in honest_models]
            for client_id in range(malicious_clients):
                attacked[client_id][0] *= np.float32(scale)
            for strategy in STRATEGIES:
                aggregate = _aggregate(strategy, attacked, counts)[0]
                robustness.append(
                    {
                        "strategy": strategy,
                        "attack": attack,
                        "malicious_fraction": malicious_clients / 4,
                        "aggregate_error_l2": float(
                            np.linalg.norm(
                                aggregate.astype(np.float64)
                                - honest_target.astype(np.float64)
                            )
                        ),
                    }
                )

    return {
        "schema_version": 1,
        "scope": (
            "aggregation microbenchmark over deterministic synthetic float32 model "
            "updates; not end-to-end training"
        ),
        "seed": seed,
        "dimensions": dimensions,
        "repeats": repeats,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "logical_cpu_count": os.cpu_count(),
            "packages": {package: version(package) for package in ("flwr", "numpy")},
        },
        "scalability": scalability,
        "robustness": robustness,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse aggregation benchmark arguments.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    argparse.Namespace
        Parsed benchmark options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=67)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run and persist the aggregation benchmark.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    None
    """
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = run_benchmark(
        dimensions=args.dimensions,
        repeats=args.repeats,
        seed=args.seed,
    )
    write_json_atomically(args.output, result, overwrite=False)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
