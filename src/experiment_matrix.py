"""Run and summarize the registered portfolio experiment matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, variance
from typing import Any, Mapping, Sequence

from src.artifact_compatibility import canonical_json_bytes
from src.evaluation_artifact import load_scientific_protocol
from src.strategy_runner import (
    REGISTERED_STRATEGIES,
    parse_args as parse_strategy_args,
    run as run_strategy,
)

DEFAULT_STRATEGIES = ("local_only", "fedavg", "fedprox", "fedprox_huber")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
SYSTEM_METRICS = ("training_time_ns", "communication_bytes", "convergence_round")


def _ordered_selection(
    values: Sequence[Any] | None, registered: Sequence[Any], name: str
) -> tuple[Any, ...]:
    """Validate and return a selection in frozen protocol order.

    Parameters
    ----------
    values : sequence of Any or None
        Caller selection, or ``None`` to select every registered value.
    registered : sequence of Any
        Frozen allowed values in their canonical order.
    name : str
        Setting name used in validation errors.

    Returns
    -------
    tuple of Any
        Unique selected values in canonical order.

    Raises
    ------
    ValueError
        If the selection is empty or contains an unregistered value.
    """
    selected = set(registered if values is None else values)
    unknown = selected.difference(registered)
    if not selected or unknown:
        raise ValueError(
            f"{name} must select registered values; invalid: {sorted(unknown)!r}"
        )
    return tuple(value for value in registered if value in selected)


def _cell_metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the final-test metrics from one runner result.

    Parameters
    ----------
    result : mapping of str to Any
        Persisted strategy-runner result.

    Returns
    -------
    mapping of str to Any
        Final-test metrics, using the local-client mean where applicable.
    """
    metrics = result.get(
        "test_mean" if result.get("strategy") == "local_only" else "test"
    )
    if not isinstance(metrics, Mapping):
        raise ValueError("cell result is missing final-test metrics")
    for name in METRICS:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"cell result has invalid {name}")
        if not math.isfinite(float(value)):
            raise ValueError(f"cell result has non-finite {name}")
    return metrics


def _validate_result(
    result: Any, strategy: str, partition: str, seed: int
) -> dict[str, Any]:
    """Validate that a persisted result belongs to the requested cell.

    Parameters
    ----------
    result : Any
        Decoded JSON value.
    strategy : str
        Requested registered strategy.
    partition : str
        Requested registered partition.
    seed : int
        Requested registered seed.

    Returns
    -------
    dict of str to Any
        Validated raw result.

    Raises
    ------
    ValueError
        If the result is malformed or belongs to another cell.
    """
    if not isinstance(result, dict) or result.get("strategy") != strategy:
        raise ValueError("cell result has the wrong strategy")
    config = result.get("config")
    if not isinstance(config, dict) or any(
        config.get(name) != expected
        for name, expected in (("partition", partition), ("seed", seed))
    ):
        raise ValueError("cell result has the wrong partition or seed")
    _cell_metrics(result)
    return result


def _cell_system_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    """Return normalized system metrics available for one cell.

    Parameters
    ----------
    result : mapping of str to Any
        Persisted strategy-runner result.

    Returns
    -------
    dict of str to float
        Training time and, for federated cells, communication and convergence.
    """
    system = result.get("system")
    if not isinstance(system, Mapping):
        return {}
    training_time = system.get(
        "client_training_time_ns", system.get("training_time_ns")
    )
    metrics = (
        {"training_time_ns": float(training_time)}
        if isinstance(training_time, int) and not isinstance(training_time, bool)
        else {}
    )
    communication = system.get("communication")
    if isinstance(communication, Mapping):
        total_bytes = communication.get("total_bytes")
        if isinstance(total_bytes, int) and not isinstance(total_bytes, bool):
            metrics["communication_bytes"] = float(total_bytes)
    convergence = system.get("convergence_round")
    if isinstance(convergence, int) and not isinstance(convergence, bool):
        metrics["convergence_round"] = float(convergence)
    return metrics


def _aggregate(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate final-test metrics across seeds by strategy and partition.

    Parameters
    ----------
    cells : sequence of mapping of str to Any
        Completed raw matrix cells.

    Returns
    -------
    list of dict of str to Any
        Canonically ordered means and sample variances.
    """
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["strategy"], cell["partition"]), []).append(cell)

    summaries = []
    for (strategy, partition), group in groups.items():
        values = {
            name: [float(_cell_metrics(cell["result"])[name]) for cell in group]
            for name in METRICS
        }
        cell_system_metrics = [_cell_system_metrics(cell["result"]) for cell in group]
        system_values = {
            name: [metrics[name] for metrics in cell_system_metrics if name in metrics]
            for name in SYSTEM_METRICS
        }
        summaries.append(
            {
                "strategy": strategy,
                "partition": partition,
                "seeds": [cell["seed"] for cell in group],
                "metrics": {
                    name: {
                        "mean": fmean(samples),
                        "sample_variance": variance(samples)
                        if len(samples) > 1
                        else None,
                    }
                    for name, samples in values.items()
                },
                "system": {
                    name: {
                        "mean": fmean(samples),
                        "sample_variance": variance(samples)
                        if len(samples) > 1
                        else None,
                    }
                    for name, samples in system_values.items()
                    if samples
                },
            }
        )
    return summaries


def _markdown(payload: Mapping[str, Any]) -> str:
    """Render a compact human-readable matrix summary.

    Parameters
    ----------
    payload : mapping of str to Any
        Canonical matrix payload.

    Returns
    -------
    str
        Markdown summary ending in one newline.
    """
    lines = ["# Experiment matrix", "", f"Status: **{payload['status']}**", ""]
    if payload["status"] == "planned":
        lines.extend(("| Strategy | Partition | Seed |", "|---|---|---:|"))
        lines.extend(
            f"| {cell['strategy']} | {cell['partition']} | {cell['seed']} |"
            for cell in payload["cells"]
        )
    else:
        lines.extend(
            (
                "Metric cells show mean (sample variance).",
                "",
                "| Strategy | Partition | Seeds | Accuracy | Precision | Recall | F1 | ROC AUC |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            )
        )
        for summary in payload["aggregates"]:
            values = [
                (
                    summary["metrics"][name]["mean"],
                    summary["metrics"][name]["sample_variance"],
                )
                for name in METRICS
            ]
            lines.append(
                f"| {summary['strategy']} | {summary['partition']} | {len(summary['seeds'])} | "
                + " | ".join(
                    f"{mean:.6f} ({sample_variance:.6f})"
                    if sample_variance is not None
                    else f"{mean:.6f} (n/a)"
                    for mean, sample_variance in values
                )
                + " |"
            )
        lines.extend(("", "System metrics retain nanoseconds and bytes in JSON."))
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Plan or execute a deterministic registered experiment matrix.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed matrix arguments.

    Returns
    -------
    dict of str to Any
        Canonical plan or completed matrix with raw cells and aggregates.
    """
    protocol = load_scientific_protocol()
    strategies = _ordered_selection(
        DEFAULT_STRATEGIES if args.strategies is None else args.strategies,
        REGISTERED_STRATEGIES,
        "strategies",
    )
    partitions = _ordered_selection(
        args.partitions, tuple(protocol["partitioning"]["registered"]), "partitions"
    )
    seeds = _ordered_selection(args.seeds, tuple(protocol["seeding"]["seeds"]), "seeds")
    planned = [
        {"strategy": strategy, "partition": partition, "seed": seed}
        for strategy in strategies
        for partition in partitions
        for seed in seeds
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plan_only:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "planned",
            "cells": planned,
            "aggregates": [],
        }
    else:
        cells = []
        for cell in planned:
            cell_dir = (
                output_dir
                / cell["strategy"]
                / cell["partition"]
                / f"seed-{cell['seed']}"
            )
            result_path = cell_dir / "results.json"
            if result_path.is_file():
                result = _validate_result(
                    json.loads(result_path.read_text(encoding="utf-8")), **cell
                )
            else:
                if cell_dir.exists():
                    raise ValueError(
                        f"incomplete cell directory blocks resume: {cell_dir}"
                    )
                runner_args = parse_strategy_args(
                    [
                        cell["strategy"],
                        "--client-data-dir",
                        args.client_data_dir,
                        "--public-artifact-dir",
                        str(args.public_artifact_dir),
                        "--evaluation-artifact-dir",
                        str(args.evaluation_artifact_dir),
                        "--output-dir",
                        str(cell_dir),
                        "--partition",
                        cell["partition"],
                        "--seed",
                        str(cell["seed"]),
                        *(["--quiet"] if args.quiet else []),
                    ]
                )
                result = _validate_result(run_strategy(runner_args), **cell)
            cells.append({**cell, "result": result})
        payload = {
            "schema_version": 1,
            "status": "complete",
            "cells": cells,
            "aggregates": _aggregate(cells),
        }

    (output_dir / "matrix-results.json").write_bytes(canonical_json_bytes(payload))
    (output_dir / "SUMMARY.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse experiment-matrix arguments.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    argparse.Namespace
        Parsed matrix options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", choices=REGISTERED_STRATEGIES)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("iid_stratified", "dirichlet_1.0", "dirichlet_0.5", "dirichlet_0.1"),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--client-data-dir", default="artifacts/clients/client-{partition}"
    )
    parser.add_argument(
        "--public-artifact-dir", type=Path, default=Path("artifacts/public")
    )
    parser.add_argument(
        "--evaluation-artifact-dir", type=Path, default=Path("artifacts/evaluation")
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the registered experiment matrix.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    None
    """
    payload = run(parse_args(argv))
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
