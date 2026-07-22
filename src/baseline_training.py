"""Train centralized and local-only baselines on prepared IMDB artifacts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from statistics import fmean
from typing import Any

import keras
import numpy as np

from src.app_manifest import AppManifest, load_app_manifest
from src.artifact_compatibility import canonical_json_bytes
from src.evaluation_artifact import load_evaluation_artifact_snapshot
from src.local_training import (
    ClientShardSnapshot,
    _tokenize_client_shard,
    build_model_from_manifest,
    load_client_shard_snapshot,
    tokenize_rows,
)
from src.paths import (
    default_evaluation_artifact_dir,
    default_public_artifact_dir,
)

DEFAULT_BASELINE_CLIENTS = 4
DEFAULT_BASELINE_EPOCHS = 20
DEFAULT_BASELINE_SEED = 67


def _validate_args(args: argparse.Namespace) -> None:
    """Validate baseline command arguments before loading artifacts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed baseline command arguments.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If numeric arguments or the client path template are invalid.
    """
    if args.baseline not in {"centralized", "local-only"}:
        raise ValueError("baseline must be centralized or local-only")
    for name in ("batch_size", "client_count", "epochs", "seed"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be a positive integer")
    if not 0 < args.validation_split < 1:
        raise ValueError("validation-split must be between 0 and 1")
    template = str(args.client_data_dir)
    if "{partition" not in template:
        raise ValueError("client-data-dir must contain {partition}")
    try:
        formatted = template.format(partition=0)
    except (AttributeError, IndexError, KeyError, ValueError) as error:
        raise ValueError(
            "client-data-dir must be a valid template using {partition}"
        ) from error
    if formatted == template:
        raise ValueError("client-data-dir must expand {partition}")


def _load_client_snapshots(
    args: argparse.Namespace, manifest: AppManifest
) -> list[ClientShardSnapshot]:
    """Load every prepared client shard and prove complete train coverage.

    Parameters
    ----------
    args : argparse.Namespace
        Validated baseline command arguments.
    manifest : AppManifest
        Validated public train artifact.

    Returns
    -------
    list of ClientShardSnapshot
        Client snapshots in ascending client ID order.

    Raises
    ------
    ValueError
        If the selected shards do not partition the complete official train split.
    """
    snapshots = [
        load_client_shard_snapshot(
            str(args.client_data_dir).format(partition=client_id), manifest, client_id
        )
        for client_id in range(args.client_count)
    ]
    source_indices = sorted(
        int(row_id.removeprefix("train:"))
        for snapshot in snapshots
        for row_id, _, _ in snapshot.rows
    )
    if source_indices != list(range(manifest.payload["dataset"]["rows"])):
        raise ValueError(
            "client shards must exactly partition the official train split"
        )
    return snapshots


def _combined_split(
    snapshots: list[ClientShardSnapshot],
    manifest: AppManifest,
    validation_split: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Union deterministic client fitted and validation sets for one model.

    Parameters
    ----------
    snapshots : list of ClientShardSnapshot
        Complete client snapshots in client ID order.
    manifest : AppManifest
        Public vocabulary and model contract.
    validation_split : float
        Per-client deterministic validation fraction.

    Returns
    -------
    tuple of tuple
        Combined fitted and validation arrays.
    """
    splits = [
        _tokenize_client_shard(snapshot, manifest, validation_split)
        for snapshot in snapshots
    ]
    return (
        (
            np.concatenate([split[0][0] for split in splits], axis=0),
            np.concatenate([split[0][1] for split in splits], axis=0),
        ),
        (
            np.concatenate([split[1][0] for split in splits], axis=0),
            np.concatenate([split[1][1] for split in splits], axis=0),
        ),
    )


def _validation_curve(history: Any, epochs: int) -> list[dict[str, float | int]]:
    """Return a finite zero-based validation curve from Keras history.

    Parameters
    ----------
    history : Any
        Keras history returned by ``model.fit``.
    epochs : int
        Required completed epoch count.

    Returns
    -------
    list of dict
        Zero-based validation loss and accuracy rows.

    Raises
    ------
    ValueError
        If history is incomplete or non-finite.
    """
    values = getattr(history, "history", {})
    losses = values.get("val_loss", [])
    accuracies = values.get("val_accuracy", [])
    if len(losses) != epochs or len(accuracies) != epochs:
        raise ValueError("baseline training returned an incomplete validation curve")
    curve = [
        {"epoch": epoch, "loss": float(loss), "accuracy": float(accuracy)}
        for epoch, (loss, accuracy) in enumerate(zip(losses, accuracies))
    ]
    if any(
        not math.isfinite(float(row[field]))
        for row in curve
        for field in ("loss", "accuracy")
    ):
        raise ValueError("baseline validation metrics must be finite")
    return curve


def _evaluate_model(
    model: Any, test_data: tuple[np.ndarray, np.ndarray]
) -> dict[str, float]:
    """Evaluate one trained model on the untouched test artifact.

    Parameters
    ----------
    model : Any
        Trained Keras model.
    test_data : tuple of numpy.ndarray
        Tokenized untouched test examples and labels.

    Returns
    -------
    dict of str to float
        Finite test loss and accuracy.

    Raises
    ------
    ValueError
        If Keras returns incomplete or non-finite metrics.
    """
    result = model.evaluate(*test_data, verbose=0)
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise ValueError("baseline evaluation must return loss and accuracy")
    metrics = {"loss": float(result[0]), "accuracy": float(result[1])}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("baseline test metrics must be finite")
    return metrics


def _fit_model(
    model: Any,
    train_data: tuple[np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> Any:
    """Fit one baseline model without test access or framework reshuffling.

    Parameters
    ----------
    model : Any
        Compiled Keras model.
    train_data : tuple of numpy.ndarray
        Fixed fitted rows.
    validation_data : tuple of numpy.ndarray
        Fixed validation rows.
    args : argparse.Namespace
        Validated baseline command arguments.

    Returns
    -------
    Any
        Keras training history.
    """
    return model.fit(
        *train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=False,
        verbose=0 if args.quiet else 1,
    )


def _load_test_data(
    args: argparse.Namespace, manifest: AppManifest
) -> tuple[np.ndarray, np.ndarray]:
    """Load and tokenize the untouched artifact after training is complete.

    Parameters
    ----------
    args : argparse.Namespace
        Validated baseline command arguments.
    manifest : AppManifest
        Public vocabulary and model contract.

    Returns
    -------
    tuple of numpy.ndarray
        Tokenized untouched test examples and labels.
    """
    snapshot = load_evaluation_artifact_snapshot(args.evaluation_artifact_dir)
    return tokenize_rows(snapshot.rows, manifest)


def _run_centralized(
    args: argparse.Namespace,
    manifest: AppManifest,
    snapshots: list[ClientShardSnapshot],
) -> tuple[dict[str, Any], list[Any]]:
    """Train and evaluate the centralized baseline.

    Parameters
    ----------
    args : argparse.Namespace
        Validated baseline command arguments.
    manifest : AppManifest
        Public vocabulary and model contract.
    snapshots : list of ClientShardSnapshot
        Complete client training shards.

    Returns
    -------
    tuple of dict and list
        JSON result payload and trained model list.
    """
    train_data, validation_data = _combined_split(
        snapshots, manifest, args.validation_split
    )
    keras.utils.set_random_seed(args.seed)
    model = build_model_from_manifest(manifest)
    history = _fit_model(model, train_data, validation_data, args)
    result = {
        "baseline": "centralized",
        "client_count": args.client_count,
        "epochs": args.epochs,
        "seed": args.seed,
        "validation": _validation_curve(history, args.epochs),
        "test": _evaluate_model(model, _load_test_data(args, manifest)),
        "models": ["centralized.keras"],
    }
    return result, [model]


def _run_local_only(
    args: argparse.Namespace,
    manifest: AppManifest,
    snapshots: list[ClientShardSnapshot],
) -> tuple[dict[str, Any], list[Any]]:
    """Train and evaluate one independent model per client.

    Parameters
    ----------
    args : argparse.Namespace
        Validated baseline command arguments.
    manifest : AppManifest
        Public vocabulary and model contract.
    snapshots : list of ClientShardSnapshot
        Complete client training shards.

    Returns
    -------
    tuple of dict and list
        JSON result payload and trained models in client ID order.
    """
    models: list[Any] = []
    clients: list[dict[str, Any]] = []
    for client_id, snapshot in enumerate(snapshots):
        train_data, validation_data = _tokenize_client_shard(
            snapshot, manifest, args.validation_split
        )
        keras.utils.set_random_seed(args.seed + client_id)
        model = build_model_from_manifest(manifest)
        history = _fit_model(model, train_data, validation_data, args)
        models.append(model)
        clients.append(
            {
                "client_id": client_id,
                "seed": args.seed + client_id,
                "validation": _validation_curve(history, args.epochs),
                "model": f"client-{client_id}.keras",
            }
        )

    test_data = _load_test_data(args, manifest)
    for client, model in zip(clients, models):
        client["test"] = _evaluate_model(model, test_data)
    result = {
        "baseline": "local_only",
        "client_count": args.client_count,
        "epochs": args.epochs,
        "seed": args.seed,
        "clients": clients,
        "test_mean": {
            metric: fmean(client["test"][metric] for client in clients)
            for metric in ("loss", "accuracy")
        },
        "models": [client["model"] for client in clients],
    }
    return result, models


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one baseline and persist its models and result payload.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed baseline command arguments.

    Returns
    -------
    dict of str to Any
        Complete JSON-compatible baseline result.

    Raises
    ------
    FileExistsError
        If the requested output directory already exists.
    ValueError
        If arguments, artifacts, training output, or metrics are invalid.
    """
    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        manifest = load_app_manifest(public_artifact_dir=args.public_artifact_dir)
        snapshots = _load_client_snapshots(args, manifest)
        result, models = (
            _run_centralized(args, manifest, snapshots)
            if args.baseline == "centralized"
            else _run_local_only(args, manifest, snapshots)
        )
        for filename, model in zip(result["models"], models):
            model.save(str(output_dir / filename))
        (output_dir / "results.json").write_bytes(canonical_json_bytes(result))
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse baseline training arguments.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    argparse.Namespace
        Parsed baseline options.
    """
    parser = argparse.ArgumentParser(
        description="Train a centralized or local-only sentiment baseline."
    )
    parser.add_argument("baseline", choices=("centralized", "local-only"))
    parser.add_argument(
        "--client-data-dir",
        default="artifacts/clients/client-{partition}",
        help="Client directory template containing {partition}.",
    )
    parser.add_argument("--client-count", type=int, default=DEFAULT_BASELINE_CLIENTS)
    parser.add_argument(
        "--public-artifact-dir", type=Path, default=default_public_artifact_dir()
    )
    parser.add_argument(
        "--evaluation-artifact-dir",
        type=Path,
        default=default_evaluation_artifact_dir(),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_BASELINE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASELINE_SEED)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the selected baseline command.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    None
    """
    run(parse_args(argv))


if __name__ == "__main__":
    main()
