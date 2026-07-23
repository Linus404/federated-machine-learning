"""Train centralized and local-only baselines on prepared IMDB artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import string
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, TypeAlias

import keras
import numpy as np

from src.app_manifest import AppManifest, load_app_manifest
from src.artifact_compatibility import canonical_json_bytes
from src.classification_metrics import evaluate_classifier
from src.evaluation_artifact import (
    load_evaluation_artifact_snapshot,
    load_scientific_protocol,
)
from src.local_training import (
    ClientShardSnapshot,
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
REGISTERED_PARTITION = "iid_stratified"

ReviewRow: TypeAlias = tuple[str, str, int]
RowSet: TypeAlias = tuple[ReviewRow, ...]
RowSplit: TypeAlias = tuple[RowSet, RowSet]
ArrayPair: TypeAlias = tuple[np.ndarray, np.ndarray]


class _EpochOrderedBatches(keras.utils.PyDataset):
    """Serve deterministic protocol-ordered batches across fit epochs."""

    def __init__(
        self,
        arrays: ArrayPair,
        orders: list[np.ndarray],
        batch_size: int,
    ) -> None:
        """Initialize fixed arrays and one registered order per epoch.

        Parameters
        ----------
        arrays : tuple of numpy.ndarray
            Token IDs and labels in ascending official row order.
        orders : list of numpy.ndarray
            Row positions in registered training order for every epoch.
        batch_size : int
            Positive registered batch size.

        Returns
        -------
        None
        """
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)
        self._x, self._y = arrays
        self._orders = orders
        self._batch_size = batch_size
        self._epoch = 0
        self._order = orders[0]

    def __len__(self) -> int:
        """Return the number of batches in one epoch.

        Returns
        -------
        int
            Ceiling of fitted rows divided by the batch size.
        """
        return math.ceil(len(self._order) / self._batch_size)

    def __getitem__(self, index: int) -> ArrayPair:
        """Return one consecutive batch from the active epoch order.

        Parameters
        ----------
        index : int
            Zero-based batch index.

        Returns
        -------
        tuple of numpy.ndarray
            Token IDs and labels for the requested batch.
        """
        start = index * self._batch_size
        positions = self._order[start : start + self._batch_size]
        return self._x[positions], self._y[positions]

    def on_epoch_begin(self) -> None:
        """Select the registered row order for the next epoch.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If Keras requests more epochs than were registered.
        """
        if self._epoch >= len(self._orders):
            raise ValueError("training requested more epochs than were registered")
        self._order = self._orders[self._epoch]
        self._epoch += 1


def _derive_seed(master_seed: int, namespace: str, attempt: int = 0) -> int:
    """Derive one frozen-protocol unsigned 64-bit seed.

    Parameters
    ----------
    master_seed : int
        Experiment master seed.
    namespace : str
        Fully expanded ASCII stochastic-operation namespace.
    attempt : int, optional
        Zero-based retry attempt.

    Returns
    -------
    int
        Unsigned 64-bit value from the first eight SHA-256 digest bytes.
    """
    material = f"{master_seed}|{namespace}|{attempt}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _generator(master_seed: int, namespace: str) -> np.random.Generator:
    """Return a fresh PCG64 generator for one frozen namespace.

    Parameters
    ----------
    master_seed : int
        Experiment master seed.
    namespace : str
        Fully expanded stochastic-operation namespace.

    Returns
    -------
    numpy.random.Generator
        Fresh generator that must be discarded after the operation.
    """
    return np.random.Generator(np.random.PCG64(_derive_seed(master_seed, namespace)))


def _tensorflow_seed(derived_seed: int) -> int:
    """Convert one derived uint64 into the registered TensorFlow seed.

    Parameters
    ----------
    derived_seed : int
        Frozen-protocol unsigned 64-bit seed.

    Returns
    -------
    int
        Positive TensorFlow-compatible seed.
    """
    return derived_seed % 2_147_483_647 or 1


def _validate_args(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    """Validate baseline command arguments before loading artifacts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed baseline command arguments.
    protocol : mapping of str to Any
        Frozen scientific protocol.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If arguments differ from the registered baseline contract.
    """
    if args.baseline not in {"centralized", "local-only"}:
        raise ValueError("baseline must be centralized or local-only")
    expected = {
        "batch_size": protocol["training"]["batch_size"],
        "client_count": protocol["strategies"]["local_only"]["client_scale"],
        "epochs": protocol["strategies"]["centralized"]["training_epochs"],
        "validation_split": protocol["training"]["validation_fraction"],
    }
    for name, value in expected.items():
        if type(getattr(args, name)) is not type(value) or getattr(args, name) != value:
            raise ValueError(
                f"{name.replace('_', '-')} must equal the frozen value {value}"
            )
    if type(args.seed) is not int or args.seed not in protocol["seeding"]["seeds"]:
        raise ValueError("seed must be one of the frozen registered seeds")
    _validate_client_data_template(args.client_data_dir)


def _validate_client_data_template(value: object) -> None:
    """Require the client directory template to expose only ``partition``.

    Parameters
    ----------
    value : object
        Candidate path template.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the template is malformed or has another replacement field.
    """
    template = str(value)
    try:
        fields = [
            field
            for _, field, _, _ in string.Formatter().parse(template)
            if field is not None
        ]
    except ValueError as error:
        raise ValueError("client-data-dir must be a valid format template") from error
    if not fields or any(field != "partition" for field in fields):
        raise ValueError("client-data-dir must contain {partition}")
    try:
        template.format(partition=0)
    except (AttributeError, IndexError, KeyError, ValueError) as error:
        raise ValueError(
            "client-data-dir must be a valid template using {partition}"
        ) from error


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


def _registered_iid_splits(
    snapshots: list[ClientShardSnapshot],
    protocol: Mapping[str, Any],
    master_seed: int,
    client_count: int,
    validation_split: float,
) -> list[RowSplit]:
    """Repartition the validated train union using the frozen IID policy.

    Parameters
    ----------
    snapshots : list of ClientShardSnapshot
        Prepared shards whose union is the complete official train split.
    protocol : mapping of str to Any
        Frozen scientific protocol.
    master_seed : int
        Experiment master seed.
    client_count : int
        Number of IID client partitions.
    validation_split : float
        Per-label validation fraction.

    Returns
    -------
    list of tuple
        Ascending fitted and validation row tuples for every client.
    """
    rows_by_index = {
        int(row[0].removeprefix("train:")): row
        for snapshot in snapshots
        for row in snapshot.rows
    }
    labels = protocol["partitioning"]["labels"]
    iid_template = protocol["seeding"]["namespaces"]["partition_iid"]
    allocations: list[list[int]] = [[] for _ in range(client_count)]
    for label in labels:
        indices = np.asarray(
            sorted(index for index, row in rows_by_index.items() if row[2] == label),
            dtype=np.int64,
        )
        namespace = iid_template.format(client_scale=client_count, label=label)
        shuffled = _generator(master_seed, namespace).permutation(indices)
        for client_id, part in enumerate(np.array_split(shuffled, client_count)):
            allocations[client_id].extend(int(index) for index in part)

    validation_template = protocol["seeding"]["namespaces"]["validation"]
    splits: list[RowSplit] = []
    for client_id, allocation in enumerate(allocations):
        fitted: list[int] = []
        validation: list[int] = []
        for label in labels:
            indices = np.asarray(
                sorted(
                    index for index in allocation if rows_by_index[index][2] == label
                ),
                dtype=np.int64,
            )
            namespace = validation_template.format(
                partition_name=REGISTERED_PARTITION,
                client_scale=client_count,
                client_id=client_id,
                label=label,
            )
            shuffled = _generator(master_seed, namespace).permutation(indices)
            validation_count = math.floor(len(indices) * validation_split)
            validation.extend(int(index) for index in shuffled[:validation_count])
            fitted.extend(int(index) for index in shuffled[validation_count:])
        splits.append(
            (
                tuple(rows_by_index[index] for index in sorted(fitted)),
                tuple(rows_by_index[index] for index in sorted(validation)),
            )
        )
    return splits


def _tokenize_split(
    split: RowSplit, manifest: AppManifest
) -> tuple[ArrayPair, ArrayPair]:
    """Tokenize one registered fitted/validation row split.

    Parameters
    ----------
    split : tuple of tuple
        Ascending fitted and validation rows.
    manifest : AppManifest
        Frozen vocabulary and model dimensions.

    Returns
    -------
    tuple of tuple
        Tokenized fitted and validation arrays.
    """
    return tokenize_rows(split[0], manifest), tokenize_rows(split[1], manifest)


def _combined_split(
    splits: list[RowSplit], manifest: AppManifest
) -> tuple[RowSplit, tuple[ArrayPair, ArrayPair]]:
    """Union registered client splits in ascending official row order.

    Parameters
    ----------
    splits : list of tuple
        Registered fitted and validation rows per client.
    manifest : AppManifest
        Frozen vocabulary and model dimensions.

    Returns
    -------
    tuple
        Combined row split and its tokenized fitted/validation arrays.
    """
    combined_rows: RowSplit = (
        tuple(
            sorted(
                (row for split in splits for row in split[0]),
                key=lambda row: int(row[0].removeprefix("train:")),
            )
        ),
        tuple(
            sorted(
                (row for split in splits for row in split[1]),
                key=lambda row: int(row[0].removeprefix("train:")),
            )
        ),
    )
    return combined_rows, _tokenize_split(combined_rows, manifest)


def _cell_id(args: argparse.Namespace) -> str:
    """Serialize the registered baseline cell identifier.

    Parameters
    ----------
    args : argparse.Namespace
        Validated baseline arguments.

    Returns
    -------
    str
        Compact sorted JSON cell identifier.
    """
    centralized = args.baseline == "centralized"
    return json.dumps(
        {
            "alpha": None,
            "client_scale": None if centralized else args.client_count,
            "matrix_kind": "centralized" if centralized else "local_only",
            "partition": None if centralized else REGISTERED_PARTITION,
            "seed": args.seed,
            "strategy": "centralized" if centralized else "local_only",
            "threat": None,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _training_orders(
    rows: RowSet,
    protocol: Mapping[str, Any],
    master_seed: int,
    cell_id: str,
    client_id: int,
    epochs: int,
    *,
    round_index: int = -1,
) -> list[np.ndarray]:
    """Return registered fitted-row positions for every epoch.

    Parameters
    ----------
    rows : tuple of tuple
        Fitted rows in ascending official row order.
    protocol : mapping of str to Any
        Frozen scientific protocol.
    master_seed : int
        Experiment master seed.
    cell_id : str
        Canonical baseline cell identifier.
    client_id : int
        Client ID, or ``-1`` for centralized training.
    epochs : int
        Number of epoch orders to derive.
    round_index : int, optional
        Federated round index, or ``-1`` for baseline training.

    Returns
    -------
    list of numpy.ndarray
        Row-position permutations, one per epoch.
    """
    official_indices = np.asarray(
        [int(row[0].removeprefix("train:")) for row in rows], dtype=np.int64
    )
    positions = {
        int(index): position for position, index in enumerate(official_indices)
    }
    template = protocol["seeding"]["namespaces"]["training_order"]
    orders = []
    for epoch in range(epochs):
        namespace = template.format(
            cell_id=cell_id,
            round_index=round_index,
            epoch_index=epoch,
            client_id=client_id,
        )
        shuffled = _generator(master_seed, namespace).permutation(official_indices)
        orders.append(
            np.asarray([positions[int(index)] for index in shuffled], dtype=np.int64)
        )
    return orders


def _model_seeds(
    protocol: Mapping[str, Any], master_seed: int, cell_id: str, client_id: int
) -> tuple[int, int]:
    """Derive registered TensorFlow model and Dropout seeds.

    Parameters
    ----------
    protocol : mapping of str to Any
        Frozen scientific protocol.
    master_seed : int
        Experiment master seed.
    cell_id : str
        Canonical baseline cell identifier.
    client_id : int
        Client ID, or ``-1`` for centralized training.

    Returns
    -------
    tuple of int
        Model-initialization and Dropout TensorFlow seeds.
    """
    namespaces = protocol["seeding"]["namespaces"]
    model_namespace = namespaces["model_initialization"].format(
        cell_id=cell_id, client_id=client_id
    )
    dropout_namespace = namespaces["dropout"].format(
        cell_id=cell_id, round_index=-1, client_id=client_id
    )
    return (
        _tensorflow_seed(_derive_seed(master_seed, model_namespace)),
        _tensorflow_seed(_derive_seed(master_seed, dropout_namespace)),
    )


def _result_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return effective baseline inputs recorded with every result.

    Parameters
    ----------
    args : argparse.Namespace
        Validated baseline arguments.

    Returns
    -------
    dict of str to Any
        JSON-compatible effective training configuration.
    """
    return {
        "batch_size": args.batch_size,
        "client_count": args.client_count,
        "epochs": args.epochs,
        "partition": REGISTERED_PARTITION,
        "seed": args.seed,
        "validation_split": args.validation_split,
    }


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
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one trained model on the untouched test artifact.

    Parameters
    ----------
    model : Any
        Trained Keras model.
    test_data : tuple of numpy.ndarray
        Tokenized untouched test examples and labels.

    Returns
    -------
    tuple of dict and numpy.ndarray
        JSON-compatible metrics and raw float32 probabilities.
    """
    result = evaluate_classifier(model, *test_data)
    metrics = {
        name: result[name]
        for name in (
            "accuracy",
            "confusion_matrix",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "roc_auc_status",
        )
    }
    return metrics, result["probabilities"]


def _fit_model(
    model: Any,
    fitted_rows: RowSet,
    train_data: ArrayPair,
    validation_data: ArrayPair,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    cell_id: str,
    client_id: int,
) -> Any:
    """Fit one baseline model without test access or framework reshuffling.

    Parameters
    ----------
    model : Any
        Compiled Keras model.
    fitted_rows : tuple of tuple
        Fitted rows in ascending official row order.
    train_data : tuple of numpy.ndarray
        Tokenized fitted rows in the same order.
    validation_data : tuple of numpy.ndarray
        Fixed validation rows.
    args : argparse.Namespace
        Validated baseline command arguments.
    protocol : mapping of str to Any
        Frozen scientific protocol.
    cell_id : str
        Canonical baseline cell identifier.
    client_id : int
        Client ID, or ``-1`` for centralized training.

    Returns
    -------
    Any
        Keras training history.
    """
    batches = _EpochOrderedBatches(
        train_data,
        _training_orders(
            fitted_rows,
            protocol,
            args.seed,
            cell_id,
            client_id,
            args.epochs,
        ),
        args.batch_size,
    )
    return model.fit(
        batches,
        validation_data=validation_data,
        epochs=args.epochs,
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
    token_ids, _ = tokenize_rows(snapshot.rows, manifest)
    labels = np.asarray([row[2] for row in snapshot.rows], dtype=np.int64)
    return token_ids, labels


def _run_centralized(
    args: argparse.Namespace,
    manifest: AppManifest,
    protocol: Mapping[str, Any],
    splits: list[RowSplit],
) -> tuple[dict[str, Any], list[Any], list[np.ndarray]]:
    """Train and evaluate the centralized baseline.

    Parameters
    ----------
    args : argparse.Namespace
        Validated baseline command arguments.
    manifest : AppManifest
        Public vocabulary and model contract.
    protocol : mapping of str to Any
        Frozen scientific protocol.
    splits : list of tuple
        Registered fitted and validation rows for every client.

    Returns
    -------
    tuple of dict and list
        JSON result payload, trained model list, and raw predictions.
    """
    rows, (train_data, validation_data) = _combined_split(splits, manifest)
    cell_id = _cell_id(args)
    model_seed, dropout_seed = _model_seeds(protocol, args.seed, cell_id, -1)
    keras.utils.set_random_seed(model_seed)
    model = build_model_from_manifest(manifest, dropout_seed=dropout_seed)
    history = _fit_model(
        model,
        rows[0],
        train_data,
        validation_data,
        args,
        protocol,
        cell_id,
        -1,
    )
    test_metrics, probabilities = _evaluate_model(
        model, _load_test_data(args, manifest)
    )
    result = {
        "baseline": "centralized",
        "config": _result_config(args),
        "seeds": {
            "dropout": dropout_seed,
            "model_initialization": model_seed,
        },
        "validation": _validation_curve(history, args.epochs),
        "test": test_metrics,
        "models": ["centralized.keras"],
        "predictions": ["centralized-predictions.npy"],
    }
    return result, [model], [probabilities]


def _run_local_only(
    args: argparse.Namespace,
    manifest: AppManifest,
    protocol: Mapping[str, Any],
    splits: list[RowSplit],
) -> tuple[dict[str, Any], list[Any], list[np.ndarray]]:
    """Train and evaluate one independent model per client.

    Parameters
    ----------
    args : argparse.Namespace
        Validated baseline command arguments.
    manifest : AppManifest
        Public vocabulary and model contract.
    protocol : mapping of str to Any
        Frozen scientific protocol.
    splits : list of tuple
        Registered fitted and validation rows for every client.

    Returns
    -------
    tuple of dict and list
        JSON result payload, trained models, and raw predictions.
    """
    models: list[Any] = []
    clients: list[dict[str, Any]] = []
    cell_id = _cell_id(args)
    for client_id, split in enumerate(splits):
        train_data, validation_data = _tokenize_split(split, manifest)
        model_seed, dropout_seed = _model_seeds(protocol, args.seed, cell_id, client_id)
        keras.utils.set_random_seed(model_seed)
        model = build_model_from_manifest(manifest, dropout_seed=dropout_seed)
        history = _fit_model(
            model,
            split[0],
            train_data,
            validation_data,
            args,
            protocol,
            cell_id,
            client_id,
        )
        models.append(model)
        clients.append(
            {
                "client_id": client_id,
                "seeds": {
                    "dropout": dropout_seed,
                    "model_initialization": model_seed,
                },
                "validation": _validation_curve(history, args.epochs),
                "model": f"client-{client_id}.keras",
            }
        )

    test_data = _load_test_data(args, manifest)
    predictions: list[np.ndarray] = []
    for client, model in zip(clients, models):
        client["test"], probabilities = _evaluate_model(model, test_data)
        client["predictions"] = f"client-{client['client_id']}-predictions.npy"
        predictions.append(probabilities)
    result = {
        "baseline": "local_only",
        "config": _result_config(args),
        "clients": clients,
        "test_mean": {
            metric: fmean(client["test"][metric] for client in clients)
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
        },
        "models": [client["model"] for client in clients],
        "predictions": [client["predictions"] for client in clients],
    }
    return result, models, predictions


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
    protocol = load_scientific_protocol()
    _validate_args(args, protocol)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        manifest = load_app_manifest(public_artifact_dir=args.public_artifact_dir)
        snapshots = _load_client_snapshots(args, manifest)
        splits = _registered_iid_splits(
            snapshots,
            protocol,
            args.seed,
            args.client_count,
            args.validation_split,
        )
        result, models, predictions = (
            _run_centralized(args, manifest, protocol, splits)
            if args.baseline == "centralized"
            else _run_local_only(args, manifest, protocol, splits)
        )
        for filename, model in zip(result["models"], models):
            model.save(str(output_dir / filename))
        for filename, probabilities in zip(result["predictions"], predictions):
            np.save(output_dir / filename, probabilities, allow_pickle=False)
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
    parser.add_argument(
        "--public-artifact-dir", type=Path, default=default_public_artifact_dir()
    )
    parser.add_argument(
        "--evaluation-artifact-dir",
        type=Path,
        default=default_evaluation_artifact_dir(),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASELINE_SEED)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(
        batch_size=64,
        client_count=DEFAULT_BASELINE_CLIENTS,
        epochs=DEFAULT_BASELINE_EPOCHS,
        validation_split=0.2,
    )
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
