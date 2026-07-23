"""Run the registered local-only and federated strategy contracts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns
from typing import Any, Mapping, TypeAlias

import keras
import numpy as np
import tensorflow as tf
from flwr.common import serde
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from flwr.server.strategy.aggregate import (
    aggregate_inplace,
    aggregate_median,
    aggregate_trimmed_avg,
)

from src.app_manifest import AppManifest, load_app_manifest
from src.artifact_compatibility import canonical_json_bytes
from src.baseline_training import (
    DEFAULT_BASELINE_CLIENTS,
    DEFAULT_BASELINE_EPOCHS,
    DEFAULT_BASELINE_SEED,
    REGISTERED_PARTITION,
    RowSet,
    RowSplit,
    _EpochOrderedBatches,
    _derive_seed,
    _load_client_snapshots,
    _model_seeds,
    _registered_partition_splits,
    _tensorflow_seed,
    _tokenize_split,
    _training_orders,
    _validate_client_data_template,
)
from src.classification_metrics import (
    LOCAL_ONLY_VALIDATION_SCOPE,
    evaluate_classifier,
)
from src.evaluation_artifact import (
    load_evaluation_artifact_snapshot,
    load_scientific_protocol,
)
from src.huber_strategy import _flatten, _unflatten, huber_aggregate
from src.local_training import build_model_from_manifest, tokenize_rows
from src.paths import default_evaluation_artifact_dir, default_public_artifact_dir

NDArrays: TypeAlias = list[np.ndarray]
ArrayPair: TypeAlias = tuple[np.ndarray, np.ndarray]
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


def _validate_aggregated_weights(aggregate: NDArrays, reference: NDArrays) -> NDArrays:
    """Validate aggregated model weights against the reference model.

    Parameters
    ----------
    aggregate : list of numpy.ndarray
        Candidate aggregate in model-weight order.
    reference : list of numpy.ndarray
        Validated reference model weights.

    Returns
    -------
    list of numpy.ndarray
        The validated aggregate unchanged.

    Raises
    ------
    ValueError
        If the aggregate has an invalid count, shape, dtype, size, or value.
    """
    if len(aggregate) != len(reference):
        raise ValueError("aggregate model weight count must match the reference model")
    for array, expected in zip(aggregate, reference, strict=True):
        if not isinstance(array, np.ndarray):
            raise ValueError("aggregate model weights must be numpy arrays")
        if array.shape != expected.shape:
            raise ValueError(
                "aggregate model weight shapes must match the reference model"
            )
        if array.dtype != np.dtype(np.float32):
            raise ValueError("aggregate model weights must use exactly float32")
        if array.ndim == 0 or array.size == 0:
            raise ValueError("aggregate model weights must be non-scalar and nonempty")
        if not np.all(np.isfinite(array)):
            raise ValueError("aggregate model weights must contain finite values")
    return aggregate


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
        aggregated = _unflatten(
            huber_aggregate(
                [_flatten(weights) for weights in client_weights],
                sample_counts,
                huber_threshold,
            ),
            client_weights[0],
        )
    else:
        weighted_models = list(zip(client_weights, sample_counts, strict=True))
        if strategy == "fedmedian":
            aggregated = aggregate_median(weighted_models)
        elif strategy == "fedtrimmedavg":
            aggregated = aggregate_trimmed_avg(
                weighted_models, proportiontocut=trimmed_fraction
            )
        else:
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
            aggregated = aggregate_inplace(results)
    return _validate_aggregated_weights(aggregated, client_weights[0])


def _cell_id(
    strategy: str,
    seed: int,
    partition: str = REGISTERED_PARTITION,
    client_count: int = DEFAULT_BASELINE_CLIENTS,
) -> str:
    """Return the canonical identifier for one strategy contract run.

    Parameters
    ----------
    strategy : str
        Registered strategy identifier.
    seed : int
        Registered master seed.
    partition : str, optional
        Registered partition name.
    client_count : int, optional
        Registered client scale.

    Returns
    -------
    str
        Compact sorted JSON cell identifier.
    """
    return json.dumps(
        {
            "alpha": (
                None
                if partition == REGISTERED_PARTITION
                else float(partition.removeprefix("dirichlet_"))
            ),
            "client_scale": client_count,
            "matrix_kind": (
                "local_only" if strategy == "local_only" else "primary_federated"
            ),
            "partition": partition,
            "seed": seed,
            "strategy": strategy,
            "threat": None,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _dropout_seed(
    protocol: Mapping[str, Any],
    master_seed: int,
    cell_id: str,
    round_index: int,
    client_id: int,
) -> int:
    """Derive one registered round-scoped Dropout seed.

    Parameters
    ----------
    protocol : mapping of str to Any
        Frozen scientific protocol.
    master_seed : int
        Registered master seed.
    cell_id : str
        Canonical cell identifier.
    round_index : int
        Zero-based federated round, or ``-1`` for local-only.
    client_id : int
        Zero-based client identifier.

    Returns
    -------
    int
        Positive TensorFlow-compatible seed.
    """
    namespace = protocol["seeding"]["namespaces"]["dropout"].format(
        cell_id=cell_id,
        round_index=round_index,
        client_id=client_id,
    )
    return _tensorflow_seed(_derive_seed(master_seed, namespace))


def _validation_data(
    split: RowSplit, manifest: AppManifest
) -> tuple[ArrayPair, ArrayPair]:
    """Tokenize one split for training and canonical validation.

    Parameters
    ----------
    split : tuple of tuple
        Ascending fitted and fixed validation rows.
    manifest : AppManifest
        Frozen vocabulary and model dimensions.

    Returns
    -------
    tuple of tuple
        Float32-label training arrays and int64-label validation arrays.
    """
    train_data, validation_data = _tokenize_split(split, manifest)
    labels = np.asarray([row[2] for row in split[1]], dtype=np.int64)
    return train_data, (validation_data[0], labels)


def _train_one_epoch(
    model: Any,
    train_data: ArrayPair,
    order: np.ndarray,
    batch_size: int,
    *,
    proximal_mu: float = 0.0,
    quiet: bool = False,
) -> None:
    """Train one deterministic local epoch.

    Parameters
    ----------
    model : Any
        Fresh or retained compiled Keras model.
    train_data : tuple of numpy.ndarray
        Token IDs and float32 labels in ascending official row order.
    order : numpy.ndarray
        Registered row-position permutation for this epoch.
    batch_size : int
        Registered batch size.
    proximal_mu : float, optional
        FedProx coefficient, or zero for the ordinary local objective.
    quiet : bool, optional
        Suppress Keras progress output.

    Returns
    -------
    None
    """
    from src.baseline_training import _EpochOrderedBatches

    batches = _EpochOrderedBatches(train_data, [order], batch_size)
    if proximal_mu == 0.0:
        model.fit(
            batches,
            epochs=1,
            shuffle=False,
            verbose=0 if quiet else 1,
        )
        return

    batches.on_epoch_begin()
    references = [
        keras.ops.convert_to_tensor(weight.numpy().copy())
        for weight in model.trainable_weights
    ]
    loss_function = keras.losses.BinaryCrossentropy()
    for batch_index in range(len(batches)):
        token_ids, labels = batches[batch_index]
        with tf.GradientTape() as tape:
            probabilities = model(token_ids, training=True)
            objective = loss_function(labels, probabilities)
            proximal_terms = [
                keras.ops.sum(keras.ops.square(weight - reference))
                for weight, reference in zip(
                    model.trainable_weights, references, strict=True
                )
            ]
            proximal = proximal_terms[0]
            for term in proximal_terms[1:]:
                proximal = proximal + term
            objective = objective + proximal * (proximal_mu / 2)
        gradients = tape.gradient(objective, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables, strict=True)
        )


def _evaluation_record(
    model: Any,
    data: ArrayPair,
    output_dir: Path,
    prediction_name: str,
    *,
    local_only: bool = False,
) -> dict[str, Any]:
    """Evaluate and persist one canonical classification result.

    Parameters
    ----------
    model : Any
        Model state immediately after the registered epoch.
    data : tuple of numpy.ndarray
        Fixed token IDs and exact int64 labels.
    output_dir : pathlib.Path
        Owned result directory.
    prediction_name : str
        Relative ``.npy`` filename for raw probabilities.
    local_only : bool, optional
        Permit undefined ROC only when these labels are single-class.

    Returns
    -------
    dict of str to Any
        JSON-compatible reported metrics and raw-prediction path.
    """
    if local_only and np.unique(data[1]).size == 1:
        result = evaluate_classifier(
            model,
            *data,
            evaluation_scope=LOCAL_ONLY_VALIDATION_SCOPE,
        )
    else:
        result = evaluate_classifier(model, *data)
    np.save(output_dir / prediction_name, result["probabilities"], allow_pickle=False)
    return {
        **{
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
        },
        "predictions": prediction_name,
    }


def _parameter_payload_bytes(weights: NDArrays) -> int:
    """Return exact Flower ``Parameters`` protobuf bytes for model weights.

    Parameters
    ----------
    weights : list of numpy.ndarray
        Complete model tensors in model order.

    Returns
    -------
    int
        Serialized protobuf byte count, excluding message metadata and transport
        framing.
    """
    return len(
        serde.parameters_to_proto(ndarrays_to_parameters(weights)).SerializeToString()
    )


def _convergence_round(curve: list[dict[str, Any]]) -> int:
    """Return the first round reaching 95 percent of peak validation accuracy.

    Parameters
    ----------
    curve : list of dict of str to Any
        Registered per-round validation records.

    Returns
    -------
    int
        Zero-based first qualifying round.
    """
    target = max(float(record["accuracy"]) for record in curve) * 0.95
    return next(
        int(record["round"]) for record in curve if float(record["accuracy"]) >= target
    )


class _LocalOnlyValidationCallback(keras.callbacks.Callback):
    """Record canonical validation metrics from each live epoch model."""

    def __init__(
        self,
        data: ArrayPair,
        output_dir: Path,
        client_id: int,
    ) -> None:
        """Initialize one client's fixed validation callback.

        Parameters
        ----------
        data : tuple of numpy.ndarray
            Fixed validation token IDs and exact int64 labels.
        output_dir : pathlib.Path
            Owned result directory.
        client_id : int
            Zero-based client identifier.

        Returns
        -------
        None
        """
        super().__init__()
        self._data = data
        self._output_dir = output_dir
        self._client_id = client_id
        self.curve: list[dict[str, Any]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Evaluate the live model after one completed epoch.

        Parameters
        ----------
        epoch : int
            Zero-based completed epoch index.
        logs : dict of str to Any or None, optional
            Keras epoch logs, unused because reporting is canonical.

        Returns
        -------
        None
        """
        self.curve.append(
            {
                "epoch": epoch,
                **_evaluation_record(
                    self.model,
                    self._data,
                    self._output_dir,
                    f"client-{self._client_id}-validation-epoch-{epoch}.npy",
                    local_only=True,
                ),
            }
        )


def _load_test_data(
    args: argparse.Namespace, manifest: AppManifest
) -> tuple[np.ndarray, np.ndarray]:
    """Load the untouched test artifact after all training completes.

    Parameters
    ----------
    args : argparse.Namespace
        Validated runner arguments.
    manifest : AppManifest
        Frozen public model and vocabulary contract.

    Returns
    -------
    tuple of numpy.ndarray
        Token IDs and exact int64 labels in official test order.
    """
    rows = load_evaluation_artifact_snapshot(args.evaluation_artifact_dir).rows
    token_ids, _ = tokenize_rows(rows, manifest)
    return token_ids, np.asarray([row[2] for row in rows], dtype=np.int64)


def _run_local_only(
    args: argparse.Namespace,
    manifest: AppManifest,
    protocol: Mapping[str, Any],
    splits: list[RowSplit],
    output_dir: Path,
) -> tuple[dict[str, Any], list[Any]]:
    """Run the four registered independent client models.

    Parameters
    ----------
    args : argparse.Namespace
        Validated runner arguments.
    manifest : AppManifest
        Frozen public model and vocabulary contract.
    protocol : mapping of str to Any
        Frozen scientific protocol.
    splits : list of tuple
        Fixed fitted and validation rows per client.
    output_dir : pathlib.Path
        Owned result directory.

    Returns
    -------
    tuple of dict and list
        Result payload and trained models.
    """
    cell_id = _cell_id(
        args.strategy,
        args.seed,
        args.partition,
        args.client_count,
    )
    models: list[Any] = []
    clients: list[dict[str, Any]] = []
    for client_id, split in enumerate(splits):
        train_data, validation_data = _validation_data(split, manifest)
        model_seed, _ = _model_seeds(protocol, args.seed, cell_id, client_id)
        dropout_seed = _dropout_seed(protocol, args.seed, cell_id, -1, client_id)
        keras.utils.set_random_seed(model_seed)
        model = build_model_from_manifest(manifest, dropout_seed=dropout_seed)
        orders = _training_orders(
            split[0],
            protocol,
            args.seed,
            cell_id,
            client_id,
            args.epochs,
        )
        callback = _LocalOnlyValidationCallback(validation_data, output_dir, client_id)
        started_at = perf_counter_ns()
        try:
            model.fit(
                _EpochOrderedBatches(train_data, orders, args.batch_size),
                callbacks=[callback],
                epochs=args.epochs,
                shuffle=False,
                verbose=0 if args.quiet else 1,
            )
        finally:
            training_time_ns = perf_counter_ns() - started_at
        models.append(model)
        clients.append(
            {
                "client_id": client_id,
                "model": f"client-{client_id}.keras",
                "seeds": {
                    "dropout": dropout_seed,
                    "model_initialization": model_seed,
                },
                "training_time_ns": training_time_ns,
                "validation": callback.curve,
            }
        )

    test_data = _load_test_data(args, manifest)
    for client, model in zip(clients, models, strict=True):
        client_id = client["client_id"]
        client["test"] = _evaluation_record(
            model,
            test_data,
            output_dir,
            f"client-{client_id}-test.npy",
        )
    result = {
        "strategy": args.strategy,
        "config": _result_config(args),
        "clients": clients,
        "test_mean": {
            metric: fmean(client["test"][metric] for client in clients)
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
        },
        "models": [client["model"] for client in clients],
        "system": {
            "training_time_ns": sum(
                int(client["training_time_ns"]) for client in clients
            )
        },
    }
    return result, models


def _federated_model_seed(
    protocol: Mapping[str, Any], master_seed: int, cell_id: str, client_id: int
) -> int:
    """Derive one registered model-construction seed.

    Parameters
    ----------
    protocol : mapping of str to Any
        Frozen scientific protocol.
    master_seed : int
        Registered master seed.
    cell_id : str
        Canonical cell identifier.
    client_id : int
        Client ID, or ``-1`` for the global model.

    Returns
    -------
    int
        Positive TensorFlow-compatible seed.
    """
    namespace = protocol["seeding"]["namespaces"]["model_initialization"].format(
        cell_id=cell_id, client_id=client_id
    )
    return _tensorflow_seed(_derive_seed(master_seed, namespace))


def _run_federated(
    args: argparse.Namespace,
    manifest: AppManifest,
    protocol: Mapping[str, Any],
    splits: list[RowSplit],
    output_dir: Path,
) -> tuple[dict[str, Any], list[Any]]:
    """Run one registered four-client federated strategy.

    Parameters
    ----------
    args : argparse.Namespace
        Validated runner arguments.
    manifest : AppManifest
        Frozen public model and vocabulary contract.
    protocol : mapping of str to Any
        Frozen scientific protocol.
    splits : list of tuple
        Fixed fitted and validation rows per client.
    output_dir : pathlib.Path
        Owned result directory.

    Returns
    -------
    tuple of dict and list
        Result payload and final global model list.
    """
    cell_id = _cell_id(
        args.strategy,
        args.seed,
        args.partition,
        args.client_count,
    )
    global_seed = _federated_model_seed(protocol, args.seed, cell_id, -1)
    keras.utils.set_random_seed(global_seed)
    global_model = build_model_from_manifest(
        manifest,
        dropout_seed=_dropout_seed(protocol, args.seed, cell_id, -1, -1),
    )
    combined_validation_rows: RowSet = tuple(
        sorted(
            (row for split in splits for row in split[1]),
            key=lambda row: int(row[0].removeprefix("train:")),
        )
    )
    validation_tokens, _ = tokenize_rows(combined_validation_rows, manifest)
    validation_data = (
        validation_tokens,
        np.asarray([row[2] for row in combined_validation_rows], dtype=np.int64),
    )
    tokenized = [_validation_data(split, manifest)[0] for split in splits]
    sample_counts = [len(split[0]) for split in splits]
    proximal_mu = (
        float(protocol["strategies"][args.strategy]["mu"])
        if args.strategy in {"fedprox", "fedprox_huber"}
        else 0.0
    )
    curve = []
    client_training_time_ns = 0
    server_to_client_bytes = 0
    client_to_server_bytes = 0
    round_durations_ns: list[int] = []
    for round_index in range(args.epochs):
        round_started_at = perf_counter_ns()
        round_start = global_model.get_weights()
        server_to_client_bytes += _parameter_payload_bytes(round_start) * len(splits)
        client_weights = []
        for client_id, (split, train_data) in enumerate(
            zip(splits, tokenized, strict=True)
        ):
            model_seed = _federated_model_seed(protocol, args.seed, cell_id, client_id)
            dropout_seed = _dropout_seed(
                protocol, args.seed, cell_id, round_index, client_id
            )
            keras.utils.set_random_seed(model_seed)
            client_model = build_model_from_manifest(
                manifest, dropout_seed=dropout_seed
            )
            client_model.set_weights(round_start)
            order = _training_orders(
                split[0],
                protocol,
                args.seed,
                cell_id,
                client_id,
                1,
                round_index=round_index,
            )[0]
            started_at = perf_counter_ns()
            try:
                _train_one_epoch(
                    client_model,
                    train_data,
                    order,
                    args.batch_size,
                    proximal_mu=proximal_mu,
                    quiet=args.quiet,
                )
            finally:
                client_training_time_ns += perf_counter_ns() - started_at
            updated_weights = client_model.get_weights()
            client_to_server_bytes += _parameter_payload_bytes(updated_weights)
            client_weights.append(updated_weights)

        global_model.set_weights(
            aggregate_model_weights(
                args.strategy,
                client_weights,
                sample_counts,
                huber_threshold=float(
                    protocol["strategies"]["fedprox_huber"]["threshold"]
                ),
                trimmed_fraction=float(protocol["strategies"]["fedtrimmedavg"]["beta"]),
            )
        )
        curve.append(
            {
                "round": round_index,
                **_evaluation_record(
                    global_model,
                    validation_data,
                    output_dir,
                    f"validation-round-{round_index}.npy",
                ),
            }
        )
        round_durations_ns.append(perf_counter_ns() - round_started_at)

    test = _evaluation_record(
        global_model,
        _load_test_data(args, manifest),
        output_dir,
        "test.npy",
    )
    return (
        {
            "strategy": args.strategy,
            "config": _result_config(args),
            "seeds": {"global_model_initialization": global_seed},
            "validation": curve,
            "test": test,
            "models": ["global.keras"],
            "system": {
                "client_training_time_ns": client_training_time_ns,
                "communication": {
                    "scope": (
                        "serialized Flower Parameters protobufs; excludes message "
                        "metadata, TLS, and transport framing"
                    ),
                    "server_to_client_bytes": server_to_client_bytes,
                    "client_to_server_bytes": client_to_server_bytes,
                    "total_bytes": server_to_client_bytes + client_to_server_bytes,
                },
                "convergence_round": _convergence_round(curve),
                "round_duration_ns": round_durations_ns,
            },
        },
        [global_model],
    )


def _result_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the effective registered run configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Validated runner arguments.

    Returns
    -------
    dict of str to Any
        JSON-compatible configuration.
    """
    return {
        "batch_size": args.batch_size,
        "client_count": args.client_count,
        "epochs": args.epochs,
        "partition": args.partition,
        "seed": args.seed,
        "validation_split": args.validation_split,
    }


def _validate_args(args: argparse.Namespace, protocol: Mapping[str, Any]) -> None:
    """Require one currently executable registered strategy cell.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.
    protocol : mapping of str to Any
        Frozen scientific protocol.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a setting differs from the registered four-client contract.
    """
    if args.strategy not in REGISTERED_STRATEGIES:
        raise ValueError(f"unsupported registered strategy: {args.strategy}")
    expected = {
        "batch_size": protocol["training"]["batch_size"],
        "client_count": protocol["partitioning"]["primary_client_scale"],
        "epochs": protocol["training"]["rounds"],
        "validation_split": protocol["training"]["validation_fraction"],
    }
    for name, value in expected.items():
        if type(getattr(args, name)) is not type(value) or getattr(args, name) != value:
            raise ValueError(
                f"{name.replace('_', '-')} must equal the frozen value {value}"
            )
    if type(args.seed) is not int or args.seed not in protocol["seeding"]["seeds"]:
        raise ValueError("seed must be one of the frozen registered seeds")
    if args.partition not in protocol["partitioning"]["registered"]:
        raise ValueError("partition must be registered by the frozen protocol")
    _validate_client_data_template(args.client_data_dir)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute and persist one registered strategy contract run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.

    Returns
    -------
    dict of str to Any
        Complete JSON-compatible result payload.
    """
    protocol = load_scientific_protocol()
    _validate_args(args, protocol)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        manifest = load_app_manifest(public_artifact_dir=args.public_artifact_dir)
        snapshots = _load_client_snapshots(args, manifest)
        accepted_attempt, splits = _registered_partition_splits(
            snapshots,
            protocol,
            args.seed,
            args.client_count,
            args.validation_split,
            args.partition,
        )
        result, models = (
            _run_local_only(args, manifest, protocol, splits, output_dir)
            if args.strategy == "local_only"
            else _run_federated(args, manifest, protocol, splits, output_dir)
        )
        result["config"]["partition_attempt"] = accepted_attempt
        for filename, model in zip(result["models"], models, strict=True):
            model.save(str(output_dir / filename))
        (output_dir / "results.json").write_bytes(canonical_json_bytes(result))
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse strategy-runner arguments.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    argparse.Namespace
        Parsed registered strategy options.
    """
    parser = argparse.ArgumentParser(
        description="Run one registered four-client strategy contract."
    )
    parser.add_argument("strategy", choices=REGISTERED_STRATEGIES)
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
    parser.add_argument(
        "--partition",
        choices=(
            "iid_stratified",
            "dirichlet_1.0",
            "dirichlet_0.5",
            "dirichlet_0.1",
        ),
        default=REGISTERED_PARTITION,
    )
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(
        batch_size=64,
        client_count=DEFAULT_BASELINE_CLIENTS,
        epochs=DEFAULT_BASELINE_EPOCHS,
        validation_split=0.2,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the selected registered strategy.

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
