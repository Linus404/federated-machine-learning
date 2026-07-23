from __future__ import annotations

import csv
import io
import logging
import math
import os
import tempfile
import warnings
from numbers import Real
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import keras
import numpy as np
from flwr.common import (
    Context,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from flwr.serverapp import ServerApp

from src import parse_run_config_bool
from src.app_manifest import load_app_manifest, resolve_public_artifact_dir
from src.artifact_history import (
    DEFAULT_ARTIFACT_RETENTION_RUNS,
    create_run_artifact_dir,
    prune_run_history,
    publish_completed_run,
)
from src.artifact_compatibility import (
    load_server_artifact_snapshot,
    read_regular_file,
    write_server_artifact_manifest,
)
from src.huber_strategy import (
    DEFAULT_HUBER_THRESHOLD,
    _flatten,
    _unflatten,
    huber_aggregate,
)
from src.local_training import build_model_from_manifest
from src.paths import (
    RunArtifactLock,
    acquire_run_artifact_lock,
    checkpoint_path,
    client_metrics_path,
    default_server_artifact_dir,
    global_model_path,
    metrics_path,
    resolve_dir,
)
from src.structured_logging import log_event, structured_logger

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"keras\..*")

DEFAULT_SERVER_ROUNDS = 20
DEFAULT_EXPECTED_CLIENTS = 4
_PROTOCOL_DTYPE = np.dtype(np.float32)


def weighted_average(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """Aggregate client accuracies by validation sample count."""
    total = sum(num_examples for num_examples, _ in metrics)
    accuracy = (
        sum(num_examples * metric["accuracy"] for num_examples, metric in metrics)
        / total
    )

    return {"accuracy": accuracy}


def _sorted_fit_results(
    results: list[Any],
    expected_client_ids: frozenset[int],
    result_kind: str = "fit",
) -> list[Any]:
    """Validate and order fit results by numeric client identifier.

    Parameters
    ----------
    results : list[Any]
        Flower ``(ClientProxy, FitRes)`` pairs for one round.
    expected_client_ids : frozenset[int]
        Exact client-ID set required for the round.
    result_kind : str, optional
        Result label used in validation errors.

    Returns
    -------
    list[Any]
        A new list sorted by ascending zero-based client ID.

    Raises
    ------
    ValueError
        If the expected set is invalid or a client ID is missing, invalid,
        duplicated, or unexpected.
    """
    if not expected_client_ids or any(
        type(client_id) is not int or client_id < 0 for client_id in expected_client_ids
    ):
        raise ValueError("expected client IDs must be nonempty non-negative integers")
    if expected_client_ids != frozenset(range(len(expected_client_ids))):
        raise ValueError("expected client IDs must be contiguous from zero")
    identified_results: list[tuple[int, Any]] = []
    for result in results:
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(f"every {result_kind} result must be a pair")
        metrics = getattr(result[1], "metrics", None)
        if not isinstance(metrics, dict):
            raise ValueError(f"every {result_kind} result must contain metrics")
        client_id = metrics.get("client_id")
        if type(client_id) is not int:
            raise ValueError(
                f"every {result_kind} result must contain a built-in integer client_id"
            )
        identified_results.append((client_id, result))

    client_ids = [client_id for client_id, _ in identified_results]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError(f"{result_kind} result client_id values must be unique")
    actual_client_ids = frozenset(client_ids)
    if actual_client_ids != expected_client_ids:
        missing = sorted(expected_client_ids - actual_client_ids)
        unexpected = sorted(actual_client_ids - expected_client_ids)
        raise ValueError(
            f"{result_kind} result client IDs must equal the expected set; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return [result for _, result in sorted(identified_results)]


def _validate_evaluate_results(
    results: list[Any], expected_client_ids: frozenset[int]
) -> list[Any]:
    """Validate and order one complete client-evaluation result set.

    Parameters
    ----------
    results : list[Any]
        Flower ``(ClientProxy, EvaluateRes)`` pairs for one round.
    expected_client_ids : frozenset[int]
        Exact client-ID set required for the round.

    Returns
    -------
    list[Any]
        Validated results sorted by ascending client ID.

    Raises
    ------
    ValueError
        If any result is missing, malformed, duplicated, unexpected, or
        contains an invalid sample count, loss, or accuracy.
    """
    ordered_results = _sorted_fit_results(
        results, expected_client_ids, result_kind="evaluation"
    )
    for _, evaluate_result in ordered_results:
        num_examples = getattr(evaluate_result, "num_examples", None)
        if type(num_examples) is not int or num_examples <= 0:
            raise ValueError(
                "evaluation result num_examples must be a positive built-in integer"
            )
        loss = getattr(evaluate_result, "loss", None)
        accuracy = evaluate_result.metrics.get("accuracy")
        if (
            not isinstance(loss, Real)
            or isinstance(loss, bool)
            or not math.isfinite(loss)
            or float(loss) < 0.0
        ):
            raise ValueError(
                "evaluation result loss must be a finite non-negative real number"
            )
        if (
            not isinstance(accuracy, Real)
            or isinstance(accuracy, bool)
            or not math.isfinite(accuracy)
            or not 0.0 <= float(accuracy) <= 1.0
        ):
            raise ValueError(
                "evaluation result accuracy must be a finite real number in [0, 1]"
            )
    return ordered_results


def _validate_aggregated_evaluation(
    loss: float | None, metrics: dict[str, Any]
) -> None:
    """Validate aggregate evaluation values before artifact publication.

    Parameters
    ----------
    loss : float or None
        Sample-weighted aggregate loss.
    metrics : dict[str, Any]
        Aggregate evaluation metrics.

    Raises
    ------
    ValueError
        If aggregate loss or accuracy is missing or outside its permitted range.
    """
    if not isinstance(metrics, dict):
        raise ValueError("aggregate evaluation metrics must be a dictionary")
    accuracy = metrics.get("accuracy")
    if (
        not isinstance(loss, Real)
        or isinstance(loss, bool)
        or not math.isfinite(loss)
        or float(loss) < 0.0
    ):
        raise ValueError(
            "aggregate evaluation loss must be a finite non-negative real number"
        )
    if (
        not isinstance(accuracy, Real)
        or isinstance(accuracy, bool)
        or not math.isfinite(accuracy)
        or not 0.0 <= float(accuracy) <= 1.0
    ):
        raise ValueError(
            "aggregate evaluation accuracy must be a finite real number in [0, 1]"
        )


def _validate_fit_results(
    results: list[Any],
    expected_client_ids: frozenset[int],
    expected_weight_shapes: tuple[tuple[int, ...], ...],
) -> tuple[list[Any], list[list[np.ndarray]]]:
    """Validate one complete round against the runtime aggregation contract.

    Parameters
    ----------
    results : list[Any]
        Flower ``(ClientProxy, FitRes)`` pairs for one round.
    expected_client_ids : frozenset[int]
        Exact client-ID set required for the round.
    expected_weight_shapes : tuple[tuple[int, ...], ...]
        Round-start model-weight shapes in model order.

    Returns
    -------
    tuple[list[Any], list[list[numpy.ndarray]]]
        Ordered fit results and their decoded, validated weights.

    Raises
    ------
    ValueError
        If a sample count or decoded model violates the aggregation contract.
    """
    if not expected_weight_shapes or any(
        not shape or any(dimension <= 0 for dimension in shape)
        for shape in expected_weight_shapes
    ):
        raise ValueError("expected model weights must have non-scalar nonempty shapes")

    ordered_results = _sorted_fit_results(results, expected_client_ids)
    decoded_weights: list[list[np.ndarray]] = []
    for _, fit_result in ordered_results:
        if type(fit_result.num_examples) is not int or fit_result.num_examples <= 0:
            raise ValueError(
                "fit result num_examples must be a positive built-in integer"
            )
        weights = parameters_to_ndarrays(fit_result.parameters)
        if len(weights) != len(expected_weight_shapes):
            raise ValueError("fit result model weight count must match round start")
        for weight, expected_shape in zip(weights, expected_weight_shapes, strict=True):
            if weight.dtype != _PROTOCOL_DTYPE:
                raise ValueError("fit result model weights must use exactly float32")
            if weight.ndim == 0 or weight.size == 0:
                raise ValueError(
                    "fit result model weights must be non-scalar and nonempty"
                )
            if weight.shape != expected_shape:
                raise ValueError(
                    "fit result model weight shapes must match round start"
                )
            if not np.all(np.isfinite(weight)):
                raise ValueError("fit result model weights must contain finite values")
        for name in (
            "training_time_ns",
            "request_parameter_bytes",
            "response_parameter_bytes",
        ):
            value = fit_result.metrics.get(name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(
                    f"fit result {name} must be a non-negative built-in integer"
                )
        decoded_weights.append(weights)
    return ordered_results, decoded_weights


def _validate_aggregated_parameters(
    parameters: Parameters, expected_weight_shapes: tuple[tuple[int, ...], ...]
) -> None:
    """Validate aggregated parameters before publication or reuse.

    Parameters
    ----------
    parameters : Parameters
        Flower parameters returned by the selected aggregation strategy.
    expected_weight_shapes : tuple[tuple[int, ...], ...]
        Required model-weight shapes in model order.

    Raises
    ------
    ValueError
        If the aggregate has an invalid count, dtype, shape, size, or value.
    """
    weights = parameters_to_ndarrays(parameters)
    if len(weights) != len(expected_weight_shapes):
        raise ValueError("aggregate model weight count must match round start")
    for weight, expected_shape in zip(weights, expected_weight_shapes, strict=True):
        if weight.dtype != _PROTOCOL_DTYPE:
            raise ValueError("aggregate model weights must use exactly float32")
        if weight.ndim == 0 or weight.size == 0:
            raise ValueError("aggregate model weights must be non-scalar and nonempty")
        if weight.shape != expected_shape:
            raise ValueError("aggregate model weight shapes must match round start")
        if not np.all(np.isfinite(weight)):
            raise ValueError("aggregate model weights must contain finite values")


def _write_checkpoint(
    artifact_dir: Path, server_round: int, weights: list[np.ndarray]
) -> Path:
    """Atomically persist one round's ordered model tensors.

    Parameters
    ----------
    artifact_dir : pathlib.Path
        Active server run artifact directory.
    server_round : int
        Positive one-based Flower server round.
    weights : list of numpy.ndarray
        Validated model weights in model order.

    Returns
    -------
    pathlib.Path
        Published checkpoint path.
    """
    path = checkpoint_path(artifact_dir, server_round)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("checkpoint directory must be a regular directory")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=".checkpoint-",
            suffix=".npz",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            np.savez(file, *weights)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def _load_checkpoint(
    path: str | Path,
    app_manifest: Any,
    expected_weight_shapes: tuple[tuple[int, ...], ...],
) -> list[np.ndarray]:
    """Load a compatible checkpoint with the exact runtime tensor contract.

    Parameters
    ----------
    path : str or pathlib.Path
        Explicit ``checkpoint-round-XXXXXX.npz`` path from an earlier run.
    app_manifest : Any
        Validated public application manifest for the new run.
    expected_weight_shapes : tuple of tuple of int
        Required model-weight shapes in model order.

    Returns
    -------
    list of numpy.ndarray
        Validated checkpoint tensors in model order.

    Raises
    ------
    ValueError
        If the source layout, public-artifact binding, archive, or any tensor is
        incompatible.
    """
    resolved = resolve_dir(path)
    round_text = resolved.stem.removeprefix("checkpoint-round-")
    if (
        not round_text.isdecimal()
        or int(round_text) <= 0
        or resolved.name != f"checkpoint-round-{int(round_text):06d}.npz"
    ):
        raise ValueError("resume checkpoint must be a generated round checkpoint")
    source_run_dir = resolved.parent
    load_server_artifact_snapshot(source_run_dir, app_manifest=app_manifest)
    try:
        content = read_regular_file(resolved, parent=resolved.parent)
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            expected_names = [
                f"arr_{index}" for index in range(len(expected_weight_shapes))
            ]
            if archive.files != expected_names:
                raise ValueError("checkpoint tensor count or order is invalid")
            weights = [archive[name].copy() for name in expected_names]
    except (OSError, ValueError) as error:
        raise ValueError("resume checkpoint is missing, unsafe, or invalid") from error
    parameters = ndarrays_to_parameters(weights)
    _validate_aggregated_parameters(parameters, expected_weight_shapes)
    return weights


class SentimentServer(FedProx):
    """Run FedProx with optional experimental Huber aggregation and artifacts."""

    expected_client_ids: frozenset[int]
    expected_weight_shapes: tuple[tuple[int, ...], ...]
    experiment_seed: int
    logger: logging.Logger | None = None

    def __init__(
        self,
        app_manifest,
        artifact_dir=None,
        artifact_lock: RunArtifactLock | None = None,
        artifact_root: str | Path | None = None,
        artifact_retention_runs: int = DEFAULT_ARTIFACT_RETENTION_RUNS,
        final_round: int | None = None,
        huber_threshold: float = DEFAULT_HUBER_THRESHOLD,
        use_huber: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.artifact_dir = resolve_dir(artifact_dir or default_server_artifact_dir())
        self._artifact_lock = artifact_lock
        self.artifact_root = resolve_dir(artifact_root or self.artifact_dir)
        self.artifact_retention_runs = artifact_retention_runs
        self.final_round = final_round
        self.app_manifest = app_manifest
        self.huber_threshold = huber_threshold
        self.use_huber = use_huber
        self._round_started_at_ns: dict[int, int] = {}
        write_server_artifact_manifest(
            self.artifact_dir, app_manifest=self.app_manifest
        )

    @property
    def model_path(self) -> Path:
        return global_model_path(self.artifact_dir)

    @property
    def metrics_path(self) -> Path:
        return metrics_path(self.artifact_dir)

    @property
    def client_metrics_path(self) -> Path:
        return client_metrics_path(self.artifact_dir)

    def _write_client_metrics(self, server_round: int, results: list[Any]) -> None:
        if server_round == 1 and self.client_metrics_path.exists():
            self.client_metrics_path.unlink()
        file_exists = self.client_metrics_path.exists()

        with self.client_metrics_path.open("a", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["round", "client_id", "loss", "accuracy", "samples"],
            )
            if not file_exists:
                writer.writeheader()
            for _, result in results:
                client_id = result.metrics.get("client_id")
                if client_id is None:
                    continue
                writer.writerow(
                    {
                        "round": server_round,
                        "client_id": int(client_id),
                        "loss": result.loss,
                        "accuracy": result.metrics.get("accuracy"),
                        "samples": result.num_examples,
                    }
                )

    def _release_artifact_lock(self) -> None:
        """Release this run's writer lock at most once.

        Returns
        -------
        None
            The lock is released when present and then detached.
        """
        artifact_lock = getattr(self, "_artifact_lock", None)
        self._artifact_lock = None
        if artifact_lock is not None:
            artifact_lock.release()

    def configure_fit(self, server_round, parameters, client_manager):
        """Record availability before selecting clients for one fit round.

        Parameters
        ----------
        server_round : int
            One-based Flower server round.
        parameters : Parameters
            Current global model parameters.
        client_manager : Any
            Flower client manager for the active run.

        Returns
        -------
        list[Any]
            Fit instructions selected by the parent strategy.
        """
        round_starts = getattr(self, "_round_started_at_ns", {})
        self._round_started_at_ns = round_starts
        round_starts[server_round] = perf_counter_ns()
        configured = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_instructions in configured:
            fit_instructions.config["server_round"] = server_round
        log_event(
            self.logger,
            logging.INFO,
            "fit round configured",
            "fit_round_configured",
            round=server_round,
            available_clients=client_manager.num_available(),
            selected_clients=len(configured),
        )
        return configured

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate one complete, validated fit round or hard-fail.

        Parameters
        ----------
        server_round : int
            One-based Flower server round.
        results : list[Any]
            Successful Flower fit results.
        failures : list[Any]
            Flower fit failures for the round.

        Returns
        -------
        tuple[Parameters, dict[str, Any]]
            Validated aggregate parameters and aggregated fit metrics.

        Raises
        ------
        RuntimeError
            If Flower reports any fit failure.
        ValueError
            If the result set, an input, or the aggregate violates the runtime
            aggregation contract.
        """
        round_started_at = getattr(self, "_round_started_at_ns", {}).pop(
            server_round, perf_counter_ns()
        )
        try:
            parameters: Parameters | None
            if failures:
                raise RuntimeError(
                    f"fit round {server_round} failed for {len(failures)} client(s)"
                )

            ordered_results, client_weights = _validate_fit_results(
                results, self.expected_client_ids, self.expected_weight_shapes
            )
            if self.use_huber:
                # Robust Huber aggregation instead of plain FedProx averaging
                reference = client_weights[0]
                vectors = [_flatten(weights) for weights in client_weights]
                counts = [result.num_examples for _, result in ordered_results]
                aggregated = huber_aggregate(vectors, counts, self.huber_threshold)
                parameters = ndarrays_to_parameters(_unflatten(aggregated, reference))
                metrics = {}
                if self.fit_metrics_aggregation_fn:
                    metrics = self.fit_metrics_aggregation_fn(
                        [
                            (result.num_examples, result.metrics)
                            for _, result in ordered_results
                        ]
                    )
            else:
                # Standard FedProx averaging
                parameters, metrics = super().aggregate_fit(
                    server_round, ordered_results, failures
                )

            if parameters is None:
                raise RuntimeError(f"fit round {server_round} produced no aggregate")
            _validate_aggregated_parameters(parameters, self.expected_weight_shapes)

            # Artifact saving
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            model = build_model_from_manifest(self.app_manifest)
            aggregate_weights = parameters_to_ndarrays(parameters)
            model.set_weights(aggregate_weights)
            model.save(str(self.model_path))
            _write_checkpoint(self.artifact_dir, server_round, aggregate_weights)
            telemetry = {
                name: sum(
                    int(result.metrics.get(name, 0)) for _, result in ordered_results
                )
                for name in (
                    "training_time_ns",
                    "request_parameter_bytes",
                    "response_parameter_bytes",
                )
            }
            log_event(
                self.logger,
                logging.INFO,
                "fit round completed",
                "fit_round_completed",
                round=server_round,
                clients=len(ordered_results),
                failures=len(failures),
                aggregation="huber" if self.use_huber else "fedprox",
                round_duration_ns=perf_counter_ns() - round_started_at,
                communication_scope=(
                    "serialized Flower Parameters protobufs; excludes message "
                    "metadata, TLS, and transport framing"
                ),
                training_time_ns=telemetry["training_time_ns"],
                request_parameter_bytes=telemetry["request_parameter_bytes"],
                response_parameter_bytes=telemetry["response_parameter_bytes"],
            )
        except BaseException:
            log_event(
                self.logger,
                logging.ERROR,
                "fit round failed",
                "fit_round_failed",
                exc_info=True,
                round=server_round,
                successful_clients=len(results),
                failures=len(failures),
                round_duration_ns=perf_counter_ns() - round_started_at,
            )
            self._release_artifact_lock()
            raise

        return parameters, metrics

    def aggregate_evaluate(
        self, server_round: int, results: list[Any], failures: list[Any]
    ) -> tuple[float | None, dict[str, Any]]:
        """Aggregate one complete evaluation round or hard-fail safely.

        Parameters
        ----------
        server_round : int
            One-based Flower server round.
        results : list[Any]
            Successful Flower evaluation results.
        failures : list[Any]
            Flower evaluation failures for the round.

        Returns
        -------
        tuple[float, dict[str, Any]]
            Finite aggregate loss and metrics.

        Raises
        ------
        RuntimeError
            If Flower reports any evaluation failure.
        ValueError
            If the complete result set or aggregate metrics are invalid.
        """
        try:
            if failures:
                raise RuntimeError(
                    f"evaluation round {server_round} failed for "
                    f"{len(failures)} client(s)"
                )
            ordered_results = _validate_evaluate_results(
                results, self.expected_client_ids
            )
            loss, metrics = super().aggregate_evaluate(
                server_round, ordered_results, []
            )
            _validate_aggregated_evaluation(loss, metrics)

            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self._write_client_metrics(server_round, ordered_results)
            if server_round == 1 and self.metrics_path.exists():
                self.metrics_path.unlink()
            file_exists = self.metrics_path.exists()

            with self.metrics_path.open("a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["round", "loss", "accuracy"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        "round": server_round,
                        "loss": loss,
                        "accuracy": metrics["accuracy"],
                    }
                )
            if server_round == self.final_round:
                publish_completed_run(self.artifact_root, self.artifact_dir)
                prune_run_history(
                    self.artifact_root,
                    self.artifact_retention_runs,
                    active_run_dir=self.artifact_dir,
                )
        except BaseException:
            log_event(
                self.logger,
                logging.ERROR,
                "evaluation round failed",
                "evaluation_round_failed",
                exc_info=True,
                round=server_round,
            )
            self._release_artifact_lock()
            raise
        log_event(
            self.logger,
            logging.INFO,
            "evaluation round completed",
            "evaluation_round_completed",
            round=server_round,
            clients=len(ordered_results),
            loss=loss,
            accuracy=float(metrics["accuracy"]),
        )
        if server_round == self.final_round:
            self._release_artifact_lock()
        return loss, metrics


def create_strategy(
    min_clients: int = DEFAULT_EXPECTED_CLIENTS,
    artifact_dir: str | Path | None = None,
    artifact_lock: RunArtifactLock | None = None,
    artifact_root: str | Path | None = None,
    artifact_retention_runs: int = DEFAULT_ARTIFACT_RETENTION_RUNS,
    final_round: int | None = None,
    public_artifact_dir: str | Path | None = None,
    proximal_mu: float = 0.1,
    use_huber: bool = False,
    huber_threshold: float = DEFAULT_HUBER_THRESHOLD,
    resume_from_checkpoint: str | Path | None = None,
    seed: int = 67,
) -> SentimentServer:
    """Create the deployment strategy with exact-round validation.

    Parameters
    ----------
    min_clients : int, optional
        Exact number of clients required in every fit round.
    artifact_dir : str or pathlib.Path, optional
        Directory for this run's server artifacts.
    artifact_lock : RunArtifactLock, optional
        Exclusive writer lock held for this run.
    artifact_root : str or pathlib.Path, optional
        Root directory containing run history.
    artifact_retention_runs : int, optional
        Number of completed runs to retain.
    final_round : int, optional
        One-based final Flower round.
    public_artifact_dir : str or pathlib.Path, optional
        Directory containing the public application manifest.
    proximal_mu : float, optional
        FedProx proximal coefficient.
    use_huber : bool, optional
        Whether to replace sample-weighted averaging with Huber aggregation.
    huber_threshold : float, optional
        Positive Huber residual threshold.
    resume_from_checkpoint : str or pathlib.Path or None, optional
        Explicit checkpoint from an earlier run. Resume starts a new immutable run,
        uses these weights as round-zero parameters, and executes ``final_round`` new
        rounds; metrics and round numbers do not continue from the source run.
    seed : int, optional
        Run-level Keras initialization seed.

    Returns
    -------
    SentimentServer
        Configured deployment strategy.

    Raises
    ------
    ValueError
        If ``min_clients`` is not a positive built-in integer.
    """
    if type(min_clients) is not int or min_clients <= 0:
        raise ValueError("min_clients must be a positive built-in integer")
    if type(seed) is not int:
        raise ValueError("seed must be a built-in integer")
    resolved_artifact_dir = resolve_dir(artifact_dir or default_server_artifact_dir())

    app_manifest = load_app_manifest(
        public_artifact_dir=public_artifact_dir,
    )
    keras.utils.set_random_seed(seed)
    initial_model = build_model_from_manifest(app_manifest)
    initial_weights = initial_model.get_weights()
    expected_weight_shapes = tuple(weight.shape for weight in initial_weights)
    if resume_from_checkpoint is not None:
        initial_weights = _load_checkpoint(
            resume_from_checkpoint, app_manifest, expected_weight_shapes
        )

    strategy = SentimentServer(
        proximal_mu=proximal_mu,
        accept_failures=False,
        artifact_dir=resolved_artifact_dir,
        artifact_lock=artifact_lock,
        artifact_root=artifact_root,
        artifact_retention_runs=artifact_retention_runs,
        final_round=final_round,
        app_manifest=app_manifest,
        huber_threshold=huber_threshold,
        use_huber=use_huber,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=ndarrays_to_parameters(initial_weights),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    strategy.expected_client_ids = frozenset(range(min_clients))
    strategy.expected_weight_shapes = expected_weight_shapes
    strategy.experiment_seed = seed
    return strategy


def server_fn(context: Context) -> ServerAppComponents:
    """Create Flower server components from one deployment run context.

    Parameters
    ----------
    context : Context
        Flower run context containing deployment configuration.

    Returns
    -------
    ServerAppComponents
        Strategy and round configuration for the Flower server application.

    Raises
    ------
    ValueError
        If numeric settings or artifact directory boundaries are invalid.

    Notes
    -----
    ``resume-from-checkpoint`` names an explicit checkpoint file. Resuming creates a
    separate run, resets metrics and Flower round numbering, and performs the
    configured number of new rounds. The source run and completed publications are
    never modified.
    """
    run_config: dict[str, Any] = context.run_config
    artifact_root = resolve_dir(
        run_config.get("server-artifact-dir", default_server_artifact_dir())
    )
    num_rounds = int(run_config.get("num-server-rounds", DEFAULT_SERVER_ROUNDS))
    retention_runs = int(
        run_config.get("artifact-retention-runs", DEFAULT_ARTIFACT_RETENTION_RUNS)
    )
    expected_clients = run_config.get("expected-client-count", DEFAULT_EXPECTED_CLIENTS)
    if num_rounds < 1:
        raise ValueError("num-server-rounds must be a positive integer")
    if retention_runs < 1:
        raise ValueError("artifact-retention-runs must be a positive integer")
    if type(expected_clients) is not int or expected_clients < 1:
        raise ValueError("expected-client-count must be a positive built-in integer")
    public_artifact_dir = resolve_public_artifact_dir(run_config)
    resolved_public_dir = public_artifact_dir.resolve()
    resolved_artifact_root = artifact_root.resolve()
    if (
        resolved_artifact_root == resolved_public_dir
        or resolved_artifact_root.is_relative_to(resolved_public_dir)
        or resolved_public_dir.is_relative_to(resolved_artifact_root)
    ):
        raise ValueError("server and public artifact directories must not overlap")
    artifact_lock = acquire_run_artifact_lock(artifact_root)
    try:
        run_dir = create_run_artifact_dir(
            artifact_root,
            run_config,
            public_artifact_dir=public_artifact_dir,
            flower_run_id=getattr(context, "run_id", None),
        )
        strategy = create_strategy(
            min_clients=expected_clients,
            artifact_dir=run_dir,
            artifact_lock=artifact_lock,
            artifact_root=artifact_root,
            artifact_retention_runs=retention_runs,
            final_round=num_rounds,
            public_artifact_dir=run_config.get("public-artifact-dir"),
            proximal_mu=float(run_config.get("proximal-mu", 0.1)),
            use_huber=parse_run_config_bool(run_config.get("use-huber"), default=False),
            huber_threshold=float(
                run_config.get("huber-threshold", DEFAULT_HUBER_THRESHOLD)
            ),
            resume_from_checkpoint=run_config.get("resume-from-checkpoint"),
            seed=int(run_config.get("experiment-seed", 67)),
        )
        prune_run_history(artifact_root, retention_runs, active_run_dir=run_dir)
        strategy.logger = structured_logger("server")
        log_event(
            strategy.logger,
            logging.INFO,
            "server ready",
            "server_ready",
            run_id=run_dir.name,
            rounds=num_rounds,
            expected_clients=expected_clients,
            artifact_directory=str(run_dir),
        )
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=num_rounds),
        )
    except BaseException:
        artifact_lock.release()
        raise


app = ServerApp(server_fn=server_fn)
