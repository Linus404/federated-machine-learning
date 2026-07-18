from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from flwr.serverapp import ServerApp

from src import parse_run_config_bool
from src.app_manifest import load_app_manifest, resolve_public_artifact_dir
from src.huber_strategy import (
    DEFAULT_HUBER_THRESHOLD,
    _flatten,
    _unflatten,
    huber_aggregate,
)
from src.local_training import DEFAULT_EMBEDDING_DIM, build_model_from_manifest
from src.paths import (
    RunArtifactLock,
    acquire_run_artifact_lock,
    client_metrics_path,
    clear_artifact_dir,
    default_server_artifact_dir,
    global_model_path,
    metrics_path,
    resolve_dir,
)

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"keras\..*")

DEFAULT_SERVER_ROUNDS = 20


def weighted_average(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """Aggregate client accuracies by validation sample count."""
    total = sum(num_examples for num_examples, _ in metrics)
    accuracy = (
        sum(num_examples * metric["accuracy"] for num_examples, metric in metrics)
        / total
    )

    return {"accuracy": accuracy}


class SentimentServer(FedProx):
    """FedProx (non-IID robust) + Huber aggregation (Byzantine robust) + artifact saving."""

    def __init__(
        self,
        app_manifest,
        artifact_dir=None,
        artifact_lock: RunArtifactLock | None = None,
        huber_threshold: float = DEFAULT_HUBER_THRESHOLD,
        use_huber: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.artifact_dir = resolve_dir(artifact_dir or default_server_artifact_dir())
        self._artifact_lock = artifact_lock
        self.app_manifest = app_manifest
        self.huber_threshold = huber_threshold
        self.use_huber = use_huber

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

    def aggregate_fit(self, server_round, results, failures):
        if self.use_huber and results:
            # Robust Huber aggregation instead of plain FedProx averaging
            reference = parameters_to_ndarrays(results[0][1].parameters)
            vectors = [
                _flatten(parameters_to_ndarrays(r.parameters)) for _, r in results
            ]
            counts = [r.num_examples for _, r in results]
            aggregated = huber_aggregate(vectors, counts, self.huber_threshold)
            parameters = ndarrays_to_parameters(_unflatten(aggregated, reference))
            metrics = {}
            if self.fit_metrics_aggregation_fn:
                metrics = self.fit_metrics_aggregation_fn(
                    [(r.num_examples, r.metrics) for _, r in results]
                )
        else:
            # Standard FedProx averaging
            parameters, metrics = super().aggregate_fit(server_round, results, failures)

        # Artifact saving
        if parameters is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            model = build_model_from_manifest(self.app_manifest)
            model.set_weights(parameters_to_ndarrays(parameters))
            model.save(str(self.model_path))

        return parameters, metrics

    def aggregate_evaluate(
        self, server_round: int, results: list[Any], failures: list[Any]
    ) -> tuple[float | None, dict[str, Any]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._write_client_metrics(server_round, results)
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
                    "accuracy": metrics.get("accuracy"),
                }
            )
        return loss, metrics


def create_strategy(
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    min_clients: int = 4,
    artifact_dir: str | Path | None = None,
    artifact_lock: RunArtifactLock | None = None,
    public_manifest_path: str | Path | None = None,
    public_artifact_dir: str | Path | None = None,
    proximal_mu: float = 0.1,
    use_huber: bool = False,
    huber_threshold: float = DEFAULT_HUBER_THRESHOLD,
) -> SentimentServer:
    resolved_artifact_dir = resolve_dir(artifact_dir or default_server_artifact_dir())

    app_manifest = load_app_manifest(
        public_manifest_path=public_manifest_path,
        public_artifact_dir=public_artifact_dir,
        config_embedding_dim=embedding_dim,
    )
    initial_model = build_model_from_manifest(app_manifest)

    return SentimentServer(
        proximal_mu=proximal_mu,
        artifact_dir=resolved_artifact_dir,
        artifact_lock=artifact_lock,
        app_manifest=app_manifest,
        huber_threshold=huber_threshold,
        use_huber=use_huber,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=ndarrays_to_parameters(initial_model.get_weights()),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )


def server_fn(context: Context) -> ServerAppComponents:
    run_config: dict[str, Any] = context.run_config
    artifact_dir = run_config.get("server-artifact-dir", default_server_artifact_dir())
    num_rounds = int(run_config.get("num-server-rounds", DEFAULT_SERVER_ROUNDS))
    public_artifact_dir = resolve_public_artifact_dir(run_config)
    protected_paths = [public_artifact_dir]
    artifact_lock = acquire_run_artifact_lock(artifact_dir)
    try:
        resolved_artifact_dir = clear_artifact_dir(
            artifact_dir, protected_paths=protected_paths
        )
        strategy = create_strategy(
            embedding_dim=int(run_config["embedding-dim"]),
            min_clients=4,
            artifact_dir=resolved_artifact_dir,
            artifact_lock=artifact_lock,
            public_manifest_path=run_config.get("public-manifest-path"),
            public_artifact_dir=run_config.get("public-artifact-dir"),
            proximal_mu=float(run_config.get("proximal-mu", 0.1)),
            use_huber=parse_run_config_bool(run_config.get("use-huber"), default=False),
            huber_threshold=float(
                run_config.get("huber-threshold", DEFAULT_HUBER_THRESHOLD)
            ),
        )
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=num_rounds),
        )
    except BaseException:
        artifact_lock.release()
        raise


app = ServerApp(server_fn=server_fn)
