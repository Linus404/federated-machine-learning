from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
from flwr.serverapp import ServerApp
from src.huber_strategy import huber_aggregate, _flatten, _unflatten
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from src.local_training import (
    DEFAULT_EMBEDDING_DIM,
    build_model,
    sequence_length,
    vocab_size,
)
from src.paths import (
    artifact_dir_path,
    data_dir_path,
    default_artifact_dir,
    default_data_dir,
    global_model_path,
    metrics_path,
)

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"keras\..*")

DEFAULT_SERVER_ROUNDS = 5


def weighted_average(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """Aggregate client accuracies by validation sample count."""
    total = sum(num_examples for num_examples, _ in metrics)
    accuracy = (
        sum(num_examples * metric["accuracy"] for num_examples, metric in metrics)
        / total
    )

    return {"accuracy": accuracy}


class ArtifactSavingFedProx(FedProx):
    """FedProx (non-IID robust) + Huber aggregation (Byzantine robust) + artifact saving."""

    def __init__(
        self,
        data_dir=None,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        artifact_dir=None,
        huber_threshold: float = 10.0,
        use_huber: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir_path(data_dir)
        self.embedding_dim = embedding_dim
        self.artifact_dir = artifact_dir_path(artifact_dir)
        self.huber_threshold = huber_threshold
        self.use_huber = use_huber

    @property
    def model_path(self) -> Path:
        return global_model_path(self.artifact_dir)

    @property
    def metrics_path(self) -> Path:
        return metrics_path(self.artifact_dir)

    def aggregate_fit(self, server_round, results, failures):
        if self.use_huber and results:
            # Robust Huber aggregation instead of plain FedProx averaging
            reference = parameters_to_ndarrays(results[0][1].parameters)
            vectors = [_flatten(parameters_to_ndarrays(r.parameters)) for _, r in results]
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

        # Artifact saving (unchanged)
        if parameters is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            model = build_model(
                vocab_size(self.data_dir),
                sequence_length(self.data_dir),
                self.embedding_dim,
            )
            model.set_weights(parameters_to_ndarrays(parameters))
            model.save(str(self.model_path))

        return parameters, metrics

    def aggregate_evaluate(
        self, server_round: int, results: list[Any], failures: list[Any]
    ) -> tuple[float | None, dict[str, Any]]:
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
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
    data_dir: str | Path | None = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    min_clients: int = 4,
    artifact_dir: str | Path | None = None,
    proximal_mu: float = 0.1,
) -> ArtifactSavingFedAvg:
    resolved_data_dir = data_dir_path(data_dir)
    resolved_artifact_dir = artifact_dir_path(artifact_dir)

    initial_model = build_model(
        vocab_size(resolved_data_dir),
        sequence_length(resolved_data_dir),
        embedding_dim,
    )

    return ArtifactSavingFedProx(
        proximal_mu=proximal_mu,
        data_dir=resolved_data_dir,
        embedding_dim=embedding_dim,
        artifact_dir=resolved_artifact_dir,
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
    """Configure FedAvg for the four simulated sentiment clients."""
    run_config: dict[str, Any] = context.run_config
    data_dir = run_config.get("data-dir", default_data_dir())
    artifact_dir = run_config.get("artifact-dir", default_artifact_dir())
    num_rounds = int(run_config.get("num-server-rounds", DEFAULT_SERVER_ROUNDS))

    strategy = create_strategy(
        data_dir=data_dir,
        embedding_dim=int(run_config.get("embedding-dim", DEFAULT_EMBEDDING_DIM)),
        min_clients=4,
        artifact_dir=artifact_dir,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
