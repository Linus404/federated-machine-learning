from __future__ import annotations

from pathlib import Path
from typing import Any

from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context

from src.local_training import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_LOCAL_EPOCHS,
    ArrayPair,
    build_model,
    load_partition,
    vocab_size,
)
from src.paths import data_dir_path, default_data_dir


class SentimentClient(NumPyClient):
    """Flower client that trains on one saved partition."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        partition: int = 0,
        epochs: int = DEFAULT_LOCAL_EPOCHS,
        batch_size: int = 64,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        validation_split: float = 0.2,
    ) -> None:
        self.data_dir = data_dir_path(data_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_data: ArrayPair
        self.val_data: ArrayPair
        self.train_data, self.val_data = load_partition(
            self.data_dir, partition, validation_split
        )
        self.model = build_model(
            vocab_size(self.data_dir), self.train_data[0].shape[1], embedding_dim
        )

    def get_parameters(self, config: dict[str, Any]) -> list[Any]:
        """Retrieve the current local model weights as a list of NumPy arrays."""
        return self.model.get_weights()

    def fit(
        self, parameters: list[Any], config: dict[str, Any]
    ) -> tuple[list[Any], int, dict[str, float]]:
        """Train locally once the server sends the latest global weights."""
        self.model.set_weights(parameters)
        history = self.model.fit(
            *self.train_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        metrics = {name: float(values[-1]) for name, values in history.history.items()}
        return self.model.get_weights(), len(self.train_data[0]), metrics

    def evaluate(
        self, parameters: list[Any], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, float]]:
        """Evaluate the global model on this client's validation split."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(*self.val_data, verbose=0)
        return float(loss), len(self.val_data[0]), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Any:
    """Create one client; Flower supplies partition-id for each SuperNode."""
    run_config: dict[str, Any] = context.run_config
    partition = int(context.node_config.get("partition-id", 0))

    return SentimentClient(
        data_dir=run_config.get("data-dir", default_data_dir()),
        partition=partition,
        epochs=int(run_config.get("local-epochs", DEFAULT_LOCAL_EPOCHS)),
        batch_size=int(run_config.get("batch-size", 64)),
        embedding_dim=int(run_config.get("embedding-dim", DEFAULT_EMBEDDING_DIM)),
        validation_split=float(run_config.get("validation-split", 0.2)),
    ).to_client()


app = ClientApp(client_fn=client_fn)
