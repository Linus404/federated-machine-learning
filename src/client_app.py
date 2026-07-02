from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context

from src import parse_run_config_bool
from src.local_training import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_LOCAL_EPOCHS,
    ArrayPair,
    build_model,
    load_partition,
    vocab_size,
)
from src.paths import data_dir_path, default_data_dir

UPDATE_NOISE_L2_NORM_CLIP = 1.0
UPDATE_NOISE_MULTIPLIER = 0.001


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
        use_update_noise: bool = False,
        update_noise_l2_norm_clip: float = UPDATE_NOISE_L2_NORM_CLIP,
        update_noise_multiplier: float = UPDATE_NOISE_MULTIPLIER,
    ) -> None:
        self.data_dir = data_dir_path(data_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_update_noise = use_update_noise
        self.update_noise_l2_norm_clip = update_noise_l2_norm_clip
        self.update_noise_multiplier = update_noise_multiplier
        self.train_data: ArrayPair
        self.val_data: ArrayPair
        self.train_data, self.val_data = load_partition(
            self.data_dir, partition, validation_split
        )
        self.model = build_model(
            vocab_size(self.data_dir), self.train_data[0].shape[1], embedding_dim
        )

    def _add_update_noise(self, weights_before: list, weights_after: list) -> list:
        """Clip and noise the weight update for an illustrative ablation."""
        noisy_weights = []

        for w_before, w_after in zip(weights_before, weights_after):
            update = w_after - w_before
            norm = np.linalg.norm(update)
            update = update / max(1.0, norm / self.update_noise_l2_norm_clip)
            noise = np.random.normal(
                0,
                self.update_noise_multiplier * self.update_noise_l2_norm_clip,
                update.shape,
            )
            noisy_weights.append(w_before + update + noise)

        return noisy_weights

    def get_parameters(self, config: dict[str, Any]) -> list[Any]:
        """Retrieve the current local model weights as a list of NumPy arrays."""
        return self.model.get_weights()

    def fit(
        self, parameters: list[Any], config: dict[str, Any]
    ) -> tuple[list[Any], int, dict[str, float]]:
        """Train locally and optionally apply illustrative update noise."""
        self.model.set_weights(parameters)
        weights_before = [w.copy() for w in self.model.get_weights()]
        history = self.model.fit(
            *self.train_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        trained_weights = self.model.get_weights()
        weights = (
            self._add_update_noise(weights_before, trained_weights)
            if self.use_update_noise
            else trained_weights
        )
        metrics = {name: float(values[-1]) for name, values in history.history.items()}

        return weights, len(self.train_data[0]), metrics

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
        use_update_noise=parse_run_config_bool(
            run_config.get("use-update-noise"), default=False
        ),
        update_noise_l2_norm_clip=float(
            run_config.get("update-noise-l2-norm-clip", UPDATE_NOISE_L2_NORM_CLIP)
        ),
        update_noise_multiplier=float(
            run_config.get("update-noise-multiplier", UPDATE_NOISE_MULTIPLIER)
        ),
    ).to_client()


app = ClientApp(client_fn=client_fn)
