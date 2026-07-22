from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import keras
import numpy as np
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import NDArrays, Scalar

from src import parse_run_config_bool
from src.app_manifest import configured_value, load_app_manifest
from src.local_training import (
    DEFAULT_LOCAL_EPOCHS,
    ArrayPair,
    build_model_from_manifest,
    load_client_shard,
)
from src.paths import resolve_dir

UPDATE_NOISE_L2_NORM_CLIP = 1.0
UPDATE_NOISE_MULTIPLIER = 0.001
CLIENT_DATA_DIR_ENV = "CLIENT_DATA_DIR"
PROXIMAL_MU_CONFIG_KEY = "proximal_mu"


def _proximal_penalty(local_weights: list[Any], global_weights: list[Any]) -> Any:
    """Return the squared L2 distance from the round's global weights."""
    terms = [
        keras.ops.sum(keras.ops.square(local_weight - global_weight))
        for local_weight, global_weight in zip(local_weights, global_weights)
    ]
    if not terms:
        return keras.ops.zeros(())
    return sum(terms[1:], terms[0])


class SentimentClient(NumPyClient):
    """Flower client that tokenizes and trains on its client-scoped shard."""

    def __init__(
        self,
        client_data_dir: str | Path,
        client_id: int = 0,
        epochs: int = DEFAULT_LOCAL_EPOCHS,
        batch_size: int = 64,
        validation_split: float = 0.2,
        public_artifact_dir: str | Path | None = None,
        use_update_noise: bool = False,
        update_noise_l2_norm_clip: float = UPDATE_NOISE_L2_NORM_CLIP,
        update_noise_multiplier: float = UPDATE_NOISE_MULTIPLIER,
    ) -> None:
        self.client_data_dir = resolve_dir(client_data_dir)
        self.client_id = client_id
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_update_noise = use_update_noise
        self.update_noise_l2_norm_clip = update_noise_l2_norm_clip
        self.update_noise_multiplier = update_noise_multiplier
        self.train_data: ArrayPair
        self.val_data: ArrayPair
        manifest = load_app_manifest(
            public_artifact_dir=public_artifact_dir,
        )
        self.train_data, self.val_data = load_client_shard(
            self.client_data_dir, manifest, validation_split
        )
        self.model = build_model_from_manifest(manifest)

    def _add_update_noise(
        self,
        weights_before: list[np.ndarray],
        weights_after: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Apply illustrative float32 update noise, not differential privacy.

        Parameters
        ----------
        weights_before : list[numpy.ndarray]
            Complete round-start model weights.
        weights_after : list[numpy.ndarray]
            Complete post-training model weights in the same order.

        Returns
        -------
        list[numpy.ndarray]
            Noisy model weights with exact ``float32`` dtype.

        Raises
        ------
        ValueError
            If the configuration or any input tensor is incompatible with the
            server's strict model-weight contract.
        """
        if len(weights_before) != len(weights_after) or not weights_before:
            raise ValueError("update-noise weight lists must be nonempty and equal")
        if (
            not np.isfinite(self.update_noise_l2_norm_clip)
            or self.update_noise_l2_norm_clip <= 0
        ):
            raise ValueError("update-noise L2 norm clip must be positive and finite")
        if (
            not np.isfinite(self.update_noise_multiplier)
            or self.update_noise_multiplier < 0
        ):
            raise ValueError("update-noise multiplier must be non-negative and finite")

        noisy_weights: list[np.ndarray] = []
        noise_scale = np.float32(
            self.update_noise_multiplier * self.update_noise_l2_norm_clip
        )
        if not np.isfinite(noise_scale):
            raise ValueError("update-noise scale must be finite in float32")

        for weight_before, weight_after in zip(
            weights_before, weights_after, strict=True
        ):
            if (
                not isinstance(weight_before, np.ndarray)
                or not isinstance(weight_after, np.ndarray)
                or weight_before.dtype != np.dtype(np.float32)
                or weight_after.dtype != np.dtype(np.float32)
            ):
                raise ValueError("update-noise weights must use exactly float32")
            if (
                weight_before.shape != weight_after.shape
                or weight_before.ndim == 0
                or weight_before.size == 0
            ):
                raise ValueError(
                    "update-noise weights must have matching non-scalar shapes"
                )
            if not np.all(np.isfinite(weight_before)) or not np.all(
                np.isfinite(weight_after)
            ):
                raise ValueError("update-noise weights must contain finite values")

            update = np.subtract(weight_after, weight_before)
            if not np.all(np.isfinite(update)):
                raise ValueError("update-noise update must contain finite values")
            norm = float(np.linalg.norm(update.astype(np.float64)))
            clipped_update = (
                np.multiply(
                    update,
                    np.float32(self.update_noise_l2_norm_clip / norm),
                )
                if norm > self.update_noise_l2_norm_clip
                else update
            )
            noise = np.random.standard_normal(update.shape).astype(np.float32)
            noise *= noise_scale
            noisy_weight = np.add(np.add(weight_before, clipped_update), noise)
            if not np.all(np.isfinite(noisy_weight)):
                raise ValueError("update-noise output must contain finite values")
            noisy_weights.append(noisy_weight)

        return noisy_weights

    def _configure_proximal_loss(self, proximal_mu: float) -> None:
        """Compile the client model with the FedProx objective for this round."""
        if proximal_mu < 0:
            raise ValueError("proximal_mu must be non-negative")

        global_weights = [
            keras.ops.convert_to_tensor(weight.numpy().copy())
            for weight in self.model.trainable_weights
        ]
        base_loss = keras.losses.BinaryCrossentropy()

        def fedprox_loss(y_true: Any, y_pred: Any) -> Any:
            proximal_term = _proximal_penalty(
                self.model.trainable_weights, global_weights
            )
            return base_loss(y_true, y_pred) + (proximal_mu / 2.0) * proximal_term

        self.model.compile(
            optimizer=self.model.optimizer,
            loss=fedprox_loss,
            metrics=["accuracy"],
        )

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Retrieve the current local model weights as a list of NumPy arrays."""
        return self.model.get_weights()

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train with illustrative update-noise, not formal differential privacy."""
        self.model.set_weights(parameters)
        if PROXIMAL_MU_CONFIG_KEY in config:
            self._configure_proximal_loss(float(config[PROXIMAL_MU_CONFIG_KEY]))
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
        metrics: dict[str, Scalar] = {
            name: float(values[-1]) for name, values in history.history.items()
        }
        metrics["client_id"] = self.client_id

        return weights, len(self.train_data[0]), metrics

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the global model on this client's validation split."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(*self.val_data, verbose=0)

        return (
            float(loss),
            len(self.val_data[0]),
            {
                "accuracy": float(accuracy),
                "client_id": self.client_id,
            },
        )


def client_fn(context: Any) -> Any:
    """Create one client; Flower supplies partition-id for each SuperNode."""
    run_config: dict[str, Any] = context.run_config
    partition = int(context.node_config.get("partition-id", 0))
    client_data_dir = _configured_client_data_dir(run_config, partition)

    return SentimentClient(
        client_data_dir=client_data_dir,
        client_id=partition,
        epochs=int(run_config.get("local-epochs", DEFAULT_LOCAL_EPOCHS)),
        batch_size=int(run_config.get("batch-size", 64)),
        validation_split=float(run_config.get("validation-split", 0.2)),
        public_artifact_dir=configured_value(run_config, "public-artifact-dir"),
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


def _configured_client_data_dir(run_config: dict[str, Any], partition: int) -> Path:
    value = os.environ.get(CLIENT_DATA_DIR_ENV) or configured_value(
        run_config, "client-data-dir"
    )
    if not value:
        raise ValueError(
            "client-data-dir or CLIENT_DATA_DIR must point to this client's raw shard"
        )
    return resolve_dir(str(value).format(partition=partition))


app = ClientApp(client_fn=client_fn)
