import unittest
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from flwr.common import ndarrays_to_parameters

import src
import src.client_app as client_app
import src.server_app as server_app


class RunConfigBoolTests(unittest.TestCase):
    def test_parse_run_config_bool_accepts_native_and_string_values(self) -> None:
        true_values = [True, "true", " TRUE ", "1", "yes", "on"]
        false_values = [False, "false", " FALSE ", "0", "no", "off"]

        for value in true_values:
            with self.subTest(value=value):
                self.assertTrue(src.parse_run_config_bool(value))

        for value in false_values:
            with self.subTest(value=value):
                self.assertFalse(src.parse_run_config_bool(value, default=True))

    def test_parse_run_config_bool_uses_default_only_for_missing_value(self) -> None:
        self.assertTrue(src.parse_run_config_bool(None, default=True))
        self.assertFalse(src.parse_run_config_bool(None, default=False))

    def test_parse_run_config_bool_rejects_ambiguous_values(self) -> None:
        with self.assertRaises(ValueError):
            src.parse_run_config_bool("sometimes")


class ClientConfigTests(unittest.TestCase):
    def test_client_fn_passes_update_noise_config_to_constructor(self) -> None:
        fake_client = Mock()
        fake_client.to_client.return_value = "flower-client"
        context = SimpleNamespace(
            run_config={
                "client-data-dir": "data/client-{partition}",
                "local-epochs": 2,
                "batch-size": 8,
                "validation-split": 0.25,
                "use-update-noise": "true",
                "update-noise-l2-norm-clip": 2.0,
                "update-noise-multiplier": 0.5,
            },
            node_config={"partition-id": "3"},
        )

        with patch.object(
            client_app, "SentimentClient", return_value=fake_client
        ) as sentiment_client:
            result = client_app.client_fn(context)

        self.assertEqual(result, "flower-client")
        sentiment_client.assert_called_once_with(
            client_id=3,
            epochs=2,
            batch_size=8,
            validation_split=0.25,
            public_artifact_dir=None,
            use_update_noise=True,
            update_noise_l2_norm_clip=2.0,
            update_noise_multiplier=0.5,
            master_seed=67,
            client_data_dir=client_app.resolve_dir("data/client-3"),
        )

    def test_client_fn_requires_private_raw_data(self) -> None:
        context = SimpleNamespace(
            run_config={},
            node_config={"partition-id": "0"},
        )

        with self.assertRaisesRegex(ValueError, "client-data-dir"):
            client_app.client_fn(context)

    def test_deployment_mount_overrides_local_partition_template(self) -> None:
        with patch.dict(os.environ, {"CLIENT_DATA_DIR": "/app/client-data"}):
            resolved = client_app._configured_client_data_dir(
                {"client-data-dir": "artifacts/clients/client-{partition}"}, 2
            )

        self.assertEqual(resolved, client_app.resolve_dir("/app/client-data"))

    def test_deployed_client_uses_manifest_for_tokenization_and_model(self) -> None:
        manifest = object()
        train_data = (
            np.array([[2, 0]], dtype="int32"),
            np.array([1], dtype="float32"),
        )
        val_data = (
            np.array([[3, 0]], dtype="int32"),
            np.array([0], dtype="float32"),
        )
        model = object()

        with (
            patch.object(client_app, "load_app_manifest", return_value=manifest),
            patch.object(
                client_app, "load_client_shard", return_value=(train_data, val_data)
            ) as load_shard,
            patch.object(
                client_app, "build_model_from_manifest", return_value=model
            ) as build_from_manifest,
        ):
            client = client_app.SentimentClient(
                client_data_dir="client-0",
                public_artifact_dir="public",
            )

        load_shard.assert_called_once_with(
            client_app.resolve_dir("client-0"), manifest, 0, 0.2
        )
        build_from_manifest.assert_called_once_with(
            manifest,
            master_seed=67,
            seed_namespace=("client", 0),
        )
        self.assertIs(client.model, model)
        self.assertIs(client.train_data, train_data)


class FakeHistory:
    history = {"loss": [0.2], "accuracy": [0.9]}


class FakeModel:
    def __init__(self) -> None:
        self.weights = [np.array([0.0, 0.0], dtype="float32")]
        self.layers = []

    def set_weights(self, parameters):
        self.weights = [weight.copy() for weight in parameters]

    def get_weights(self):
        return [weight.copy() for weight in self.weights]

    def fit(self, *args, **kwargs):
        self.weights = [np.array([1.0, 2.0], dtype="float32")]
        return FakeHistory()


class UpdateNoiseFitTests(unittest.TestCase):
    def make_client(self, use_update_noise: bool):
        client = client_app.SentimentClient.__new__(client_app.SentimentClient)
        client.model = FakeModel()
        client.train_data = (
            np.array([[1], [2]], dtype="int32"),
            np.array([0.0, 1.0], dtype="float32"),
        )
        client.epochs = 1
        client.batch_size = 1
        client.client_id = 3
        client.use_update_noise = use_update_noise
        client.update_noise_l2_norm_clip = 1.0
        client.update_noise_multiplier = 0.001
        client.master_seed = 67
        return client

    def test_fit_returns_trained_weights_when_update_noise_is_disabled(self) -> None:
        client = self.make_client(use_update_noise=False)
        client._add_update_noise = Mock(return_value=[np.array([9.0], dtype="float32")])

        weights, num_examples, metrics = client.fit(
            [np.array([0.0, 0.0], dtype="float32")], {"server_round": 1}
        )

        client._add_update_noise.assert_not_called()
        np.testing.assert_array_equal(weights[0], np.array([1.0, 2.0], dtype="float32"))
        self.assertEqual(num_examples, 2)
        self.assertEqual(metrics, {"loss": 0.2, "accuracy": 0.9, "client_id": 3})

    def test_fit_applies_update_noise_when_enabled(self) -> None:
        client = self.make_client(use_update_noise=True)
        noisy_weights = [np.array([9.0], dtype="float32")]
        client._add_update_noise = Mock(return_value=noisy_weights)

        weights, _, _ = client.fit(
            [np.array([0.0, 0.0], dtype="float32")], {"server_round": 1}
        )

        client._add_update_noise.assert_called_once()
        before, after, server_round = client._add_update_noise.call_args.args
        np.testing.assert_array_equal(before[0], [0.0, 0.0])
        np.testing.assert_array_equal(after[0], [1.0, 2.0])
        self.assertEqual(server_round, 1)
        self.assertIs(weights, noisy_weights)

    def test_real_update_noise_remains_float32_through_server_validation(
        self,
    ) -> None:
        client = self.make_client(use_update_noise=True)
        client.client_id = 0
        np.random.seed(17)

        weights, num_examples, metrics = client.fit(
            [np.array([0.0, 0.0], dtype=np.float32)], {"server_round": 1}
        )
        result = (
            object(),
            SimpleNamespace(
                parameters=ndarrays_to_parameters(weights),
                num_examples=num_examples,
                metrics=metrics,
            ),
        )
        _, decoded = server_app._validate_fit_results([result], frozenset({0}), ((2,),))

        self.assertEqual(weights[0].dtype, np.dtype(np.float32))
        self.assertEqual(decoded[0][0].dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(decoded[0][0], weights[0])

    def test_update_noise_rejects_incompatible_weight_tensors(self) -> None:
        client = self.make_client(use_update_noise=True)
        valid = np.array([0.0, 1.0], dtype=np.float32)
        invalid_pairs = [
            ([valid.astype(np.float64)], [valid]),
            ([valid], [valid.astype(np.float64)]),
            ([valid], [np.array([[0.0, 1.0]], dtype=np.float32)]),
            ([valid], [np.array([0.0, np.nan], dtype=np.float32)]),
            ([], []),
        ]

        for before, after in invalid_pairs:
            with self.subTest(before=before, after=after):
                with self.assertRaises(ValueError):
                    client._add_update_noise(before, after, 1)

    def test_update_noise_is_repeatable_and_separated_without_global_rng(self) -> None:
        before = [np.array([0.0, 0.0], dtype=np.float32)]
        after = [np.array([1.0, 2.0], dtype=np.float32)]
        first = self.make_client(use_update_noise=True)
        second = self.make_client(use_update_noise=True)

        with patch.object(
            client_app.np.random,
            "standard_normal",
            side_effect=AssertionError("global NumPy RNG used"),
        ):
            repeated = first._add_update_noise(before, after, 4)
            same_stream = second._add_update_noise(before, after, 4)
            next_round = first._add_update_noise(before, after, 5)
            first.client_id += 1
            next_client = first._add_update_noise(before, after, 4)

        np.testing.assert_array_equal(repeated[0], same_stream[0])
        self.assertFalse(np.array_equal(repeated[0], next_round[0]))
        self.assertFalse(np.array_equal(repeated[0], next_client[0]))

    def test_fit_rejects_missing_or_invalid_round_seed_namespace(self) -> None:
        client = self.make_client(use_update_noise=False)
        for value in (None, True, 0, -1, "1"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "server_round"),
            ):
                config = {} if value is None else {"server_round": value}
                client.fit([np.array([0.0, 0.0], dtype=np.float32)], config)

    def test_fit_configures_client_side_fedprox_loss(self) -> None:
        client = self.make_client(use_update_noise=False)
        client._configure_proximal_loss = Mock()

        client.fit(
            [np.array([0.0, 0.0], dtype="float32")],
            {"proximal_mu": 0.25, "server_round": 1},
        )

        client._configure_proximal_loss.assert_called_once_with(0.25)


class ProximalPenaltyTests(unittest.TestCase):
    def test_penalty_is_squared_distance_from_global_weights(self) -> None:
        penalty = client_app._proximal_penalty(
            [np.array([3.0, 5.0])],
            [np.array([1.0, 2.0])],
        )

        self.assertAlmostEqual(float(penalty), 13.0)

    def test_configured_loss_penalizes_movement_from_round_weights(self) -> None:
        model = client_app.keras.Sequential(
            [
                client_app.keras.Input(shape=(1,)),
                client_app.keras.layers.Dense(1, activation="sigmoid", use_bias=False),
            ]
        )
        model.compile(optimizer="sgd", loss="binary_crossentropy")
        model.set_weights([np.array([[1.0]], dtype="float32")])
        client = client_app.SentimentClient.__new__(client_app.SentimentClient)
        client.model = model
        client._configure_proximal_loss(0.5)
        y_true = np.array([[1.0]], dtype="float32")
        y_pred = np.array([[0.5]], dtype="float32")

        loss_at_global = float(model.loss(y_true, y_pred))
        model.set_weights([np.array([[3.0]], dtype="float32")])
        loss_after_movement = float(model.loss(y_true, y_pred))

        self.assertAlmostEqual(loss_after_movement - loss_at_global, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
