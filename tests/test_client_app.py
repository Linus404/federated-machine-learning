import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import src
import src.client_app as client_app


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
                "data-dir": "data",
                "local-epochs": 2,
                "batch-size": 8,
                "embedding-dim": 16,
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
            data_dir="data",
            partition=3,
            epochs=2,
            batch_size=8,
            embedding_dim=16,
            validation_split=0.25,
            use_update_noise=True,
            update_noise_l2_norm_clip=2.0,
            update_noise_multiplier=0.5,
        )


class FakeHistory:
    history = {"loss": [0.2], "accuracy": [0.9]}


class FakeModel:
    def __init__(self) -> None:
        self.weights = [np.array([0.0, 0.0], dtype="float32")]

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
        client.use_update_noise = use_update_noise
        client.update_noise_l2_norm_clip = 1.0
        client.update_noise_multiplier = 0.001
        return client

    def test_fit_returns_trained_weights_when_update_noise_is_disabled(self) -> None:
        client = self.make_client(use_update_noise=False)
        client._add_update_noise = Mock(return_value=[np.array([9.0], dtype="float32")])

        weights, num_examples, metrics = client.fit(
            [np.array([0.0, 0.0], dtype="float32")], {}
        )

        client._add_update_noise.assert_not_called()
        np.testing.assert_array_equal(weights[0], np.array([1.0, 2.0], dtype="float32"))
        self.assertEqual(num_examples, 2)
        self.assertEqual(metrics, {"loss": 0.2, "accuracy": 0.9})

    def test_fit_applies_update_noise_when_enabled(self) -> None:
        client = self.make_client(use_update_noise=True)
        noisy_weights = [np.array([9.0], dtype="float32")]
        client._add_update_noise = Mock(return_value=noisy_weights)

        weights, _, _ = client.fit([np.array([0.0, 0.0], dtype="float32")], {})

        client._add_update_noise.assert_called_once()
        self.assertIs(weights, noisy_weights)


if __name__ == "__main__":
    unittest.main()
