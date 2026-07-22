import tomllib
import unittest
from pathlib import Path

import keras
import numpy as np

from src.client_app import SentimentClient
from src.contracts import dirichlet_split
from src.huber_strategy import huber_aggregate
from src.local_training import _stratified_split_indices, build_model


class ScientificContractTests(unittest.TestCase):
    def test_default_dirichlet_split_is_deterministic(self) -> None:
        labels = np.asarray([0, 1] * 6)

        split = dirichlet_split(labels, num_clients=4)

        self.assertEqual(
            split,
            {
                0: [0, 9, 6, 7],
                1: [4, 1],
                2: [3, 10, 8, 11, 5],
                3: [2],
            },
        )
        self.assertEqual(
            sorted(index for shard in split.values() for index in shard),
            list(range(12)),
        )

    def test_default_validation_split_is_deterministic_and_stratified(self) -> None:
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

        train_indices, validation_indices = _stratified_split_indices(labels, 0.25)

        np.testing.assert_array_equal(train_indices, [7, 3, 0, 1, 4, 5])
        np.testing.assert_array_equal(validation_indices, [6, 2])
        np.testing.assert_array_equal(np.unique(labels[train_indices]), [0, 1])
        np.testing.assert_array_equal(np.unique(labels[validation_indices]), [0, 1])

    def test_model_architecture_and_compile_contract(self) -> None:
        model = build_model(vocab_size=17, sequence_length=11, embedding_dim=5)

        embedding = model.get_layer("token_embedding")
        convolution = model.get_layer("padding_safe_conv")
        dense_layers = [
            layer for layer in model.layers if isinstance(layer, keras.layers.Dense)
        ]
        dropout = next(
            layer for layer in model.layers if isinstance(layer, keras.layers.Dropout)
        )

        self.assertEqual(model.input_shape, (None, 11))
        self.assertEqual((embedding.input_dim, embedding.output_dim), (17, 5))
        self.assertTrue(embedding.trainable)
        self.assertEqual(convolution.filters, 64)
        self.assertEqual(convolution.kernel_size, (3,))
        self.assertEqual(convolution.padding, "same")
        self.assertEqual(convolution.activation.__name__, "relu")
        self.assertFalse(convolution.use_bias)
        self.assertEqual(
            [(layer.units, layer.activation.__name__) for layer in dense_layers],
            [(32, "relu"), (1, "sigmoid")],
        )
        self.assertEqual(dropout.rate, 0.3)
        self.assertIsInstance(dropout.seed, int)
        self.assertIsInstance(model.optimizer, keras.optimizers.Adam)
        self.assertEqual(model.loss, "binary_crossentropy")

    def test_default_application_training_configuration_is_stable(self) -> None:
        config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
            "tool"
        ]["flwr"]["app"]["config"]

        self.assertEqual(
            config,
            {
                "num-server-rounds": 20,
                "local-epochs": 1,
                "batch-size": 64,
                "validation-split": 0.2,
                "master-seed": 67,
                "public-artifact-dir": "artifacts/public",
                "server-artifact-dir": "artifacts/server",
                "artifact-retention-runs": 10,
                "expected-client-count": 4,
                "client-data-dir": "artifacts/clients/client-{partition}",
                "proximal-mu": 0.1,
                "use-huber": False,
                "huber-threshold": 10.0,
                "use-update-noise": False,
                "update-noise-l2-norm-clip": 1.0,
                "update-noise-multiplier": 0.001,
            },
        )

    def test_dashboard_documentation_uses_default_server_artifact_dir(self) -> None:
        config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
            "tool"
        ]["flwr"]["app"]["config"]
        artifact_dir = config["server-artifact-dir"]
        readme = Path("README.md").read_text(encoding="utf-8")
        compose = Path("compose.yaml").read_text(encoding="utf-8")

        self.assertIn(f"FML_SERVER_ARTIFACT_DIR={artifact_dir} ", readme)
        self.assertNotIn(f'$env:FML_SERVER_ARTIFACT_DIR = "{artifact_dir}"', readme)
        self.assertIn("FML_SERVER_ARTIFACT_DIR: /app/artifacts/server", compose)

    def test_huber_aggregation_result_is_stable(self) -> None:
        result = huber_aggregate(
            [
                np.asarray([0.0], dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
                np.asarray([100.0], dtype=np.float32),
            ],
            [1, 1, 1],
            threshold=1.0,
        )

        self.assertEqual(result.dtype, np.dtype(np.float32))
        self.assertAlmostEqual(float(result[0]), 1.0178052186965942)

    def test_update_noise_clips_each_weight_update_before_adding_noise(self) -> None:
        client = SentimentClient.__new__(SentimentClient)
        client.client_id = 0
        client.master_seed = 67
        client.update_noise_l2_norm_clip = 2.0
        client.update_noise_multiplier = 0.0
        before = [np.asarray([1.0, 1.0], dtype=np.float32)]
        after = [np.asarray([4.0, 5.0], dtype=np.float32)]

        result = client._add_update_noise(before, after, 1)

        self.assertEqual(result[0].dtype, np.dtype(np.float32))
        np.testing.assert_allclose(result[0], [2.2, 2.6])
        self.assertAlmostEqual(
            float(np.linalg.norm(result[0] - before[0])), 2.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
