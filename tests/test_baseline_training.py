import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np

from src.baseline_training import run


def baseline_args(root: Path, baseline: str) -> argparse.Namespace:
    """Return a compact baseline command fixture.

    Parameters
    ----------
    root : pathlib.Path
        Temporary test root.
    baseline : str
        Baseline mode under test.

    Returns
    -------
    argparse.Namespace
        Complete command arguments for two prepared clients.
    """
    return argparse.Namespace(
        baseline=baseline,
        batch_size=4,
        client_count=2,
        client_data_dir=str(root / "client-{partition}"),
        epochs=1,
        evaluation_artifact_dir=root / "evaluation",
        output_dir=root / "output",
        public_artifact_dir=root / "public",
        quiet=True,
        seed=67,
        validation_split=0.25,
    )


def training_snapshot(*indices: int) -> SimpleNamespace:
    """Return a validated-client-shaped snapshot fixture.

    Parameters
    ----------
    *indices : int
        Official training row indices owned by the client.

    Returns
    -------
    types.SimpleNamespace
        Snapshot exposing immutable-style rows.
    """
    return SimpleNamespace(
        rows=tuple(
            (f"train:{index}", f"review {index}", index % 2) for index in indices
        )
    )


def model(history: SimpleNamespace, evaluation: tuple[float, float]) -> MagicMock:
    """Return a model double that writes its requested model path.

    Parameters
    ----------
    history : types.SimpleNamespace
        Keras-shaped training history.
    evaluation : tuple of float
        Test loss and accuracy.

    Returns
    -------
    unittest.mock.MagicMock
        Configured model double.
    """
    value = MagicMock()
    value.fit.return_value = history
    value.evaluate.return_value = evaluation
    value.save.side_effect = lambda path: Path(path).write_bytes(b"model")
    return value


class BaselineTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        """Create shared validated artifact and array fixtures."""
        self.manifest = SimpleNamespace(payload={"dataset": {"rows": 4}})
        self.snapshots = [training_snapshot(0, 2), training_snapshot(1, 3)]
        self.splits = [
            (
                (np.array([[0]], dtype=np.int32), np.array([0], dtype=np.float32)),
                (np.array([[2]], dtype=np.int32), np.array([1], dtype=np.float32)),
            ),
            (
                (np.array([[1]], dtype=np.int32), np.array([1], dtype=np.float32)),
                (np.array([[3]], dtype=np.int32), np.array([0], dtype=np.float32)),
            ),
        ]
        self.test_data = (
            np.array([[10], [11]], dtype=np.int32),
            np.array([0, 1], dtype=np.float32),
        )
        self.history = SimpleNamespace(
            history={"val_loss": [0.5], "val_accuracy": [0.75]}
        )

    def test_centralized_trains_before_loading_untouched_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = baseline_args(Path(tmpdir), "centralized")
            trained_model = model(self.history, (0.4, 0.8))
            events: list[str] = []

            def record_fit(*_args: object, **_kwargs: object) -> SimpleNamespace:
                events.append("fit")
                return self.history

            trained_model.fit.side_effect = record_fit

            def load_test(_path: Path) -> SimpleNamespace:
                self.assertEqual(events, ["fit"])
                events.append("test")
                return SimpleNamespace(rows=(("test:0", "test", 0),))

            with (
                patch(
                    "src.baseline_training.load_app_manifest",
                    return_value=self.manifest,
                ),
                patch(
                    "src.baseline_training.load_client_shard_snapshot",
                    side_effect=self.snapshots,
                ) as load_shard,
                patch(
                    "src.baseline_training._tokenize_client_shard",
                    side_effect=self.splits,
                ),
                patch(
                    "src.baseline_training.load_evaluation_artifact_snapshot",
                    side_effect=load_test,
                ),
                patch(
                    "src.baseline_training.tokenize_rows", return_value=self.test_data
                ),
                patch(
                    "src.baseline_training.build_model_from_manifest",
                    return_value=trained_model,
                ),
                patch("src.baseline_training.keras.utils.set_random_seed") as set_seed,
            ):
                result = run(args)

            self.assertEqual(events, ["fit", "test"])
            load_shard.assert_has_calls(
                [
                    call(str(Path(tmpdir) / "client-0"), self.manifest, 0),
                    call(str(Path(tmpdir) / "client-1"), self.manifest, 1),
                ]
            )
            set_seed.assert_called_once_with(67)
            fit_args, fit_kwargs = trained_model.fit.call_args
            np.testing.assert_array_equal(
                fit_args[0], np.array([[0], [1]], dtype=np.int32)
            )
            np.testing.assert_array_equal(
                fit_kwargs["validation_data"][0],
                np.array([[2], [3]], dtype=np.int32),
            )
            self.assertFalse(fit_kwargs["shuffle"])
            self.assertEqual(result["test"], {"loss": 0.4, "accuracy": 0.8})
            self.assertTrue((args.output_dir / "centralized.keras").is_file())
            self.assertEqual(
                json.loads((args.output_dir / "results.json").read_text()), result
            )

    def test_local_only_trains_every_model_before_shared_test_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = baseline_args(Path(tmpdir), "local-only")
            models = [
                model(self.history, (0.6, 0.7)),
                model(self.history, (0.4, 0.9)),
            ]
            events: list[str] = []
            for client_id, trained_model in enumerate(models):

                def record_fit(
                    *_args: object,
                    client_id: int = client_id,
                    **_kwargs: object,
                ) -> SimpleNamespace:
                    events.append(f"fit-{client_id}")
                    return self.history

                trained_model.fit.side_effect = record_fit

            def load_test(_path: Path) -> SimpleNamespace:
                self.assertEqual(events, ["fit-0", "fit-1"])
                events.append("test")
                return SimpleNamespace(rows=(("test:0", "test", 0),))

            with (
                patch(
                    "src.baseline_training.load_app_manifest",
                    return_value=self.manifest,
                ),
                patch(
                    "src.baseline_training.load_client_shard_snapshot",
                    side_effect=self.snapshots,
                ),
                patch(
                    "src.baseline_training._tokenize_client_shard",
                    side_effect=self.splits,
                ),
                patch(
                    "src.baseline_training.load_evaluation_artifact_snapshot",
                    side_effect=load_test,
                ),
                patch(
                    "src.baseline_training.tokenize_rows", return_value=self.test_data
                ),
                patch(
                    "src.baseline_training.build_model_from_manifest",
                    side_effect=models,
                ),
                patch("src.baseline_training.keras.utils.set_random_seed") as set_seed,
            ):
                result = run(args)

            self.assertEqual(events, ["fit-0", "fit-1", "test"])
            self.assertEqual(set_seed.call_args_list, [call(67), call(68)])
            self.assertEqual(result["test_mean"], {"loss": 0.5, "accuracy": 0.8})
            self.assertEqual(
                [client["test"] for client in result["clients"]],
                [
                    {"loss": 0.6, "accuracy": 0.7},
                    {"loss": 0.4, "accuracy": 0.9},
                ],
            )
            self.assertTrue((args.output_dir / "client-0.keras").is_file())
            self.assertTrue((args.output_dir / "client-1.keras").is_file())


if __name__ == "__main__":
    unittest.main()
