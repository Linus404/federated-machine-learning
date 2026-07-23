import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import keras
import numpy as np

from src.baseline_training import (
    _derive_seed,
    _registered_iid_splits,
    _training_orders,
    parse_args,
    run,
)
from src.evaluation_artifact import load_scientific_protocol
from src.local_training import build_model


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
        Complete command arguments for four prepared clients.
    """
    return argparse.Namespace(
        baseline=baseline,
        batch_size=64,
        client_count=4,
        client_data_dir=str(root / "client-{partition}"),
        epochs=20,
        evaluation_artifact_dir=root / "evaluation",
        output_dir=root / "output",
        public_artifact_dir=root / "public",
        quiet=True,
        seed=67,
        validation_split=0.2,
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


def canonical_evaluation(score: float) -> dict[str, object]:
    """Return a canonical evaluator-shaped result fixture.

    Parameters
    ----------
    score : float
        Scalar value used for every reported metric.

    Returns
    -------
    dict of str to object
        Complete evaluator result including raw probabilities.
    """
    return {
        "accuracy": score,
        "confusion_matrix": [[1, 0], [0, 1]],
        "precision": score,
        "recall": score,
        "f1": score,
        "roc_auc": score,
        "roc_auc_status": "defined",
        "probabilities": np.array([0.25, 0.75], dtype=np.float32),
    }


class BaselineTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        """Create shared validated artifact and array fixtures."""
        self.protocol = load_scientific_protocol()
        self.manifest = SimpleNamespace(payload={"dataset": {"rows": 8}})
        self.snapshots = [
            training_snapshot(0, 4),
            training_snapshot(1, 5),
            training_snapshot(2, 6),
            training_snapshot(3, 7),
        ]
        self.row_splits = [
            ((self.snapshots[0].rows[0],), (self.snapshots[0].rows[1],)),
            ((self.snapshots[1].rows[0],), (self.snapshots[1].rows[1],)),
            ((self.snapshots[2].rows[0],), (self.snapshots[2].rows[1],)),
            ((self.snapshots[3].rows[0],), (self.snapshots[3].rows[1],)),
        ]
        self.array_splits = [
            (
                (
                    np.array([[client_id]], dtype=np.int32),
                    np.array([client_id % 2], dtype=np.float32),
                ),
                (
                    np.array([[client_id + 4]], dtype=np.int32),
                    np.array([client_id % 2], dtype=np.float32),
                ),
            )
            for client_id in range(4)
        ]
        self.combined_rows = (
            tuple(snapshot.rows[0] for snapshot in self.snapshots),
            tuple(snapshot.rows[1] for snapshot in self.snapshots),
        )
        self.combined_arrays = (
            (
                np.array([[0], [1], [2], [3]], dtype=np.int32),
                np.array([0, 1, 0, 1], dtype=np.float32),
            ),
            (
                np.array([[4], [5], [6], [7]], dtype=np.int32),
                np.array([0, 1, 0, 1], dtype=np.float32),
            ),
        )
        self.test_data = (
            np.array([[10], [11]], dtype=np.int32),
            np.array([0, 1], dtype=np.float32),
        )
        self.history = SimpleNamespace(
            history={"val_loss": [0.5] * 20, "val_accuracy": [0.75] * 20}
        )

    def test_registered_iid_and_validation_assignments_match_golden_rows(self) -> None:
        rows = tuple(
            (f"train:{index}", f"review {index}", 0 if index < 20 else 1)
            for index in range(40)
        )
        splits = _registered_iid_splits(
            [SimpleNamespace(rows=rows)], self.protocol, 67, 4, 0.2
        )

        self.assertEqual(
            [
                [int(row[0].removeprefix("train:")) for row in fitted]
                for fitted, _ in splits
            ],
            [
                [6, 13, 16, 19, 35, 37, 38, 39],
                [2, 3, 7, 17, 23, 27, 31, 36],
                [8, 10, 15, 18, 22, 28, 29, 30],
                [0, 5, 9, 12, 21, 24, 26, 32],
            ],
        )
        self.assertEqual(
            [
                [int(row[0].removeprefix("train:")) for row in validation]
                for _, validation in splits
            ],
            [[4, 20], [14, 34], [11, 33], [1, 25]],
        )

    def test_registered_seed_and_training_orders_match_golden_values(self) -> None:
        self.assertEqual(
            _derive_seed(
                67,
                "partition/iid/4/round--1/epoch--1/client--1/label-0",
            ),
            7719479962854520267,
        )
        rows = tuple(
            (f"train:{index}", f"review {index}", index % 2) for index in range(8)
        )
        orders = _training_orders(rows, self.protocol, 67, "CELL", -1, 2)
        self.assertEqual(
            [order.tolist() for order in orders],
            [[2, 6, 5, 3, 0, 7, 4, 1], [7, 0, 2, 4, 3, 6, 1, 5]],
        )

    def test_model_uses_registered_explicit_dropout_seed(self) -> None:
        built = build_model(16, 8, 4, dropout_seed=123)
        dropout = next(
            layer for layer in built.layers if isinstance(layer, keras.layers.Dropout)
        )
        self.assertEqual(dropout.seed, 123)

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
                    "src.baseline_training.load_scientific_protocol",
                    return_value=self.protocol,
                ),
                patch(
                    "src.baseline_training.load_app_manifest",
                    return_value=self.manifest,
                ),
                patch(
                    "src.baseline_training.load_client_shard_snapshot",
                    side_effect=self.snapshots,
                ) as load_shard,
                patch(
                    "src.baseline_training._registered_iid_splits",
                    return_value=self.row_splits,
                ),
                patch(
                    "src.baseline_training._combined_split",
                    return_value=(self.combined_rows, self.combined_arrays),
                ),
                patch("src.baseline_training._model_seeds", return_value=(101, 202)),
                patch(
                    "src.baseline_training._training_orders",
                    return_value=[np.arange(4)] * 20,
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
                ) as build_model_from_manifest,
                patch(
                    "src.baseline_training.evaluate_classifier",
                    return_value=canonical_evaluation(0.8),
                ),
                patch("src.baseline_training.keras.utils.set_random_seed") as set_seed,
            ):
                result = run(args)

            self.assertEqual(events, ["fit", "test"])
            load_shard.assert_has_calls(
                [
                    call(str(Path(tmpdir) / "client-0"), self.manifest, 0),
                    call(str(Path(tmpdir) / "client-1"), self.manifest, 1),
                    call(str(Path(tmpdir) / "client-2"), self.manifest, 2),
                    call(str(Path(tmpdir) / "client-3"), self.manifest, 3),
                ]
            )
            set_seed.assert_called_once_with(101)
            build_model_from_manifest.assert_called_once_with(
                self.manifest, dropout_seed=202
            )
            fit_args, fit_kwargs = trained_model.fit.call_args
            batches = fit_args[0]
            np.testing.assert_array_equal(
                batches[0][0], np.array([[0], [1], [2], [3]], dtype=np.int32)
            )
            np.testing.assert_array_equal(
                fit_kwargs["validation_data"][0],
                np.array([[4], [5], [6], [7]], dtype=np.int32),
            )
            self.assertFalse(fit_kwargs["shuffle"])
            self.assertEqual(
                result["test"],
                {
                    "accuracy": 0.8,
                    "confusion_matrix": [[1, 0], [0, 1]],
                    "precision": 0.8,
                    "recall": 0.8,
                    "f1": 0.8,
                    "roc_auc": 0.8,
                    "roc_auc_status": "defined",
                },
            )
            self.assertEqual(
                result["config"],
                {
                    "batch_size": 64,
                    "client_count": 4,
                    "epochs": 20,
                    "partition": "iid_stratified",
                    "seed": 67,
                    "validation_split": 0.2,
                },
            )
            self.assertTrue((args.output_dir / "centralized.keras").is_file())
            np.testing.assert_array_equal(
                np.load(args.output_dir / "centralized-predictions.npy"), [0.25, 0.75]
            )
            self.assertEqual(
                json.loads((args.output_dir / "results.json").read_text()), result
            )

    def test_local_only_trains_every_model_before_shared_test_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = baseline_args(Path(tmpdir), "local-only")
            models = [
                model(self.history, (0.6, 0.7)),
                model(self.history, (0.4, 0.9)),
                model(self.history, (0.8, 0.5)),
                model(self.history, (0.2, 0.9)),
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
                self.assertEqual(events, ["fit-0", "fit-1", "fit-2", "fit-3"])
                events.append("test")
                return SimpleNamespace(rows=(("test:0", "test", 0),))

            with (
                patch(
                    "src.baseline_training.load_scientific_protocol",
                    return_value=self.protocol,
                ),
                patch(
                    "src.baseline_training.load_app_manifest",
                    return_value=self.manifest,
                ),
                patch(
                    "src.baseline_training.load_client_shard_snapshot",
                    side_effect=self.snapshots,
                ),
                patch(
                    "src.baseline_training._registered_iid_splits",
                    return_value=self.row_splits,
                ),
                patch(
                    "src.baseline_training._tokenize_split",
                    side_effect=self.array_splits,
                ),
                patch(
                    "src.baseline_training._model_seeds",
                    side_effect=[(101, 201), (102, 202), (103, 203), (104, 204)],
                ),
                patch(
                    "src.baseline_training._training_orders",
                    side_effect=[[np.array([0])] * 20 for _ in range(4)],
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
                ) as build_model_from_manifest,
                patch(
                    "src.baseline_training.evaluate_classifier",
                    side_effect=[
                        canonical_evaluation(score) for score in (0.7, 0.9, 0.5, 0.9)
                    ],
                ),
                patch("src.baseline_training.keras.utils.set_random_seed") as set_seed,
            ):
                result = run(args)

            self.assertEqual(events, ["fit-0", "fit-1", "fit-2", "fit-3", "test"])
            self.assertEqual(
                set_seed.call_args_list, [call(101), call(102), call(103), call(104)]
            )
            self.assertEqual(
                build_model_from_manifest.call_args_list,
                [
                    call(self.manifest, dropout_seed=201),
                    call(self.manifest, dropout_seed=202),
                    call(self.manifest, dropout_seed=203),
                    call(self.manifest, dropout_seed=204),
                ],
            )
            self.assertEqual(
                result["test_mean"],
                {
                    "accuracy": 0.75,
                    "precision": 0.75,
                    "recall": 0.75,
                    "f1": 0.75,
                    "roc_auc": 0.75,
                },
            )
            self.assertEqual(
                [client["test"] for client in result["clients"]],
                [
                    {
                        key: value
                        for key, value in canonical_evaluation(score).items()
                        if key != "probabilities"
                    }
                    for score in (0.7, 0.9, 0.5, 0.9)
                ],
            )
            for client_id in range(4):
                self.assertTrue(
                    (args.output_dir / f"client-{client_id}.keras").is_file()
                )
                self.assertTrue(
                    (args.output_dir / f"client-{client_id}-predictions.npy").is_file()
                )

    def test_save_failure_removes_owned_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = baseline_args(Path(tmpdir), "centralized")
            trained_model = model(self.history, (0.4, 0.8))
            trained_model.save.side_effect = OSError("model save failed")

            with (
                patch(
                    "src.baseline_training.load_scientific_protocol",
                    return_value=self.protocol,
                ),
                patch(
                    "src.baseline_training.load_app_manifest",
                    return_value=self.manifest,
                ),
                patch(
                    "src.baseline_training.load_client_shard_snapshot",
                    side_effect=self.snapshots,
                ),
                patch(
                    "src.baseline_training._registered_iid_splits",
                    return_value=self.row_splits,
                ),
                patch(
                    "src.baseline_training._combined_split",
                    return_value=(self.combined_rows, self.combined_arrays),
                ),
                patch("src.baseline_training._model_seeds", return_value=(101, 202)),
                patch(
                    "src.baseline_training._training_orders",
                    return_value=[np.arange(4)] * 20,
                ),
                patch(
                    "src.baseline_training.load_evaluation_artifact_snapshot",
                    return_value=SimpleNamespace(rows=(("test:0", "test", 0),)),
                ),
                patch(
                    "src.baseline_training.tokenize_rows", return_value=self.test_data
                ),
                patch(
                    "src.baseline_training.build_model_from_manifest",
                    return_value=trained_model,
                ),
                patch(
                    "src.baseline_training.evaluate_classifier",
                    return_value=canonical_evaluation(0.8),
                ),
                patch("src.baseline_training.keras.utils.set_random_seed"),
            ):
                with self.assertRaisesRegex(OSError, "model save failed"):
                    run(args)

            self.assertFalse(args.output_dir.exists())

    def test_invalid_client_template_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = baseline_args(Path(tmpdir), "centralized")
            args.client_data_dir = "{{partition}}"

            with self.assertRaisesRegex(ValueError, "contain"):
                run(args)

            self.assertFalse(args.output_dir.exists())

    def test_nonregistered_training_overrides_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for field, value in (
                ("batch_size", 4),
                ("client_count", 2),
                ("epochs", 1),
                ("seed", 68),
                ("validation_split", 0.25),
            ):
                with self.subTest(field=field):
                    args = baseline_args(Path(tmpdir), "centralized")
                    setattr(args, field, value)

                    with self.assertRaisesRegex(ValueError, "frozen"):
                        run(args)

                    self.assertFalse(args.output_dir.exists())

    def test_cli_does_not_expose_frozen_training_overrides(self) -> None:
        for option in (
            "--batch-size",
            "--client-count",
            "--epochs",
            "--validation-split",
        ):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["centralized", "--output-dir", "baseline", option, "1"])


if __name__ == "__main__":
    unittest.main()
