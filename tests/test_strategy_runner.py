import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import keras
import numpy as np

from src.evaluation_artifact import load_scientific_protocol
from src.strategy_runner import (
    RowSplit,
    _convergence_round,
    _parameter_payload_bytes,
    _run_federated,
    _run_local_only,
    _train_one_epoch,
    _validate_args,
    aggregate_model_weights,
)


def runner_args(root: Path, strategy: str) -> argparse.Namespace:
    """Return fixed four-client strategy-runner arguments.

    Parameters
    ----------
    root : pathlib.Path
        Temporary output root.
    strategy : str
        Registered strategy identifier.

    Returns
    -------
    argparse.Namespace
        Complete runner arguments.
    """
    return argparse.Namespace(
        strategy=strategy,
        batch_size=64,
        client_count=4,
        client_data_dir=str(root / "client-{partition}"),
        epochs=20,
        evaluation_artifact_dir=root / "evaluation",
        output_dir=root,
        partition="iid_stratified",
        public_artifact_dir=root / "public",
        quiet=True,
        seed=67,
        validation_split=0.2,
    )


def canonical_result(labels: np.ndarray) -> dict[str, object]:
    """Return a canonical-evaluator-shaped result.

    Parameters
    ----------
    labels : numpy.ndarray
        Exact labels supplied to the evaluator.

    Returns
    -------
    dict of str to object
        Reported metrics and direct raw probabilities.
    """
    return {
        "accuracy": 0.5,
        "confusion_matrix": [[1, 0], [1, 0]],
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": None if np.unique(labels).size == 1 else 0.5,
        "roc_auc_status": (
            "undefined_single_class" if np.unique(labels).size == 1 else "defined"
        ),
        "probabilities": np.full(labels.shape, 0.25, dtype=np.float32),
    }


class StrategyAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        """Create the frozen four-client aggregation probe."""
        vectors = [
            [11.0, -8.0, 4.0, 3.0],
            [-10.0, 20.0, 21.0, -51.0],
            [14.0, -9.0, 7.0, -7.0],
            [9.0, -6.0, 5.0, 1.0],
        ]
        self.weights = [
            [
                np.asarray(vector[:2], dtype=np.float32),
                np.asarray(vector[2:], dtype=np.float32),
            ]
            for vector in vectors
        ]
        self.counts = [1, 2, 3, 4]

    def test_registered_flower_aggregations_match_golden_probe(self) -> None:
        expected = {
            "fedavg": [6.8999996, -1.9000001, 8.7, -11.6],
            "fedprox": [6.8999996, -1.9000001, 8.7, -11.6],
            "fedmedian": [10.0, -7.0, 6.0, -3.0],
            "fedtrimmedavg": [10.0, -7.0, 6.0, -3.0],
        }

        for strategy, vector in expected.items():
            with self.subTest(strategy=strategy):
                actual = aggregate_model_weights(strategy, self.weights, self.counts)
                np.testing.assert_array_equal(
                    np.concatenate(actual), np.asarray(vector, dtype=np.float32)
                )

    def test_huber_aggregation_preserves_model_shapes_and_float32(self) -> None:
        actual = aggregate_model_weights(
            "fedprox_huber", self.weights, self.counts, huber_threshold=10.0
        )

        self.assertEqual([array.shape for array in actual], [(2,), (2,)])
        self.assertEqual(
            [array.dtype for array in actual],
            [np.dtype(np.float32), np.dtype(np.float32)],
        )

    def test_invalid_aggregation_input_fails_without_repair(self) -> None:
        invalid = [
            ([], []),
            (self.weights, [1, 2, 3]),
            (self.weights, [1, 2, 3, True]),
            (
                [
                    *self.weights[:3],
                    [np.asarray([1.0], dtype=np.float32)],
                ],
                self.counts,
            ),
        ]

        for weights, counts in invalid:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                aggregate_model_weights("fedavg", weights, counts)

    def test_invalid_aggregation_output_fails_without_repair(self) -> None:
        valid = [array.copy() for array in self.weights[0]]
        invalid = [
            valid[:1],
            [np.asarray(1.0, dtype=np.float32), valid[1]],
            [np.asarray([], dtype=np.float32), valid[1]],
            [valid[0].astype(np.float64), valid[1]],
            [np.ones(3, dtype=np.float32), valid[1]],
            [np.asarray([np.inf, 0.0], dtype=np.float32), valid[1]],
        ]

        for aggregate in invalid:
            with (
                self.subTest(aggregate=aggregate),
                patch(
                    "src.strategy_runner.aggregate_inplace",
                    return_value=aggregate,
                ),
                self.assertRaises(ValueError),
            ):
                aggregate_model_weights("fedavg", self.weights, self.counts)

    def test_median_and_trimmed_average_reject_float32_overflow(self) -> None:
        maximum = np.finfo(np.float32).max
        weights = [[np.asarray([maximum], dtype=np.float32)] for _ in range(4)]

        for strategy in ("fedmedian", "fedtrimmedavg"):
            with (
                self.subTest(strategy=strategy),
                np.errstate(over="ignore"),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                aggregate_model_weights(strategy, weights, self.counts)

    def test_system_metrics_use_serialized_parameters_and_registered_convergence(
        self,
    ) -> None:
        payload_size = _parameter_payload_bytes(self.weights[0])
        curve = [
            {"round": 0, "accuracy": 0.4},
            {"round": 1, "accuracy": 0.8},
            {"round": 2, "accuracy": 0.82},
        ]

        self.assertGreater(payload_size, sum(array.nbytes for array in self.weights[0]))
        self.assertEqual(_convergence_round(curve), 1)


class StrategyExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        """Create compact four-client row and array fixtures."""
        self.protocol = load_scientific_protocol()
        self.manifest = MagicMock()
        self.splits: list[RowSplit] = [
            (
                ((f"train:{client_id}", "fitted", client_id % 2),),
                (
                    (f"train:{client_id + 4}", "validation", 0),
                    (f"train:{client_id + 8}", "validation", 1),
                ),
            )
            for client_id in range(4)
        ]
        self.train_data = (
            np.asarray([[1]], dtype=np.int32),
            np.asarray([0], dtype=np.float32),
        )
        self.mixed_validation = (
            np.asarray([[2], [3]], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int64),
        )

    def test_local_only_evaluates_each_live_epoch_and_scopes_only_single_class(
        self,
    ) -> None:
        single_class = (
            np.asarray([[2], [3]], dtype=np.int32),
            np.asarray([1, 1], dtype=np.int64),
        )
        models = [MagicMock() for _ in range(4)]
        evaluations: list[tuple[object, int, dict[str, object]]] = []

        for model in models:

            def fit(
                *_args: object,
                _model: object = model,
                **kwargs: object,
            ) -> None:
                callbacks = kwargs["callbacks"]
                epochs = kwargs["epochs"]
                assert isinstance(callbacks, list)
                assert isinstance(epochs, int)
                for callback in callbacks:
                    callback.set_model(_model)
                for epoch in range(epochs):
                    _model.live_epoch = epoch
                    for callback in callbacks:
                        callback.on_epoch_end(epoch)

            model.fit.side_effect = fit

        def evaluate(
            model: object,
            _tokens: np.ndarray,
            labels: np.ndarray,
            **kwargs: object,
        ) -> dict[str, object]:
            evaluations.append((model, model.live_epoch, kwargs))
            return canonical_result(labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = runner_args(output_dir, "local_only")
            with (
                patch(
                    "src.strategy_runner._validation_data",
                    side_effect=[
                        (self.train_data, single_class),
                        *[(self.train_data, self.mixed_validation)] * 3,
                    ],
                ),
                patch(
                    "src.strategy_runner._training_orders",
                    return_value=[np.asarray([0])] * 20,
                ),
                patch(
                    "src.strategy_runner.build_model_from_manifest",
                    side_effect=models,
                ),
                patch("src.strategy_runner.keras.utils.set_random_seed"),
                patch(
                    "src.strategy_runner._load_test_data",
                    return_value=self.mixed_validation,
                ),
                patch(
                    "src.strategy_runner.evaluate_classifier",
                    side_effect=evaluate,
                ),
            ):
                result, actual_models = _run_local_only(
                    args,
                    self.manifest,
                    self.protocol,
                    self.splits,
                    output_dir,
                )

        self.assertEqual(actual_models, models)
        self.assertTrue(all(model.fit.call_count == 1 for model in models))
        self.assertTrue(
            all(model.fit.call_args.kwargs["epochs"] == 20 for model in models)
        )
        self.assertEqual(len(evaluations), 84)
        self.assertEqual(
            [evaluation[2] for evaluation in evaluations[:20]],
            [{"evaluation_scope": "local_only_validation_only"}] * 20,
        )
        self.assertEqual([evaluation[2] for evaluation in evaluations[20:]], [{}] * 64)
        self.assertEqual(
            [evaluation[1] for evaluation in evaluations[:80]],
            list(range(20)) * 4,
        )
        self.assertTrue(
            all(len(client["validation"]) == 20 for client in result["clients"])
        )

    def test_federated_evaluates_fixed_union_immediately_after_every_round(
        self,
    ) -> None:
        events: list[str] = []
        weights = [np.asarray([1.0], dtype=np.float32)]
        global_model = MagicMock()
        global_model.get_weights.return_value = weights
        clients = [MagicMock() for _ in range(80)]
        for client in clients:
            client.get_weights.return_value = weights

        def train_epoch(*_args: object, **_kwargs: object) -> None:
            events.append("train")

        def aggregate(*_args: object, **_kwargs: object) -> list[np.ndarray]:
            events.append("aggregate")
            return weights

        def evaluate(
            _model: object,
            _tokens: np.ndarray,
            labels: np.ndarray,
        ) -> dict[str, object]:
            events.append("evaluate")
            return canonical_result(labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = runner_args(output_dir, "fedavg")
            with (
                patch(
                    "src.strategy_runner._validation_data",
                    return_value=(self.train_data, self.mixed_validation),
                ),
                patch(
                    "src.strategy_runner.tokenize_rows",
                    return_value=(
                        np.arange(8, dtype=np.int32).reshape(8, 1),
                        np.zeros(8, dtype=np.float32),
                    ),
                ),
                patch(
                    "src.strategy_runner._training_orders",
                    return_value=[np.asarray([0])],
                ),
                patch(
                    "src.strategy_runner._train_one_epoch",
                    side_effect=train_epoch,
                ),
                patch(
                    "src.strategy_runner.aggregate_model_weights",
                    side_effect=aggregate,
                ) as aggregate_weights,
                patch(
                    "src.strategy_runner.build_model_from_manifest",
                    side_effect=[global_model, *clients],
                ),
                patch("src.strategy_runner.keras.utils.set_random_seed"),
                patch(
                    "src.strategy_runner._load_test_data",
                    return_value=self.mixed_validation,
                ),
                patch(
                    "src.strategy_runner.evaluate_classifier",
                    side_effect=evaluate,
                ),
            ):
                result, models = _run_federated(
                    args,
                    self.manifest,
                    self.protocol,
                    self.splits,
                    output_dir,
                )

        self.assertEqual(models, [global_model])
        self.assertEqual(aggregate_weights.call_count, 20)
        self.assertEqual(len(result["validation"]), 20)
        self.assertEqual(
            [events[index : index + 6] for index in range(0, len(events) - 1, 6)],
            [["train"] * 4 + ["aggregate", "evaluate"]] * 20,
        )
        self.assertEqual(events[-1], "evaluate")

    def test_fedprox_epoch_updates_a_real_rank_one_label_batch(self) -> None:
        model = keras.Sequential(
            [
                keras.Input((1,)),
                keras.layers.Dense(
                    1,
                    activation="sigmoid",
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
                ),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")

        _train_one_epoch(
            model,
            (
                np.asarray([[0.0], [1.0]], dtype=np.float32),
                np.asarray([0.0, 1.0], dtype=np.float32),
            ),
            np.asarray([0, 1], dtype=np.int64),
            2,
            proximal_mu=0.1,
            quiet=True,
        )

        self.assertGreater(float(model.get_weights()[0][0, 0]), 0.0)

    def test_runner_rejects_malformed_client_template(self) -> None:
        args = runner_args(Path("output"), "fedavg")
        args.client_data_dir = "client-{unknown}"

        with self.assertRaisesRegex(ValueError, "contain"):
            _validate_args(args, self.protocol)


if __name__ == "__main__":
    unittest.main()
