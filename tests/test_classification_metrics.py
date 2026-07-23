import unittest
from unittest.mock import MagicMock

import numpy as np

from src.classification_metrics import (
    LOCAL_ONLY_VALIDATION_SCOPE,
    classification_metrics,
    evaluate_classifier,
)


class ClassificationMetricTests(unittest.TestCase):
    def test_mixed_ties_match_frozen_golden_vector(self) -> None:
        labels = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
        probabilities = np.array([0.1, 0.5, 0.5, 0.9, 0.2, 0.8], dtype=np.float32)

        result = classification_metrics(labels, probabilities)

        np.testing.assert_array_equal(
            result["predicted_positive"], [False, True, True, True, False, True]
        )
        self.assertEqual(result["confusion_matrix"], [[1, 2], [1, 2]])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 2 / 3)
        self.assertEqual(result["f1"], 4 / 7)
        np.testing.assert_array_equal(
            result["roc_thresholds"],
            np.asarray(
                [
                    np.inf,
                    np.float32(0.9),
                    np.float32(0.8),
                    np.float32(0.5),
                    np.float32(0.2),
                    np.float32(0.1),
                ],
                dtype=np.float64,
            ),
        )
        np.testing.assert_array_equal(result["fpr"], [0, 0, 1 / 3, 2 / 3, 2 / 3, 1])
        np.testing.assert_array_equal(result["tpr"], [0, 1 / 3, 1 / 3, 2 / 3, 1, 1])
        self.assertEqual(result["roc_auc"], 0.6111111111111112)
        self.assertEqual(result["roc_auc_status"], "defined")

    def test_tied_scores_use_registered_zero_division(self) -> None:
        result = classification_metrics(
            np.array([0, 1, 0, 1], dtype=np.int64),
            np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32),
        )
        self.assertEqual(result["confusion_matrix"], [[2, 0], [2, 0]])
        self.assertEqual(
            [
                result[name]
                for name in ("accuracy", "precision", "recall", "f1", "roc_auc")
            ],
            [0.5, 0.0, 0.0, 0.0, 0.5],
        )

    def test_local_only_single_class_roc_is_explicitly_undefined(self) -> None:
        result = classification_metrics(
            np.array([1, 1], dtype=np.int64),
            np.array([0.2, 0.8], dtype=np.float32),
            evaluation_scope=LOCAL_ONLY_VALIDATION_SCOPE,
        )
        self.assertEqual(result["confusion_matrix"], [[0, 0], [1, 1]])
        self.assertEqual(result["f1"], 2 / 3)
        for name in ("roc_thresholds", "fpr", "tpr", "roc_auc"):
            self.assertIsNone(result[name])
        self.assertEqual(result["roc_auc_status"], "undefined_single_class")

    def test_hostile_metric_inputs_are_rejected_without_repair(self) -> None:
        labels = np.array([0, 1], dtype=np.int64)
        probabilities = np.array([0.25, 0.75], dtype=np.float32)
        hostile = [
            (labels.astype(np.int32), probabilities),
            (labels, probabilities.astype(np.float64)),
            (labels[:1], probabilities),
            (labels.reshape(1, 2), probabilities),
            (np.array([0, 2], dtype=np.int64), probabilities),
            (np.array([1, 1], dtype=np.int64), probabilities),
            (labels, np.array([np.nan, 0.5], dtype=np.float32)),
            (labels, np.array([-0.1, 0.5], dtype=np.float32)),
        ]
        for hostile_labels, hostile_probabilities in hostile:
            with self.subTest(
                labels=hostile_labels, probabilities=hostile_probabilities
            ):
                with self.assertRaises(ValueError):
                    classification_metrics(hostile_labels, hostile_probabilities)

    def test_inference_preserves_batches_order_shape_and_dtype(self) -> None:
        token_ids = np.arange(257 * 2, dtype=np.int32).reshape(257, 2)
        labels = np.resize(np.array([0, 1], dtype=np.int64), 257)
        labels[-1] = 1
        model = MagicMock(
            side_effect=[
                np.linspace(0.1, 0.9, 256, dtype=np.float32).reshape(-1, 1),
                np.array([[0.75]], dtype=np.float32),
            ]
        )

        result = evaluate_classifier(model, token_ids, labels)

        self.assertEqual(len(model.call_args_list), 2)
        np.testing.assert_array_equal(model.call_args_list[0].args[0], token_ids[:256])
        np.testing.assert_array_equal(model.call_args_list[1].args[0], token_ids[256:])
        self.assertEqual(
            [invocation.kwargs for invocation in model.call_args_list],
            [{"training": False}, {"training": False}],
        )
        self.assertEqual(result["probabilities"].shape, (257,))
        self.assertEqual(result["probabilities"].dtype, np.dtype("float32"))

    def test_inference_rejects_wrong_model_output_contract(self) -> None:
        token_ids = np.ones((2, 3), dtype=np.int32)
        for output in (
            np.array([0.2, 0.8], dtype=np.float32),
            np.array([[0.2], [0.8]], dtype=np.float64),
        ):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    evaluate_classifier(
                        MagicMock(return_value=output),
                        token_ids,
                        np.array([0, 1], dtype=np.int64),
                    )


if __name__ == "__main__":
    unittest.main()
