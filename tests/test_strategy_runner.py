import unittest

import numpy as np

from src.strategy_runner import aggregate_model_weights


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
                actual = aggregate_model_weights(
                    strategy, self.weights, self.counts
                )
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


if __name__ == "__main__":
    unittest.main()
