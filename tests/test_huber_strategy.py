import unittest

import numpy as np

from src.huber_strategy import _flatten, _unflatten, huber_aggregate


class HuberAggregationTests(unittest.TestCase):
    def test_huber_aggregate_downweights_outlier(self) -> None:
        vectors = [
            np.array([0.0]),
            np.array([1.0]),
            np.array([100.0]),
        ]
        sample_counts = [1, 1, 1]

        robust = huber_aggregate(vectors, sample_counts, threshold=1.0)
        weighted = np.average(vectors, axis=0, weights=sample_counts)

        self.assertLess(abs(float(robust[0]) - 0.5), abs(float(weighted[0]) - 0.5))
        self.assertLess(float(robust[0]), 10.0)

    def test_flatten_unflatten_preserves_reference_shapes(self) -> None:
        reference = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([5.0, 6.0, 7.0]),
        ]

        vector = _flatten(reference)
        restored = _unflatten(vector, reference)

        self.assertEqual([array.shape for array in restored], [(2, 2), (3,)])
        for actual, expected in zip(restored, reference):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
