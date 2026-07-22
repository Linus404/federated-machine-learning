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

    def test_huber_aggregate_rejects_non_protocol_inputs(self) -> None:
        invalid_inputs = [
            ([], [], 1.0),
            ([np.array([0.0]), np.array([1.0, 2.0])], [1, 1], 1.0),
            ([np.array([np.nan])], [1], 1.0),
            ([np.array([0.0])], [0], 1.0),
            ([np.array([0.0])], [1], 0.0),
        ]

        for vectors, counts, threshold in invalid_inputs:
            with self.subTest(vectors=vectors, counts=counts, threshold=threshold):
                with self.assertRaises(ValueError):
                    huber_aggregate(vectors, counts, threshold)

    def test_huber_aggregate_accepts_only_float32_and_float64_vectors(self) -> None:
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                result = huber_aggregate(
                    [np.array([0.0], dtype=dtype), np.array([2.0], dtype=dtype)],
                    [1, 1],
                    1.0,
                )

                self.assertEqual(result.dtype, np.dtype(np.float64))
                np.testing.assert_array_equal(result, np.array([1.0]))

    def test_huber_aggregate_rejects_unsupported_dtype_before_conversion(self) -> None:
        unsupported_vectors = [
            np.array([1.0], dtype=np.float16),
            np.array([1 + 2j], dtype=np.complex64),
            np.array([1 + 2j], dtype=np.complex128),
            np.array([1], dtype=np.int64),
            np.array([True], dtype=np.bool_),
            np.array(["1.0"]),
            np.array([1.0], dtype=object),
        ]

        for vector in unsupported_vectors:
            with self.subTest(dtype=vector.dtype):
                with self.assertRaisesRegex(ValueError, "float32 or float64"):
                    huber_aggregate([vector], [1], 1.0)

    def test_flatten_rejects_unsupported_tensor_before_dtype_promotion(self) -> None:
        unsupported_weight_sets = [
            [
                np.array([1.0], dtype=np.float32),
                np.array([2], dtype=np.int64),
            ],
            [
                np.array([1.0], dtype=np.float32),
                np.array([2 + 3j], dtype=np.complex64),
            ],
        ]

        for weights in unsupported_weight_sets:
            with self.subTest(dtypes=[weight.dtype for weight in weights]):
                with self.assertRaisesRegex(ValueError, "float32 or float64"):
                    _flatten(weights)


if __name__ == "__main__":
    unittest.main()
