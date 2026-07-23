"""Tests for deterministic aggregation scaling and attack benchmarks."""

from __future__ import annotations

import unittest

from src.aggregation_benchmark import run_benchmark


class AggregationBenchmarkTests(unittest.TestCase):
    def test_benchmark_covers_larger_client_scales_and_attacks(self) -> None:
        result = run_benchmark(
            dimensions=16,
            repeats=2,
            seed=67,
            client_counts=(4, 16, 64),
        )

        self.assertEqual(
            {row["client_count"] for row in result["scalability"]}, {4, 16, 64}
        )
        self.assertEqual(len(result["scalability"]), 12)
        self.assertEqual(len(result["robustness"]), 16)
        self.assertGreater(result["environment"]["logical_cpu_count"], 0)
        self.assertIn("numpy", result["environment"]["packages"])
        self.assertTrue(
            all(row["mean_aggregation_time_ns"] > 0 for row in result["scalability"])
        )
        self.assertTrue(
            all(row["fit_parameter_bytes"] > 0 for row in result["scalability"])
        )
        self.assertEqual(
            {row["attack"] for row in result["robustness"]},
            {"outlier", "sign_flip"},
        )


if __name__ == "__main__":
    unittest.main()
