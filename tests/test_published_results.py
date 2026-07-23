"""Contracts for the checked-in portfolio experiment results."""

from __future__ import annotations

import json
import math
import statistics
import unittest
from pathlib import Path


class PublishedResultTests(unittest.TestCase):
    def test_portfolio_matrix_is_complete_and_internally_consistent(self) -> None:
        """Require every advertised cell, metric, and confusion matrix."""
        payload = json.loads(
            Path("results/portfolio-matrix.json").read_text(encoding="utf-8")
        )

        self.assertEqual(payload["status"], "complete")
        self.assertEqual(len(payload["cells"]), 60)
        self.assertEqual(len(payload["aggregates"]), 12)
        self.assertEqual(
            {cell["strategy"] for cell in payload["cells"]},
            {"local_only", "fedavg", "fedprox", "fedprox_huber"},
        )
        self.assertEqual(
            {cell["partition"] for cell in payload["cells"]},
            {"iid_stratified", "dirichlet_0.5", "dirichlet_0.1"},
        )
        self.assertEqual(
            {cell["seed"] for cell in payload["cells"]},
            {67, 101, 211, 307, 401},
        )

        for cell in payload["cells"]:
            matrices = (
                cell["confusion_matrix"]
                if cell["strategy"] == "local_only"
                else [cell["confusion_matrix"]]
            )
            for matrix in matrices:
                self.assertEqual(sum(sum(row) for row in matrix), 25_000)
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                self.assertTrue(math.isfinite(cell["metrics"][metric]))
            self.assertRegex(cell["results_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(cell["provenance_sha256"], r"^[0-9a-f]{64}$")

        for aggregate in payload["aggregates"]:
            self.assertEqual(len(aggregate["seeds"]), 5)
            cells = [
                cell
                for cell in payload["cells"]
                if cell["strategy"] == aggregate["strategy"]
                and cell["partition"] == aggregate["partition"]
            ]
            self.assertEqual([cell["seed"] for cell in cells], aggregate["seeds"])
            for name, metric in aggregate["metrics"].items():
                self.assertTrue(math.isfinite(metric["mean"]))
                self.assertGreaterEqual(metric["sample_variance"], 0.0)
                values = [cell["metrics"][name] for cell in cells]
                self.assertAlmostEqual(metric["mean"], statistics.fmean(values))
                self.assertAlmostEqual(
                    metric["sample_variance"],
                    statistics.variance(values),
                )

    def test_fedavg_wins_every_paired_accuracy_comparison(self) -> None:
        """Require the published result behind the main comparative claim."""
        payload = json.loads(
            Path("results/portfolio-matrix.json").read_text(encoding="utf-8")
        )
        cells = {
            (cell["strategy"], cell["partition"], cell["seed"]): cell
            for cell in payload["cells"]
        }

        for partition in ("iid_stratified", "dirichlet_0.5", "dirichlet_0.1"):
            for seed in (67, 101, 211, 307, 401):
                fedavg = cells[("fedavg", partition, seed)]["metrics"]["accuracy"]
                for comparator in ("local_only", "fedprox", "fedprox_huber"):
                    with self.subTest(
                        partition=partition,
                        seed=seed,
                        comparator=comparator,
                    ):
                        self.assertGreater(
                            fedavg,
                            cells[(comparator, partition, seed)]["metrics"]["accuracy"],
                        )


if __name__ == "__main__":
    unittest.main()
