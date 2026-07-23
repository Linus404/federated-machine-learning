"""Contracts for the checked-in portfolio experiment results."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


class PublishedResultTests(unittest.TestCase):
    def test_portfolio_matrix_is_complete_and_internally_consistent(self) -> None:
        """Require every advertised cell, metric, and confusion matrix."""
        payload = json.loads(
            Path("results/portfolio-matrix.json").read_text(encoding="utf-8")
        )

        self.assertEqual(payload["status"], "complete")
        self.assertEqual(len(payload["cells"]), 36)
        self.assertEqual(len(payload["aggregates"]), 12)
        self.assertEqual(
            {cell["strategy"] for cell in payload["cells"]},
            {"local_only", "fedavg", "fedprox", "fedprox_huber"},
        )
        self.assertEqual(
            {cell["partition"] for cell in payload["cells"]},
            {"iid_stratified", "dirichlet_0.5", "dirichlet_0.1"},
        )
        self.assertEqual({cell["seed"] for cell in payload["cells"]}, {67, 101, 211})

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
            self.assertEqual(len(aggregate["seeds"]), 3)
            for metric in aggregate["metrics"].values():
                self.assertTrue(math.isfinite(metric["mean"]))
                self.assertGreaterEqual(metric["sample_variance"], 0.0)


if __name__ == "__main__":
    unittest.main()
