"""Tests for resumable experiment-matrix execution and aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiment_matrix import parse_args, run


def _result(strategy: str, partition: str, seed: int) -> dict[str, object]:
    """Return one valid strategy-runner-shaped test result.

    Parameters
    ----------
    strategy : str
        Cell strategy.
    partition : str
        Cell partition.
    seed : int
        Cell seed and metric source.

    Returns
    -------
    dict of str to object
        Minimal valid raw result.
    """
    metrics = {
        "accuracy": seed / 100,
        "precision": seed / 100,
        "recall": seed / 100,
        "f1": seed / 100,
        "roc_auc": seed / 100,
        "confusion_matrix": [[seed, 0], [0, seed]],
    }
    return {
        "strategy": strategy,
        "config": {"partition": partition, "seed": seed},
        "test_mean" if strategy == "local_only" else "test": metrics,
    }


class ExperimentMatrixTests(unittest.TestCase):
    def test_execution_is_aggregated_canonical_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "matrix"
            args = parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--strategies",
                    "fedavg",
                    "--partitions",
                    "iid_stratified",
                    "--seeds",
                    "67",
                    "101",
                    "--quiet",
                ]
            )

            def execute(runner_args):
                result = _result(
                    runner_args.strategy, runner_args.partition, runner_args.seed
                )
                runner_args.output_dir.mkdir(parents=True)
                (runner_args.output_dir / "results.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                return result

            with patch(
                "src.experiment_matrix.run_strategy", side_effect=execute
            ) as mocked:
                payload = run(args)
                self.assertEqual(mocked.call_count, 2)

            metric = payload["aggregates"][0]["metrics"]["accuracy"]
            self.assertAlmostEqual(metric["mean"], 0.84)
            self.assertAlmostEqual(metric["sample_variance"], 0.0578)
            self.assertEqual(
                payload["cells"][0]["result"]["test"]["confusion_matrix"],
                [[67, 0], [0, 67]],
            )
            canonical = (output_dir / "matrix-results.json").read_text(encoding="utf-8")
            self.assertEqual(json.loads(canonical), payload)
            self.assertIn(
                "| fedavg | iid_stratified | 2 |",
                (output_dir / "SUMMARY.md").read_text(encoding="utf-8"),
            )

            with patch("src.experiment_matrix.run_strategy") as mocked:
                self.assertEqual(run(args), payload)
                mocked.assert_not_called()

    def test_plan_only_uses_registered_order_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(
                [
                    "--output-dir",
                    temporary,
                    "--strategies",
                    "fedprox",
                    "local_only",
                    "--partitions",
                    "dirichlet_0.5",
                    "--seeds",
                    "101",
                    "67",
                    "--plan-only",
                ]
            )
            with patch("src.experiment_matrix.run_strategy") as mocked:
                payload = run(args)
            mocked.assert_not_called()
            self.assertEqual(payload["status"], "planned")
            self.assertEqual(
                [(cell["strategy"], cell["seed"]) for cell in payload["cells"]],
                [
                    ("local_only", 67),
                    ("local_only", 101),
                    ("fedprox", 67),
                    ("fedprox", 101),
                ],
            )


if __name__ == "__main__":
    unittest.main()
