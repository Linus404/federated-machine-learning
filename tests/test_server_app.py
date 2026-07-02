import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import src.server_app as server_app


class ServerStartupArtifactCleanupTests(unittest.TestCase):
    def test_server_fn_clears_artifacts_before_creating_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            artifact_dir = Path(tmpdir) / "artifacts"
            data_dir.mkdir()
            artifact_dir.mkdir()
            stale_file = artifact_dir / "metrics.csv"
            stale_file.write_text("stale", encoding="utf-8")

            context = SimpleNamespace(
                run_config={
                    "data-dir": data_dir,
                    "artifact-dir": artifact_dir,
                    "num-server-rounds": 1,
                    "proximal-mu": 0.25,
                    "use-huber": "true",
                    "huber-threshold": 3.5,
                }
            )
            cleanup_saw_stale_file = False
            captured_strategy_kwargs = {}

            def create_strategy_probe(**kwargs):
                nonlocal cleanup_saw_stale_file
                cleanup_saw_stale_file = not stale_file.exists()
                captured_strategy_kwargs.update(kwargs)
                return Mock(name="strategy")

            with (
                patch.object(
                    server_app, "create_strategy", side_effect=create_strategy_probe
                ),
                patch.object(
                    server_app,
                    "ServerAppComponents",
                    side_effect=lambda **kwargs: kwargs,
                ),
            ):
                components = server_app.server_fn(context)

            self.assertTrue(cleanup_saw_stale_file)
            self.assertFalse(stale_file.exists())
            self.assertEqual(
                captured_strategy_kwargs["artifact_dir"], artifact_dir.resolve()
            )
            self.assertEqual(captured_strategy_kwargs["proximal_mu"], 0.25)
            self.assertTrue(captured_strategy_kwargs["use_huber"])
            self.assertEqual(captured_strategy_kwargs["huber_threshold"], 3.5)
            self.assertEqual(components["config"].num_rounds, 1)

    def test_server_fn_refuses_to_clear_data_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            data_file = data_dir / "partition_0_x.npy"
            data_file.write_text("data", encoding="utf-8")
            context = SimpleNamespace(
                run_config={
                    "data-dir": data_dir,
                    "artifact-dir": data_dir,
                    "num-server-rounds": 1,
                }
            )

            with patch.object(server_app, "create_strategy") as create_strategy:
                with self.assertRaises(ValueError):
                    server_app.server_fn(context)

            create_strategy.assert_not_called()
            self.assertTrue(data_file.exists())


class MetricAggregationTests(unittest.TestCase):
    def test_weighted_average_uses_num_examples(self) -> None:
        metrics = [
            (2, {"accuracy": 0.5}),
            (6, {"accuracy": 1.0}),
        ]

        self.assertEqual(server_app.weighted_average(metrics), {"accuracy": 0.875})


if __name__ == "__main__":
    unittest.main()
