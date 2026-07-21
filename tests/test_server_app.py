import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from flwr.common import Context

import src.server_app as server_app


class ServerStartupArtifactCleanupTests(unittest.TestCase):
    def test_server_fn_clears_artifacts_before_creating_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            public_dir = Path(tmpdir) / "public"
            artifact_dir = Path(tmpdir) / "artifacts"
            public_dir.mkdir()
            artifact_dir.mkdir()
            server_artifact_dir = artifact_dir / "server"
            server_artifact_dir.mkdir()
            stale_file = server_artifact_dir / "metrics.csv"
            stale_file.write_text("stale", encoding="utf-8")

            context = cast(
                Context,
                SimpleNamespace(
                    run_config={
                        "public-artifact-dir": public_dir,
                        "server-artifact-dir": artifact_dir,
                        "num-server-rounds": 1,
                        "proximal-mu": 0.25,
                        "use-huber": "true",
                        "huber-threshold": 3.5,
                    }
                ),
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
                components = cast(dict[str, Any], server_app.server_fn(context))

            self.assertTrue(cleanup_saw_stale_file)
            self.assertFalse(stale_file.exists())
            self.assertEqual(
                captured_strategy_kwargs["artifact_dir"], artifact_dir.resolve()
            )
            self.assertEqual(captured_strategy_kwargs["proximal_mu"], 0.25)
            self.assertTrue(captured_strategy_kwargs["use_huber"])
            self.assertEqual(captured_strategy_kwargs["huber_threshold"], 3.5)
            self.assertEqual(components["config"].num_rounds, 1)
            captured_strategy_kwargs["artifact_lock"].release()

    def test_server_fn_refuses_to_clear_public_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            public_dir = Path(tmpdir) / "public"
            public_dir.mkdir()
            manifest = public_dir / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            context = cast(
                Context,
                SimpleNamespace(
                    run_config={
                        "public-artifact-dir": public_dir,
                        "server-artifact-dir": public_dir,
                        "num-server-rounds": 1,
                    }
                ),
            )

            with patch.object(server_app, "create_strategy") as create_strategy:
                with self.assertRaises(ValueError):
                    server_app.server_fn(context)

            create_strategy.assert_not_called()
            self.assertTrue(manifest.exists())

    def test_server_fn_rejects_overlapping_artifact_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            public_dir = Path(tmpdir) / "public"
            artifact_dir = Path(tmpdir) / "server"
            public_dir.mkdir()
            context = cast(
                Context,
                SimpleNamespace(
                    run_config={
                        "public-artifact-dir": public_dir,
                        "server-artifact-dir": artifact_dir,
                        "num-server-rounds": 1,
                    }
                ),
            )
            strategies = []

            def create_strategy_probe(**kwargs):
                strategy = SimpleNamespace(artifact_lock=kwargs["artifact_lock"])
                strategies.append(strategy)
                return strategy

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
                first = cast(dict[str, Any], server_app.server_fn(context))
                with self.assertRaisesRegex(RuntimeError, "already writing"):
                    server_app.server_fn(context)

                first["strategy"].artifact_lock.release()
                second = cast(dict[str, Any], server_app.server_fn(context))
                second["strategy"].artifact_lock.release()

            self.assertEqual(len(strategies), 2)

    def test_server_fn_releases_artifact_lock_after_startup_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            public_dir = Path(tmpdir) / "public"
            artifact_dir = Path(tmpdir) / "server"
            public_dir.mkdir()
            context = cast(
                Context,
                SimpleNamespace(
                    run_config={
                        "public-artifact-dir": public_dir,
                        "server-artifact-dir": artifact_dir,
                        "num-server-rounds": 1,
                    }
                ),
            )

            with patch.object(
                server_app, "create_strategy", side_effect=ValueError("invalid app")
            ):
                with self.assertRaisesRegex(ValueError, "invalid app"):
                    server_app.server_fn(context)

            lock = server_app.acquire_run_artifact_lock(artifact_dir)
            lock.release()


class MetricAggregationTests(unittest.TestCase):
    def test_weighted_average_uses_num_examples(self) -> None:
        metrics = [
            (2, {"accuracy": 0.5}),
            (6, {"accuracy": 1.0}),
        ]

        self.assertEqual(server_app.weighted_average(metrics), {"accuracy": 0.875})

    def test_client_evaluation_metrics_are_written_per_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy = server_app.SentimentServer.__new__(server_app.SentimentServer)
            strategy.artifact_dir = Path(tmpdir)
            results = [
                (
                    Mock(),
                    SimpleNamespace(
                        loss=0.4,
                        num_examples=12,
                        metrics={"accuracy": 0.75, "client_id": 2},
                    ),
                )
            ]

            strategy._write_client_metrics(1, results)

            self.assertEqual(
                strategy.client_metrics_path.read_text(encoding="utf-8").splitlines(),
                [
                    "round,client_id,loss,accuracy,samples",
                    "1,2,0.4,0.75,12",
                ],
            )


if __name__ == "__main__":
    unittest.main()
