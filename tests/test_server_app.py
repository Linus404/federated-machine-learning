import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays

import src.server_app as server_app
from src.artifact_compatibility import load_server_artifact_manifest


class ServerStartupArtifactHistoryTests(unittest.TestCase):
    def test_server_fn_creates_a_new_run_without_overwriting_history(self) -> None:
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
            captured_strategy_kwargs = {}

            def create_strategy_probe(**kwargs):
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

            self.assertTrue(stale_file.exists())
            run_dir = captured_strategy_kwargs["artifact_dir"]
            self.assertEqual(run_dir.parent, artifact_dir.resolve() / "runs")
            self.assertEqual(captured_strategy_kwargs["proximal_mu"], 0.25)
            self.assertTrue(captured_strategy_kwargs["use_huber"])
            self.assertEqual(captured_strategy_kwargs["huber_threshold"], 3.5)
            self.assertEqual(components["config"].num_rounds, 1)
            provenance = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["run_id"], run_dir.name)
            self.assertEqual(provenance["run_config"]["num-server-rounds"], 1)
            self.assertEqual(provenance["schema_version"], 1)
            captured_strategy_kwargs["artifact_lock"].release()

    def test_server_fn_refuses_overlapping_public_artifacts(self) -> None:
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
    def test_strategy_publishes_server_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(server_app.FedProx, "__init__", return_value=None):
                server_app.SentimentServer(
                    app_manifest=object(), artifact_dir=Path(tmpdir)
                )

            self.assertEqual(
                load_server_artifact_manifest(Path(tmpdir))["schema_version"], 1
            )

    def test_weighted_average_uses_num_examples(self) -> None:
        metrics = [
            (2, {"accuracy": 0.5}),
            (6, {"accuracy": 1.0}),
        ]

        self.assertEqual(server_app.weighted_average(metrics), {"accuracy": 0.875})

    def test_plain_fit_aggregation_uses_ascending_client_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy = server_app.SentimentServer.__new__(server_app.SentimentServer)
            strategy.use_huber = False
            strategy.accept_failures = True
            strategy.inplace = True
            strategy.fit_metrics_aggregation_fn = None
            strategy.artifact_dir = Path(tmpdir)
            strategy.app_manifest = object()
            results = [
                (
                    Mock(),
                    SimpleNamespace(
                        parameters=ndarrays_to_parameters(
                            [np.array([value], dtype=np.float32)]
                        ),
                        num_examples=1,
                        metrics={"client_id": client_id},
                    ),
                )
                for client_id, value in [(2, 1.0), (1, -1e20), (0, 1e20)]
            ]
            model = Mock()

            with patch.object(
                server_app, "build_model_from_manifest", return_value=model
            ):
                parameters, _ = strategy.aggregate_fit(1, results, [])

            self.assertIsNotNone(parameters)
            np.testing.assert_array_equal(
                parameters_to_ndarrays(parameters)[0],
                np.array([np.float32(1.0 / 3.0)], dtype=np.float32),
            )

    def test_huber_fit_aggregation_uses_ascending_client_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            strategy = server_app.SentimentServer.__new__(server_app.SentimentServer)
            strategy.use_huber = True
            strategy.accept_failures = True
            strategy.fit_metrics_aggregation_fn = None
            strategy.huber_threshold = 10.0
            strategy.artifact_dir = Path(tmpdir)
            strategy.app_manifest = object()
            results = [
                (
                    Mock(),
                    SimpleNamespace(
                        parameters=ndarrays_to_parameters(
                            [np.array([value], dtype=np.float32)]
                        ),
                        num_examples=count,
                        metrics={"client_id": client_id},
                    ),
                )
                for client_id, value, count in [
                    (2, 3.0, 30),
                    (0, 1.0, 10),
                    (1, 2.0, 20),
                ]
            ]
            model = Mock()

            with (
                patch.object(
                    server_app,
                    "huber_aggregate",
                    return_value=np.array([2.0], dtype=np.float64),
                ) as aggregate,
                patch.object(
                    server_app, "build_model_from_manifest", return_value=model
                ),
            ):
                strategy.aggregate_fit(1, results, [])

            vectors, counts, threshold = aggregate.call_args.args
            np.testing.assert_array_equal(
                np.stack(vectors), np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
            )
            self.assertEqual(counts, [10, 20, 30])
            self.assertEqual(threshold, 10.0)

    def test_fit_aggregation_rejects_missing_or_duplicate_client_ids(self) -> None:
        missing = [(Mock(), SimpleNamespace(metrics={}))]
        duplicate = [
            (Mock(), SimpleNamespace(metrics={"client_id": 1})),
            (Mock(), SimpleNamespace(metrics={"client_id": 1})),
        ]

        for results in (missing, duplicate):
            with self.subTest(results=results):
                with self.assertRaises(ValueError):
                    server_app._sorted_fit_results(results)

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

    def test_final_evaluation_publishes_run_and_releases_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "11111111-1111-4111-8111-111111111111"
            run_dir.mkdir(parents=True)
            (run_dir / "global_model.keras").write_bytes(b"model")
            strategy = server_app.SentimentServer.__new__(server_app.SentimentServer)
            strategy.artifact_dir = run_dir
            strategy.artifact_root = root
            strategy.artifact_retention_runs = 3
            strategy.final_round = 2
            strategy._artifact_lock = Mock()

            with (
                patch.object(
                    server_app.FedProx,
                    "aggregate_evaluate",
                    return_value=(0.5, {"accuracy": 0.75}),
                ),
                patch.object(server_app, "publish_completed_run") as publish,
                patch.object(server_app, "prune_run_history") as prune,
            ):
                strategy.aggregate_evaluate(2, [], [])

            publish.assert_called_once_with(root, run_dir)
            prune.assert_called_once_with(root, 3, active_run_dir=run_dir)
            strategy._artifact_lock.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
