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
from src.app_manifest import expected_train_dataset
from src.artifact_compatibility import (
    SERVER_ARTIFACT_SCHEMA_VERSION,
    load_server_artifact_manifest,
)
from tests.artifact_helpers import fake_app_manifest


def server_app_manifest(public_dir: Path) -> SimpleNamespace:
    """Return a public snapshot suitable for server startup provenance.

    Parameters
    ----------
    public_dir : pathlib.Path
        Public directory used for diagnostic vocabulary identity.

    Returns
    -------
    types.SimpleNamespace
        App-manifest-shaped immutable snapshot.
    """
    snapshot = fake_app_manifest()
    dataset = expected_train_dataset()
    snapshot.payload["dataset"] = dataset
    snapshot.manifest_bytes = json.dumps({"dataset": dataset}).encode("utf-8")
    snapshot.vocabulary_path = public_dir / "vocab.txt"
    return snapshot


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
                        "expected-client-count": 16,
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
                    server_app,
                    "load_app_manifest",
                    return_value=server_app_manifest(public_dir),
                ),
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
            self.assertEqual(captured_strategy_kwargs["min_clients"], 16)
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

    def test_server_fn_threads_one_public_snapshot_through_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            public_dir = root / "public"
            public_dir.mkdir()
            artifact_dir = root / "server"
            snapshot = server_app_manifest(public_dir)
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
            strategy = SimpleNamespace()
            run_dir = artifact_dir / "runs" / "run"
            with (
                patch.object(
                    server_app,
                    "load_app_manifest",
                    side_effect=(snapshot, AssertionError("pointer reopened")),
                ) as load_manifest,
                patch.object(
                    server_app,
                    "create_run_artifact_dir",
                    return_value=run_dir,
                ) as create_run,
                patch.object(server_app, "prune_run_history"),
                patch.object(
                    server_app, "create_strategy", return_value=strategy
                ) as create_strategy,
                patch.object(
                    server_app,
                    "ServerAppComponents",
                    side_effect=lambda **kwargs: kwargs,
                ),
            ):
                result = cast(dict[str, Any], server_app.server_fn(context))

            load_manifest.assert_called_once_with(public_artifact_dir=public_dir)
            self.assertIs(create_run.call_args.kwargs["app_manifest"], snapshot)
            self.assertIs(create_strategy.call_args.kwargs["app_manifest"], snapshot)
            self.assertIs(result["strategy"], strategy)
            create_strategy.call_args.kwargs["artifact_lock"].release()

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
                    server_app,
                    "load_app_manifest",
                    return_value=server_app_manifest(public_dir),
                ),
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

            with (
                patch.object(
                    server_app,
                    "load_app_manifest",
                    return_value=server_app_manifest(public_dir),
                ),
                patch.object(
                    server_app,
                    "create_strategy",
                    side_effect=ValueError("invalid app"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "invalid app"):
                    server_app.server_fn(context)

            lock = server_app.acquire_run_artifact_lock(artifact_dir)
            lock.release()

    def test_server_fn_rejects_invalid_expected_client_count(self) -> None:
        for value in (0, -1, True, 4.5, "4"):
            with self.subTest(value=value):
                context = cast(
                    Context,
                    SimpleNamespace(
                        run_config={
                            "expected-client-count": value,
                            "num-server-rounds": 1,
                        }
                    ),
                )

                with self.assertRaisesRegex(ValueError, "positive built-in integer"):
                    server_app.server_fn(context)


class MetricAggregationTests(unittest.TestCase):
    def test_create_strategy_configures_exact_clients_and_failure_rejection(
        self,
    ) -> None:
        model = Mock()
        model.get_weights.return_value = [np.zeros(2, dtype=np.float32)]
        strategy = SimpleNamespace()

        with (
            patch.object(server_app, "load_app_manifest", return_value=object()),
            patch.object(server_app, "build_model_from_manifest", return_value=model),
            patch.object(
                server_app, "SentimentServer", return_value=strategy
            ) as server_type,
        ):
            actual = server_app.create_strategy(min_clients=16)

        self.assertIs(actual, strategy)
        self.assertEqual(strategy.expected_client_ids, frozenset(range(16)))
        self.assertEqual(strategy.expected_weight_shapes, ((2,),))
        strategy_kwargs = server_type.call_args.kwargs
        self.assertFalse(strategy_kwargs["accept_failures"])
        self.assertEqual(strategy_kwargs["min_fit_clients"], 16)
        self.assertEqual(strategy_kwargs["min_available_clients"], 16)

    def test_strategy_publishes_server_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(server_app.FedProx, "__init__", return_value=None):
                server_app.SentimentServer(
                    app_manifest=fake_app_manifest(), artifact_dir=Path(tmpdir)
                )

            self.assertEqual(
                load_server_artifact_manifest(Path(tmpdir))["schema_version"],
                SERVER_ARTIFACT_SCHEMA_VERSION,
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
            strategy.accept_failures = False
            strategy.inplace = True
            strategy.fit_metrics_aggregation_fn = None
            strategy.expected_client_ids = frozenset(range(3))
            strategy.expected_weight_shapes = ((1,),)
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
            strategy.fit_metrics_aggregation_fn = None
            strategy.huber_threshold = 10.0
            strategy.expected_client_ids = frozenset(range(3))
            strategy.expected_weight_shapes = ((1,),)
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
                    return_value=np.array([2.0], dtype=np.float32),
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

    def test_fit_aggregation_rejects_incomplete_duplicate_or_out_of_range_ids(
        self,
    ) -> None:
        missing = [(Mock(), SimpleNamespace(metrics={}))]
        duplicate = [
            (Mock(), SimpleNamespace(metrics={"client_id": 1})),
            (Mock(), SimpleNamespace(metrics={"client_id": 1})),
        ]
        incomplete = [(Mock(), SimpleNamespace(metrics={"client_id": 0}))]
        out_of_range = [
            (Mock(), SimpleNamespace(metrics={"client_id": 0})),
            (Mock(), SimpleNamespace(metrics={"client_id": 2})),
        ]
        non_builtin = [
            (Mock(), SimpleNamespace(metrics={"client_id": np.int64(0)})),
            (Mock(), SimpleNamespace(metrics={"client_id": 1})),
        ]

        for results in (missing, duplicate, incomplete, out_of_range, non_builtin):
            with self.subTest(results=results):
                with self.assertRaises(ValueError):
                    server_app._sorted_fit_results(results, frozenset({0, 1}))

        with self.assertRaisesRegex(ValueError, "contiguous from zero"):
            server_app._sorted_fit_results(out_of_range, frozenset({0, 2}))

    def test_fit_failure_hard_fails_before_client_id_validation(self) -> None:
        invalid_results = [(Mock(), SimpleNamespace(metrics={}))]

        for use_huber in (False, True):
            with self.subTest(use_huber=use_huber):
                strategy = server_app.SentimentServer.__new__(
                    server_app.SentimentServer
                )
                strategy.use_huber = use_huber

                with self.assertRaisesRegex(RuntimeError, "failed for 1 client"):
                    strategy.aggregate_fit(1, invalid_results, [RuntimeError("failed")])

    def test_huber_fit_aggregation_rejects_mismatched_model_shapes(self) -> None:
        strategy = server_app.SentimentServer.__new__(server_app.SentimentServer)
        strategy.use_huber = True
        strategy.huber_threshold = 1.0
        strategy.expected_client_ids = frozenset({0, 1})
        strategy.expected_weight_shapes = ((2,),)
        results = [
            (
                Mock(),
                SimpleNamespace(
                    parameters=ndarrays_to_parameters(weights),
                    num_examples=1,
                    metrics={"client_id": client_id},
                ),
            )
            for client_id, weights in enumerate(
                (
                    [np.array([1.0, 2.0], dtype=np.float32)],
                    [np.array([[1.0, 2.0]], dtype=np.float32)],
                )
            )
        ]

        with self.assertRaisesRegex(ValueError, "shapes must match round start"):
            strategy.aggregate_fit(1, results, [])

    def test_fit_validation_rejects_hostile_sample_counts(self) -> None:
        weights = [np.array([1.0], dtype=np.float32)]

        for count in (0, -1, True, np.int64(1)):
            with self.subTest(count=count):
                result = (
                    Mock(),
                    SimpleNamespace(
                        parameters=ndarrays_to_parameters(weights),
                        num_examples=count,
                        metrics={"client_id": 0},
                    ),
                )
                with self.assertRaisesRegex(ValueError, "positive built-in integer"):
                    server_app._validate_fit_results([result], frozenset({0}), ((1,),))

    def test_fit_validation_rejects_hostile_model_tensors(self) -> None:
        invalid_tensors = [
            np.array([1.0], dtype=np.float64),
            np.array([1], dtype=np.int64),
            np.array([np.nan], dtype=np.float32),
            np.array([np.inf], dtype=np.float32),
            np.array(1.0, dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([[1.0]], dtype=np.float32),
        ]

        for tensor in invalid_tensors:
            with self.subTest(dtype=tensor.dtype, shape=tensor.shape):
                result = (
                    Mock(),
                    SimpleNamespace(
                        parameters=ndarrays_to_parameters([tensor]),
                        num_examples=1,
                        metrics={"client_id": 0},
                    ),
                )
                with self.assertRaises(ValueError):
                    server_app._validate_fit_results([result], frozenset({0}), ((1,),))

    def test_plain_aggregation_rejects_missing_aggregate(self) -> None:
        strategy = server_app.SentimentServer.__new__(server_app.SentimentServer)
        strategy.use_huber = False
        strategy.expected_client_ids = frozenset({0})
        strategy.expected_weight_shapes = ((1,),)
        result = (
            Mock(),
            SimpleNamespace(
                parameters=ndarrays_to_parameters([np.array([1.0], dtype=np.float32)]),
                num_examples=1,
                metrics={"client_id": 0},
            ),
        )

        with patch.object(server_app.FedProx, "aggregate_fit", return_value=(None, {})):
            with self.assertRaisesRegex(RuntimeError, "produced no aggregate"):
                strategy.aggregate_fit(1, [result], [])

    def test_plain_aggregation_rejects_invalid_strategy_output(self) -> None:
        strategy = server_app.SentimentServer.__new__(server_app.SentimentServer)
        strategy.use_huber = False
        strategy.expected_client_ids = frozenset({0})
        strategy.expected_weight_shapes = ((1,),)
        result = (
            Mock(),
            SimpleNamespace(
                parameters=ndarrays_to_parameters([np.array([1.0], dtype=np.float32)]),
                num_examples=1,
                metrics={"client_id": 0},
            ),
        )
        invalid_aggregate = ndarrays_to_parameters(
            [np.array([np.nan], dtype=np.float32)]
        )

        with patch.object(
            server_app.FedProx,
            "aggregate_fit",
            return_value=(invalid_aggregate, {}),
        ):
            with self.assertRaisesRegex(ValueError, "finite values"):
                strategy.aggregate_fit(1, [result], [])

    def test_terminal_fit_failures_release_writer_without_publication(self) -> None:
        valid_parameters = ndarrays_to_parameters([np.array([1.0], dtype=np.float32)])
        valid_result = (
            Mock(),
            SimpleNamespace(
                parameters=valid_parameters,
                num_examples=1,
                metrics={"client_id": 0},
            ),
        )
        cases = {
            "reported_client_failure": (
                [],
                [RuntimeError("client failed")],
                None,
                None,
            ),
            "malformed_result": (
                [(Mock(), SimpleNamespace(metrics=None))],
                [],
                None,
                None,
            ),
            "aggregation_error": (
                [valid_result],
                [],
                RuntimeError("aggregation failed"),
                None,
            ),
            "persistence_error": (
                [valid_result],
                [],
                None,
                OSError("model save failed"),
            ),
        }

        for name, (
            results,
            failures,
            aggregation_error,
            persistence_error,
        ) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                run_dir = root / "runs" / "11111111-1111-4111-8111-111111111111"
                strategy = server_app.SentimentServer.__new__(
                    server_app.SentimentServer
                )
                strategy.use_huber = False
                strategy.accept_failures = False
                strategy.inplace = True
                strategy.fit_metrics_aggregation_fn = None
                strategy.expected_client_ids = frozenset({0})
                strategy.expected_weight_shapes = ((1,),)
                strategy.artifact_dir = run_dir
                strategy.artifact_root = root
                strategy.app_manifest = object()
                strategy._artifact_lock = server_app.acquire_run_artifact_lock(run_dir)
                model = Mock()
                model.save.side_effect = persistence_error

                with (
                    patch.object(
                        server_app.FedProx,
                        "aggregate_fit",
                        return_value=(valid_parameters, {}),
                        side_effect=aggregation_error,
                    ),
                    patch.object(
                        server_app, "build_model_from_manifest", return_value=model
                    ),
                    patch.object(server_app, "publish_completed_run") as publish,
                    self.assertRaises((RuntimeError, ValueError, OSError)),
                ):
                    strategy.aggregate_fit(1, results, failures)

                publish.assert_not_called()
                self.assertFalse((root / "current.json").exists())
                replacement_lock = server_app.acquire_run_artifact_lock(run_dir)
                replacement_lock.release()

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
            strategy.app_manifest = fake_app_manifest()
            artifact_lock = Mock()
            strategy._artifact_lock = artifact_lock
            strategy.expected_client_ids = frozenset({0, 1})
            strategy.accept_failures = False
            strategy.evaluate_metrics_aggregation_fn = server_app.weighted_average
            results = [
                (
                    Mock(),
                    SimpleNamespace(
                        loss=loss,
                        num_examples=count,
                        metrics={"accuracy": accuracy, "client_id": client_id},
                    ),
                )
                for client_id, loss, accuracy, count in [
                    (1, 0.7, 0.5, 3),
                    (0, 0.2, 1.0, 1),
                ]
            ]

            with (
                patch.object(server_app, "publish_completed_run") as publish,
                patch.object(server_app, "prune_run_history") as prune,
            ):
                loss, metrics = strategy.aggregate_evaluate(2, results, [])

            self.assertAlmostEqual(loss, 0.575)
            self.assertEqual(metrics, {"accuracy": 0.625})
            publish.assert_called_once_with(
                root, run_dir, app_manifest=strategy.app_manifest
            )
            prune.assert_called_once_with(root, 3, active_run_dir=run_dir)
            artifact_lock.release.assert_called_once_with()

    def test_final_publication_failures_release_real_writer_lock(self) -> None:
        for error in (OSError("publication failed"), KeyboardInterrupt()):
            with (
                self.subTest(error=type(error).__name__),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                run_dir = root / "runs" / "11111111-1111-4111-8111-111111111111"
                strategy = server_app.SentimentServer.__new__(
                    server_app.SentimentServer
                )
                strategy.artifact_dir = run_dir
                strategy.artifact_root = root
                strategy.artifact_retention_runs = 3
                strategy.final_round = 1
                strategy.app_manifest = fake_app_manifest()
                artifact_lock = server_app.acquire_run_artifact_lock(run_dir)
                strategy._artifact_lock = artifact_lock
                strategy.expected_client_ids = frozenset({0})
                strategy.accept_failures = False
                strategy.evaluate_metrics_aggregation_fn = server_app.weighted_average
                result = (
                    Mock(),
                    SimpleNamespace(
                        loss=0.4,
                        num_examples=2,
                        metrics={"accuracy": 0.75, "client_id": 0},
                    ),
                )

                with (
                    patch.object(
                        artifact_lock, "release", wraps=artifact_lock.release
                    ) as release,
                    patch.object(
                        server_app, "publish_completed_run", side_effect=error
                    ),
                    patch.object(server_app, "prune_run_history") as prune,
                    self.assertRaises(type(error)) as raised,
                ):
                    strategy.aggregate_evaluate(1, [result], [])

                self.assertIs(raised.exception, error)
                release.assert_called_once_with()
                prune.assert_not_called()
                self.assertFalse((root / "current.json").exists())
                replacement_lock = server_app.acquire_run_artifact_lock(run_dir)
                replacement_lock.release()

    def test_final_evaluation_rejects_hostile_results_without_publication(
        self,
    ) -> None:
        def valid(client_id: int) -> tuple[Mock, SimpleNamespace]:
            """Build one valid evaluation result.

            Parameters
            ----------
            client_id : int
                Zero-based client identifier.

            Returns
            -------
            tuple[unittest.mock.Mock, types.SimpleNamespace]
                Flower-like client proxy and evaluation result.
            """
            return (
                Mock(),
                SimpleNamespace(
                    loss=0.4,
                    num_examples=2,
                    metrics={"accuracy": 0.75, "client_id": client_id},
                ),
            )

        hostile_cases = {
            "failure": ([valid(0), valid(1)], [RuntimeError("client failed")]),
            "empty": ([], []),
            "partial": ([valid(0)], []),
            "duplicate": ([valid(0), valid(0)], []),
            "unexpected": ([valid(0), valid(2)], []),
            "malformed": ([(Mock(), SimpleNamespace(metrics=None))], []),
            "missing_values": (
                [
                    (
                        Mock(),
                        SimpleNamespace(metrics={"accuracy": 0.75, "client_id": 0}),
                    ),
                    valid(1),
                ],
                [],
            ),
        }

        for name, (results, failures) in hostile_cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                strategy = server_app.SentimentServer.__new__(
                    server_app.SentimentServer
                )
                strategy.artifact_dir = Path(tmpdir)
                strategy.artifact_root = Path(tmpdir)
                strategy.artifact_retention_runs = 3
                strategy.final_round = 2
                artifact_lock = Mock()
                strategy._artifact_lock = artifact_lock
                strategy.expected_client_ids = frozenset({0, 1})

                with (
                    patch.object(server_app, "publish_completed_run") as publish,
                    patch.object(server_app, "prune_run_history") as prune,
                ):
                    with self.assertRaises((RuntimeError, ValueError)):
                        strategy.aggregate_evaluate(2, results, failures)

                publish.assert_not_called()
                prune.assert_not_called()
                artifact_lock.release.assert_called_once_with()
                self.assertFalse(strategy.metrics_path.exists())
                self.assertFalse(strategy.client_metrics_path.exists())

    def test_final_evaluation_rejects_invalid_aggregate_without_publication(
        self,
    ) -> None:
        invalid_aggregates = [
            (np.nan, {"accuracy": 0.75}),
            (-0.1, {"accuracy": 0.75}),
            (0.4, {"accuracy": -0.1}),
            (0.4, {"accuracy": 1.1}),
        ]

        for aggregate in invalid_aggregates:
            with (
                self.subTest(aggregate=aggregate),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                strategy = server_app.SentimentServer.__new__(
                    server_app.SentimentServer
                )
                strategy.artifact_dir = Path(tmpdir)
                strategy.artifact_root = Path(tmpdir)
                strategy.artifact_retention_runs = 3
                strategy.final_round = 1
                artifact_lock = Mock()
                strategy._artifact_lock = artifact_lock
                strategy.expected_client_ids = frozenset({0})
                result = (
                    Mock(),
                    SimpleNamespace(
                        loss=0.4,
                        num_examples=2,
                        metrics={"accuracy": 0.75, "client_id": 0},
                    ),
                )

                with (
                    patch.object(
                        server_app.FedProx,
                        "aggregate_evaluate",
                        return_value=aggregate,
                    ),
                    patch.object(server_app, "publish_completed_run") as publish,
                    self.assertRaises(ValueError),
                ):
                    strategy.aggregate_evaluate(1, [result], [])

                publish.assert_not_called()
                artifact_lock.release.assert_called_once_with()
                self.assertFalse(strategy.metrics_path.exists())
                self.assertFalse(strategy.client_metrics_path.exists())

    def test_final_evaluation_rejects_invalid_values_without_publication(self) -> None:
        invalid_values = [
            ("num_examples", 0),
            ("num_examples", -1),
            ("num_examples", True),
            ("num_examples", np.int64(1)),
            ("loss", np.nan),
            ("loss", np.inf),
            ("loss", -0.1),
            ("accuracy", None),
            ("accuracy", np.nan),
            ("accuracy", np.inf),
            ("accuracy", -0.1),
            ("accuracy", 1.1),
        ]

        for field, value in invalid_values:
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                evaluation = {
                    "loss": 0.4,
                    "num_examples": 2,
                    "metrics": {"accuracy": 0.75, "client_id": 0},
                }
                if field == "accuracy":
                    evaluation["metrics"]["accuracy"] = value
                else:
                    evaluation[field] = value
                strategy = server_app.SentimentServer.__new__(
                    server_app.SentimentServer
                )
                strategy.artifact_dir = Path(tmpdir)
                strategy.artifact_root = Path(tmpdir)
                strategy.artifact_retention_runs = 3
                strategy.final_round = 1
                artifact_lock = Mock()
                strategy._artifact_lock = artifact_lock
                strategy.expected_client_ids = frozenset({0})

                with (
                    patch.object(server_app, "publish_completed_run") as publish,
                    self.assertRaises(ValueError),
                ):
                    strategy.aggregate_evaluate(
                        1, [(Mock(), SimpleNamespace(**evaluation))], []
                    )

                publish.assert_not_called()
                artifact_lock.release.assert_called_once_with()
                self.assertFalse(strategy.metrics_path.exists())
                self.assertFalse(strategy.client_metrics_path.exists())


if __name__ == "__main__":
    unittest.main()
