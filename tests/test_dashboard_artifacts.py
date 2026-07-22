import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import dashboard
import numpy as np
from src.artifact_compatibility import (
    SERVER_ARTIFACT_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_bytes,
    write_server_artifact_manifest,
)
from tests.artifact_helpers import compatible_model, fake_app_manifest
from tests.test_artifact_history import (
    RUN_IDS,
    _TEST_PROTOCOL,
    create_public_snapshot,
    create_run,
    publish_completed_run,
)


def write_server_manifest(path: Path, version: int | None) -> None:
    """Write a server manifest for one compatibility test.

    Parameters
    ----------
    path : pathlib.Path
        Server artifact directory.
    version : int or None
        Schema version, or ``None`` to omit it.

    Returns
    -------
    None
    """
    manifest_path = write_server_artifact_manifest(
        path, app_manifest=fake_app_manifest()
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = version
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


class DashboardArtifactTests(unittest.TestCase):
    def test_metrics_loader_treats_only_a_fresh_root_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(dashboard, "ARTIFACT_ROOT", root):
                self.assertIsNone(dashboard.load_metrics())

            (root / "metrics.csv").write_text(
                "malformed legacy state", encoding="utf-8"
            )
            with (
                patch.object(dashboard, "ARTIFACT_ROOT", root),
                patch.object(
                    dashboard,
                    "load_public_snapshot",
                    return_value=fake_app_manifest(),
                ),
                self.assertRaisesRegex(ValueError, "no valid schema_version"),
            ):
                dashboard.load_metrics()

            (root / "metrics.csv").unlink()
            (root / "current.json").write_text("not-json", encoding="utf-8")
            with (
                patch.object(dashboard, "ARTIFACT_ROOT", root),
                patch.object(
                    dashboard,
                    "load_public_snapshot",
                    return_value=fake_app_manifest(),
                ),
                self.assertRaisesRegex(ValueError, "invalid current-run index"),
            ):
                dashboard.load_metrics()

    def test_model_loader_checks_each_schema_version_state(self) -> None:
        for version, error in (
            (None, "no valid"),
            (SERVER_ARTIFACT_SCHEMA_VERSION - 1, "older"),
            (SERVER_ARTIFACT_SCHEMA_VERSION + 1, "newer"),
        ):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                model_path = path / "global_model.keras"
                model_path.touch()
                write_server_manifest(path, version)

                with (
                    patch.object(
                        dashboard,
                        "load_public_snapshot",
                        return_value=fake_app_manifest(),
                    ),
                    patch.object(dashboard.keras.models, "load_model") as load_model,
                    self.assertRaisesRegex(ValueError, error),
                ):
                    dashboard.load_model(model_path)
                load_model.assert_not_called()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model_path = path / "global_model.keras"
            model_path.touch()
            write_server_manifest(path, SERVER_ARTIFACT_SCHEMA_VERSION)
            sentinel = compatible_model()
            with (
                patch.object(
                    dashboard,
                    "load_public_snapshot",
                    return_value=fake_app_manifest(),
                ),
                patch.object(
                    dashboard.keras.models, "load_model", return_value=sentinel
                ),
            ):
                self.assertIs(dashboard.load_model(model_path), sentinel)

    def test_metrics_loader_checks_each_schema_version_state(self) -> None:
        for version, error in (
            (None, "no valid"),
            (SERVER_ARTIFACT_SCHEMA_VERSION - 1, "older"),
            (SERVER_ARTIFACT_SCHEMA_VERSION + 1, "newer"),
        ):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                metrics_path = path / "metrics.csv"
                metrics_path.write_text(
                    "round,loss,accuracy\n1,0.5,0.75\n", encoding="utf-8"
                )
                write_server_manifest(path, version)

                with (
                    patch.object(
                        dashboard,
                        "load_public_snapshot",
                        return_value=fake_app_manifest(),
                    ),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    dashboard.load_metrics(metrics_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            metrics_path = path / "metrics.csv"
            metrics_path.write_text(
                "round,loss,accuracy\n1,0.5,0.75\n", encoding="utf-8"
            )
            write_server_manifest(path, SERVER_ARTIFACT_SCHEMA_VERSION)

            with patch.object(
                dashboard,
                "load_public_snapshot",
                return_value=fake_app_manifest(),
            ):
                metrics = dashboard.load_metrics(metrics_path)

            self.assertIsNotNone(metrics)
            assert metrics is not None
            self.assertEqual(
                metrics.to_dict("records"),
                [
                    {
                        "round": 1,
                        "loss": 0.5,
                        "accuracy": 0.75,
                    }
                ],
            )

    def test_model_loader_uses_verified_snapshot_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model_path = path / "global_model.keras"
            model_path.write_bytes(b"verified-model")
            write_server_manifest(path, SERVER_ARTIFACT_SCHEMA_VERSION)

            def inspect_snapshot(loaded_path: Path) -> object:
                model_path.write_bytes(b"attacker-controlled")
                self.assertEqual(Path(loaded_path).read_bytes(), b"verified-model")
                return compatible_model()

            with (
                patch.object(
                    dashboard,
                    "load_public_snapshot",
                    return_value=fake_app_manifest(),
                ),
                patch.object(
                    dashboard.keras.models,
                    "load_model",
                    side_effect=inspect_snapshot,
                ),
            ):
                dashboard.load_model(model_path)

    def test_model_loader_rejects_public_binding_mismatch_before_keras(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model_path = path / "global_model.keras"
            model_path.write_bytes(b"model")
            write_server_manifest(path, SERVER_ARTIFACT_SCHEMA_VERSION)
            incompatible = fake_app_manifest()
            incompatible.manifest_bytes = b"different public manifest\n"

            with (
                patch.object(
                    dashboard, "load_public_snapshot", return_value=incompatible
                ),
                patch.object(dashboard.keras.models, "load_model") as load_model,
                self.assertRaisesRegex(ValueError, "does not match"),
            ):
                dashboard.load_model(model_path)

            load_model.assert_not_called()

    def test_model_loader_rejects_serialized_model_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            model_path = path / "global_model.keras"
            model_path.write_bytes(b"model")
            write_server_manifest(path, SERVER_ARTIFACT_SCHEMA_VERSION)
            incompatible_model = compatible_model()
            incompatible_model.input_shape = (None, 499)

            with (
                patch.object(
                    dashboard,
                    "load_public_snapshot",
                    return_value=fake_app_manifest(),
                ),
                patch.object(
                    dashboard.keras.models,
                    "load_model",
                    return_value=incompatible_model,
                ),
                self.assertRaisesRegex(ValueError, "model dimensions"),
            ):
                dashboard.load_model(model_path)

    def test_model_loader_rejects_runtime_mismatch_before_artifact_consumption(
        self,
    ) -> None:
        with (
            patch.object(np, "__version__", "99.0.0"),
            patch.object(dashboard, "load_server_artifact_snapshot") as load_snapshot,
            patch.object(dashboard.keras.models, "load_model") as load_model,
            self.assertRaisesRegex(ValueError, "framework versions differ"),
        ):
            dashboard.load_model(Path("unused/global_model.keras"))

        load_snapshot.assert_not_called()
        load_model.assert_not_called()

    def test_both_metrics_files_reject_mismatched_public_for_current_and_explicit_paths(
        self,
    ) -> None:
        for filename in ("metrics.csv", "client_metrics.csv"):
            for explicit in (False, True):
                with (
                    self.subTest(filename=filename, explicit=explicit),
                    tempfile.TemporaryDirectory() as tmpdir,
                ):
                    root = Path(tmpdir)
                    (root / filename).write_text(
                        "round,loss,accuracy\n1,0.5,0.75\n", encoding="utf-8"
                    )
                    write_server_manifest(root, SERVER_ARTIFACT_SCHEMA_VERSION)
                    incompatible = fake_app_manifest()
                    incompatible.manifest_bytes = b"different public manifest\n"
                    path = root / filename if explicit else None

                    with (
                        patch.object(dashboard, "ARTIFACT_ROOT", root),
                        patch.object(
                            dashboard,
                            "load_public_snapshot",
                            return_value=incompatible,
                        ),
                        self.assertRaisesRegex(ValueError, "does not match"),
                    ):
                        dashboard.load_metrics(path, filename=filename)

    def test_explicit_historical_paths_reject_rechecksummed_wrong_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            public = create_public_snapshot(root, "public")
            run = create_run(
                root,
                RUN_IDS[0],
                "2026-01-01T00:00:00Z",
                provenance_app_manifest=public,
                artifact_app_manifest=public,
            )
            with patch(
                "src.app_manifest.load_scientific_protocol",
                return_value=_TEST_PROTOCOL,
            ):
                publish_completed_run(root, run)
            provenance_path = run / "run_manifest.json"
            provenance = json.loads(provenance_path.read_bytes())
            provenance["run_id"] = RUN_IDS[1]
            provenance_bytes = canonical_json_bytes(provenance)
            provenance_path.write_bytes(provenance_bytes)
            artifact_path = run / "artifact_manifest.json"
            artifact = json.loads(artifact_path.read_bytes())
            artifact["sizes"]["run_manifest.json"] = len(provenance_bytes)
            artifact["checksums"]["run_manifest.json"] = sha256_bytes(provenance_bytes)
            artifact_path.write_bytes(canonical_json_bytes(artifact))

            with patch(
                "src.app_manifest.load_scientific_protocol",
                return_value=_TEST_PROTOCOL,
            ):
                for loader, path in (
                    (dashboard.load_model, run / "global_model.keras"),
                    (dashboard.load_metrics, run / "metrics.csv"),
                ):
                    with (
                        self.subTest(loader=loader.__name__),
                        self.assertRaisesRegex(ValueError, "run_id"),
                    ):
                        loader(path, app_manifest=public)

    def test_public_snapshot_cache_is_keyed_by_immutable_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first" / "public"
            second = Path(tmpdir) / "second" / "public"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            first_snapshot = fake_app_manifest()
            second_snapshot = fake_app_manifest()
            dashboard._load_public_snapshot_at.clear()
            with (
                patch.object(
                    dashboard,
                    "resolve_public_artifact_dir",
                    side_effect=(first, first, second),
                ),
                patch.object(
                    dashboard,
                    "load_app_manifest",
                    side_effect=(first_snapshot, second_snapshot),
                ) as load_manifest,
            ):
                self.assertIs(dashboard.load_public_snapshot(), first_snapshot)
                self.assertIs(dashboard.load_public_snapshot(), first_snapshot)
                self.assertIs(dashboard.load_public_snapshot(), second_snapshot)

            dashboard._load_public_snapshot_at.clear()
            self.assertEqual(load_manifest.call_count, 2)

    def test_no_arg_vectorizer_changes_with_the_selected_public_generation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first" / "public"
            second = Path(tmpdir) / "second" / "public"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            first_snapshot = SimpleNamespace(
                payload={"sequence_length": 3},
                vocabulary_terms=("", "[UNK]", "first"),
            )
            second_snapshot = SimpleNamespace(
                payload={"sequence_length": 4},
                vocabulary_terms=("", "[UNK]", "second"),
            )
            first_vectorizer = object()
            second_vectorizer = object()
            dashboard._load_public_snapshot_at.clear()
            with (
                patch.object(
                    dashboard,
                    "resolve_public_artifact_dir",
                    side_effect=(first, second),
                ),
                patch.object(
                    dashboard,
                    "load_app_manifest",
                    side_effect=(first_snapshot, second_snapshot),
                ),
                patch.object(
                    dashboard,
                    "create_text_vectorizer",
                    side_effect=(first_vectorizer, second_vectorizer),
                ) as create_vectorizer,
            ):
                self.assertIs(dashboard.load_vectorizer(), first_vectorizer)
                self.assertIs(dashboard.load_vectorizer(), second_vectorizer)

            dashboard._load_public_snapshot_at.clear()
            self.assertEqual(
                create_vectorizer.call_args_list[0].kwargs["vocabulary"], ("first",)
            )
            self.assertEqual(
                create_vectorizer.call_args_list[1].kwargs["vocabulary"], ("second",)
            )

    def test_prediction_reuses_one_public_snapshot_at_every_boundary(self) -> None:
        snapshot = fake_app_manifest()
        model = SimpleNamespace(predict=lambda *_args, **_kwargs: [[0.75]])
        vectorizer = Mock(side_effect=lambda value: value)
        with (
            patch.object(
                dashboard, "load_public_snapshot", return_value=snapshot
            ) as load_snapshot,
            patch.object(
                dashboard, "_load_bound_model", return_value=model
            ) as load_model,
            patch.object(
                dashboard, "load_vectorizer", return_value=vectorizer
            ) as load_vectorizer,
        ):
            probability, label = dashboard.predict_sentiment("good")

        load_snapshot.assert_called_once_with()
        load_model.assert_called_once_with(snapshot)
        load_vectorizer.assert_called_once_with(snapshot)
        self.assertEqual((probability, label), (0.75, "positive"))

    def test_vectorizer_rejects_conflicting_preimport_environment(self) -> None:
        with (
            patch.dict(os.environ, {"KERAS_BACKEND": "jax"}),
            patch.object(
                dashboard,
                "load_public_snapshot",
                return_value=SimpleNamespace(
                    payload={"sequence_length": 500}, vocabulary_terms=("", "[UNK]")
                ),
            ),
            patch(
                "src.text_preprocessing.keras.layers.TextVectorization"
            ) as vectorizer,
            self.assertRaisesRegex(ValueError, "startup environment differs"),
        ):
            dashboard.load_vectorizer()

        vectorizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
