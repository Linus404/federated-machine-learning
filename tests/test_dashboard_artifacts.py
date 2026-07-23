import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import dashboard
import numpy as np
from src.artifact_compatibility import (
    SERVER_ARTIFACT_SCHEMA_VERSION,
    write_server_artifact_manifest,
)
from tests.artifact_helpers import compatible_model, fake_app_manifest


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
    def test_prediction_rejects_malformed_model_outputs(self) -> None:
        """Reject invalid inference outputs before rendering them.

        Returns
        -------
        None
        """
        manifest = fake_app_manifest()
        manifest.vocabulary_terms = ("", "[UNK]")
        for output in (
            np.array([0.5]),
            np.array([[np.nan]]),
            np.array([[1.1]]),
            np.array([["invalid"]]),
        ):
            with self.subTest(output=output):
                model = MagicMock()
                model.predict.return_value = output
                with (
                    patch.object(dashboard, "validate_protocol_runtime"),
                    patch.object(
                        dashboard,
                        "load_app_manifest",
                        return_value=manifest,
                    ),
                    patch.object(dashboard, "_load_bound_model", return_value=model),
                    patch.object(
                        dashboard,
                        "create_text_vectorizer",
                        return_value=lambda inputs: inputs,
                    ),
                    self.assertRaisesRegex(ValueError, "prediction"),
                ):
                    dashboard.predict_sentiment("A review")

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
                self.assertRaisesRegex(ValueError, "no valid schema_version"),
            ):
                dashboard.load_metrics()

            (root / "metrics.csv").unlink()
            (root / "current.json").write_text("not-json", encoding="utf-8")
            with (
                patch.object(dashboard, "ARTIFACT_ROOT", root),
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
                        dashboard, "load_app_manifest", return_value=fake_app_manifest()
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
                    dashboard, "load_app_manifest", return_value=fake_app_manifest()
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

                with self.assertRaisesRegex(ValueError, error):
                    dashboard.load_metrics(metrics_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            metrics_path = path / "metrics.csv"
            metrics_path.write_text(
                "round,loss,accuracy\n1,0.5,0.75\n", encoding="utf-8"
            )
            write_server_manifest(path, SERVER_ARTIFACT_SCHEMA_VERSION)

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
                    dashboard, "load_app_manifest", return_value=fake_app_manifest()
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
                patch.object(dashboard, "load_app_manifest", return_value=incompatible),
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
                    dashboard, "load_app_manifest", return_value=fake_app_manifest()
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

    def test_vectorizer_rejects_conflicting_preimport_environment(self) -> None:
        with (
            patch.dict(os.environ, {"KERAS_BACKEND": "jax"}),
            patch.object(
                dashboard,
                "load_app_manifest",
                return_value=SimpleNamespace(
                    payload={"sequence_length": 500}, vocabulary_terms=("", "[UNK]")
                ),
            ),
            patch(
                "src.text_preprocessing.keras.layers.TextVectorization"
            ) as vectorizer,
            self.assertRaisesRegex(ValueError, "startup environment differs"),
        ):
            dashboard.load_vectorizer.clear()
            dashboard.load_vectorizer()

        dashboard.load_vectorizer.clear()
        vectorizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
