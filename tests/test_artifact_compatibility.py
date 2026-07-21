import json
import tempfile
import unittest
from pathlib import Path

from src.app_manifest import load_app_manifest
from src.artifact_compatibility import (
    ARTIFACT_SCHEMA_VERSION,
    SERVER_ARTIFACTS,
    load_server_artifact_manifest,
    validate_artifact_schema,
    write_server_artifact_manifest,
)


class ArtifactCompatibilityTests(unittest.TestCase):
    def test_current_schema_is_supported(self) -> None:
        payload = {"schema_version": ARTIFACT_SCHEMA_VERSION}

        self.assertIs(validate_artifact_schema(payload, "test artifact"), payload)

    def test_unversioned_schema_is_rejected_with_regeneration_guidance(self) -> None:
        with self.assertRaisesRegex(ValueError, "regenerate"):
            validate_artifact_schema({}, "test artifact")

    def test_older_schema_is_rejected_with_regeneration_guidance(self) -> None:
        with self.assertRaisesRegex(ValueError, "older.*regenerate"):
            validate_artifact_schema({"schema_version": 0}, "test artifact")

    def test_newer_schema_is_rejected_with_regeneration_guidance(self) -> None:
        with self.assertRaisesRegex(ValueError, "newer.*regenerate"):
            validate_artifact_schema({"schema_version": 2}, "test artifact")

    def test_public_manifest_rejects_incompatible_schema_before_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "manifest.json").write_text(
                json.dumps({"schema_version": 2}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "public manifest.*newer"):
                load_app_manifest(public_artifact_dir=path)

    def test_public_manifest_checks_each_schema_version_state(self) -> None:
        valid_payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "embedding_dim": 100,
            "sequence_length": 500,
            "vocabulary_size": 3,
            "vocabulary": {"filename": "vocab.txt"},
        }
        for version, error in ((None, "no valid"), (0, "older"), (2, "newer")):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                payload = dict(valid_payload)
                if version is None:
                    payload.pop("schema_version")
                else:
                    payload["schema_version"] = version
                (path / "manifest.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

                with self.assertRaisesRegex(ValueError, error):
                    load_app_manifest(public_artifact_dir=path)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "manifest.json").write_text(
                json.dumps(valid_payload), encoding="utf-8"
            )
            manifest = load_app_manifest(public_artifact_dir=path)
            self.assertEqual(manifest.payload, valid_payload)

    def test_server_artifact_manifest_declares_supported_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            manifest_path = write_server_artifact_manifest(path)

            self.assertEqual(
                load_server_artifact_manifest(path)["artifacts"], SERVER_ARTIFACTS
            )
            self.assertEqual(manifest_path.parent, path)

    def test_server_artifact_manifest_accepts_additive_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            manifest_path = write_server_artifact_manifest(path)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["producer"] = "test"
            payload["artifacts"]["model"]["description"] = "global model"
            payload["artifacts"]["diagnostics"] = {"filename": "debug.json"}
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(load_server_artifact_manifest(path), payload)

    def test_server_artifact_manifest_rejects_changed_layout_without_version(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            write_server_artifact_manifest(path)
            manifest_path = path / "artifact_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["artifacts"]["metrics"]["columns"].append("unsupported")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match.*regenerate"):
                load_server_artifact_manifest(path)

    def test_unversioned_server_artifact_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "schema_version.*regenerate"):
                load_server_artifact_manifest(Path(tmpdir))

    def test_server_artifact_manifest_rejects_older_and_newer_schemas(self) -> None:
        for version, direction in ((0, "older"), (2, "newer")):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                manifest_path = write_server_artifact_manifest(path)
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload["schema_version"] = version
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, direction):
                    load_server_artifact_manifest(path)


if __name__ == "__main__":
    unittest.main()
