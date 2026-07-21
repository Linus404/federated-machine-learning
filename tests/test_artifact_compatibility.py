import json
import tempfile
import unittest
from pathlib import Path

from src.app_manifest import load_app_manifest
from src.artifact_compatibility import (
    ARTIFACT_SCHEMA_VERSION,
    validate_artifact_schema,
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


if __name__ == "__main__":
    unittest.main()
