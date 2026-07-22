import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.app_manifest import load_app_manifest
from src.artifact_compatibility import (
    ARTIFACT_SCHEMA_VERSION,
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    SERVER_ARTIFACT_SCHEMA_VERSION,
    SERVER_ARTIFACTS,
    canonical_json_bytes,
    load_server_artifact_manifest,
    load_server_artifact_snapshot,
    validate_artifact_schema,
    write_server_artifact_manifest,
)
from tests.artifact_helpers import fake_app_manifest


class ArtifactCompatibilityTests(unittest.TestCase):
    def test_compatibility_policy_records_client_schema_two_migration(self) -> None:
        policy = (
            Path(__file__).resolve().parent.parent / "COMPATIBILITY.md"
        ).read_text(encoding="utf-8")

        self.assertIn("client-N/client_metadata.json` uses `schema_version: 2`", policy)
        self.assertIn("client schema `1` shards", policy)
        self.assertIn("Regenerate public schema\n`2`, client schema `2`", policy)
        self.assertNotIn("all other persisted contracts remain schema `1`", policy)

    def test_compatibility_policy_lists_exact_field_contracts(self) -> None:
        policy = (
            Path(__file__).resolve().parent.parent / "COMPATIBILITY.md"
        ).read_text(encoding="utf-8")

        policy_lines = set(policy.splitlines())
        exact_contracts = {
            "them. Exact field sets are required for:",
            "- client metadata, records contracts, and public-manifest binding objects;",
            "- public-manifest dataset and vocabulary metadata objects;",
            "- server binding and model-dimension objects;",
            "- evaluation manifests and records;",
            "- prepared-generation indexes; and",
            "- run-provenance private-client-shard status, identity, and public-manifest",
            "  objects.",
        }

        self.assertLessEqual(exact_contracts, policy_lines)

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
                json.dumps({"schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION + 1}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "public manifest.*newer"):
                load_app_manifest(public_artifact_dir=path)

    def test_public_manifest_checks_each_schema_version_state(self) -> None:
        vocabulary = b"\n[UNK]\ngood\n"
        vocabulary_sha256 = hashlib.sha256(vocabulary).hexdigest()
        dataset = {
            "id": "example/imdb",
            "config": "plain_text",
            "revision": "frozen",
            "datasets_version": "1.0.0",
            "split": "train",
            "rows": 3,
            "raw_parquet_sha256": "1" * 64,
            "content_sha256": "2" * 64,
        }
        protocol = {
            "dataset": {
                **{
                    key: dataset[key]
                    for key in (
                        "id",
                        "config",
                        "revision",
                        "datasets_version",
                    )
                },
                "splits": {
                    "train": {
                        key: dataset[key]
                        for key in (
                            "rows",
                            "raw_parquet_sha256",
                            "content_sha256",
                        )
                    }
                },
            },
            "preprocessing": {
                "vocabulary_size": 3,
                "max_tokens": 3,
                "output_sequence_length": 500,
                "vocabulary_sha256": vocabulary_sha256,
            },
            "model": {
                "vocabulary_size": 3,
                "sequence_length": 500,
                "embedding_dimension": 100,
            },
        }
        valid_payload = {
            "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
            "embedding_dim": 100,
            "sequence_length": 500,
            "vocabulary_size": 3,
            "vocabulary": {
                "filename": "vocab.txt",
                "sha256": vocabulary_sha256,
                "size_bytes": len(vocabulary),
            },
            "dataset": dataset,
        }
        for version, error in (
            (None, "no valid"),
            (1, "older.*regenerate or migrate"),
            (PUBLIC_ARTIFACT_SCHEMA_VERSION + 1, "newer"),
        ):
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
            (path / "vocab.txt").write_bytes(vocabulary)
            (path / "manifest.json").write_bytes(canonical_json_bytes(valid_payload))
            manifest = load_app_manifest(public_artifact_dir=path, protocol=protocol)
            self.assertEqual(manifest.payload, valid_payload)
            with self.assertRaises(TypeError):
                manifest.payload["dataset"]["id"] = "mutated"
            with self.assertRaises(TypeError):
                manifest.payload["vocabulary"]["sha256"] = "0" * 64

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "vocab.txt").write_bytes(vocabulary)
            nested_payload = {
                **valid_payload,
                "producer": {"history": [{"revision": "frozen"}]},
            }
            (path / "manifest.json").write_bytes(canonical_json_bytes(nested_payload))
            manifest = load_app_manifest(public_artifact_dir=path, protocol=protocol)
            with self.assertRaises(TypeError):
                manifest.payload["producer"]["history"][0]["revision"] = "mutated"

        invalid_dimensions = (
            ("embedding_dim", -3),
            ("sequence_length", 4.75),
            ("sequence_length", True),
            ("vocabulary_size", 4),
        )
        for field, value in invalid_dimensions:
            with (
                self.subTest(field=field, value=value),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                path = Path(tmpdir)
                (path / "vocab.txt").write_bytes(vocabulary)
                payload = {**valid_payload, field: value}
                (path / "manifest.json").write_bytes(canonical_json_bytes(payload))
                with self.assertRaisesRegex(ValueError, "protocol dimensions"):
                    load_app_manifest(public_artifact_dir=path, protocol=protocol)

    def test_server_artifact_manifest_declares_supported_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            manifest_path = write_server_artifact_manifest(
                path, app_manifest=fake_app_manifest()
            )

            self.assertEqual(
                canonical_json_bytes(
                    {"artifacts": load_server_artifact_manifest(path)["artifacts"]}
                ),
                canonical_json_bytes({"artifacts": SERVER_ARTIFACTS}),
            )
            self.assertEqual(manifest_path.parent, path)

    def test_server_artifact_manifest_accepts_additive_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            manifest_path = write_server_artifact_manifest(
                path, app_manifest=fake_app_manifest()
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["producer"] = "test"
            payload["artifacts"]["model"]["description"] = "global model"
            payload["artifacts"]["diagnostics"] = {"filename": "debug.json"}
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(
                canonical_json_bytes(load_server_artifact_manifest(path)),
                canonical_json_bytes(payload),
            )

    def test_server_artifact_manifest_rejects_changed_layout_without_version(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            write_server_artifact_manifest(path, app_manifest=fake_app_manifest())
            manifest_path = path / "artifact_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["artifacts"]["metrics"]["columns"].append("unsupported")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match.*regenerate"):
                load_server_artifact_manifest(path)

    def test_server_snapshot_binding_is_exact_and_deeply_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            app_manifest = fake_app_manifest()
            manifest_path = write_server_artifact_manifest(
                path, app_manifest=app_manifest
            )
            snapshot = load_server_artifact_snapshot(path, app_manifest=app_manifest)

            with self.assertRaises(TypeError):
                snapshot.manifest["binding"]["model_dimensions"]["sequence_length"] = 1
            with self.assertRaises(TypeError):
                snapshot.manifest["artifacts"]["metrics"]["columns"][0] = "epoch"

            for field, value in (
                ("public_manifest_checksum", "sha256:" + "0" * 64),
                ("vocabulary_checksum", "sha256:" + "1" * 64),
                ("model_dimensions", {"vocabulary_size": 1}),
            ):
                with self.subTest(field=field):
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    payload["binding"][field] = value
                    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "bound|binding"):
                        load_server_artifact_snapshot(path, app_manifest=app_manifest)
                    write_server_artifact_manifest(path, app_manifest=app_manifest)

    def test_schema_one_server_artifact_requires_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            manifest_path = write_server_artifact_manifest(
                path, app_manifest=fake_app_manifest()
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["schema_version"] = 1
            payload.pop("binding")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "older.*regenerate"):
                load_server_artifact_snapshot(path)

    def test_unversioned_server_artifact_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "schema_version.*regenerate"):
                load_server_artifact_manifest(Path(tmpdir))

    def test_server_artifact_manifest_rejects_older_and_newer_schemas(self) -> None:
        for version, direction in (
            (SERVER_ARTIFACT_SCHEMA_VERSION - 1, "older"),
            (SERVER_ARTIFACT_SCHEMA_VERSION + 1, "newer"),
        ):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)
                manifest_path = write_server_artifact_manifest(
                    path, app_manifest=fake_app_manifest()
                )
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload["schema_version"] = version
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, direction):
                    load_server_artifact_manifest(path)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO files require POSIX")
    def test_finalization_rejects_non_regular_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            for name in ("global_model.keras", "metrics.csv", "run_manifest.json"):
                (path / name).touch()
            os.mkfifo(path / "unexpected.pipe")

            with self.assertRaisesRegex(ValueError, "contained regular file"):
                write_server_artifact_manifest(
                    path, app_manifest=fake_app_manifest(), finalized=True
                )

    def test_finalization_rejects_hard_linked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run"
            path.mkdir()
            outside = Path(tmpdir) / "outside.keras"
            outside.touch()
            os.link(outside, path / "global_model.keras")
            for name in ("metrics.csv", "run_manifest.json"):
                (path / name).touch()

            with self.assertRaisesRegex(ValueError, "contained regular file"):
                write_server_artifact_manifest(
                    path, app_manifest=fake_app_manifest(), finalized=True
                )


if __name__ == "__main__":
    unittest.main()
