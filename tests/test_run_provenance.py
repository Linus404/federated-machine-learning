import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.artifact_compatibility import ARTIFACT_SCHEMA_VERSION
from src.run_provenance import (
    load_run_provenance_manifest,
    write_run_provenance_manifest,
)


def write_public_dataset_contract(path: Path) -> None:
    vocabulary = "\n[UNK]\ngood\nbad\n"
    (path / "vocab.txt").write_text(vocabulary, encoding="utf-8")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "embedding_dim": 100,
                "sequence_length": 500,
                "vocabulary_size": 4,
                "vocabulary": {"filename": "vocab.txt"},
                "provenance": "stanfordnlp/imdb training split",
            }
        ),
        encoding="utf-8",
    )


class RunProvenanceTests(unittest.TestCase):
    def test_manifest_records_run_inputs_and_public_dataset_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            public_dir = root / "public"
            server_dir = root / "server"
            public_dir.mkdir()
            write_public_dataset_contract(public_dir)
            run_config = {
                "num-server-rounds": 3,
                "public-artifact-dir": public_dir,
                "random-seed": 19,
                "use-huber": True,
            }

            with (
                patch(
                    "src.run_provenance.uuid.uuid4",
                    return_value="12345678-1234-4234-9234-123456789abc",
                ),
                patch(
                    "src.run_provenance._code_revision",
                    return_value={"commit": "a" * 40, "dirty": False},
                ),
                patch(
                    "src.run_provenance._environment_metadata",
                    return_value={"python": "3.12.1"},
                ),
            ):
                manifest_path = write_run_provenance_manifest(
                    server_dir,
                    run_config,
                    public_artifact_dir=public_dir,
                    flower_run_id=42,
                    created_at="2026-07-21T20:00:00Z",
                )

            payload = load_run_provenance_manifest(manifest_path)
            self.assertEqual(payload["schema_version"], ARTIFACT_SCHEMA_VERSION)
            self.assertEqual(payload["run_id"], "12345678-1234-4234-9234-123456789abc")
            self.assertEqual(payload["flower_run_id"], 42)
            self.assertEqual(payload["created_at"], "2026-07-21T20:00:00Z")
            self.assertEqual(
                payload["run_config"],
                {
                    "num-server-rounds": 3,
                    "public-artifact-dir": str(public_dir),
                    "random-seed": 19,
                    "use-huber": True,
                },
            )
            self.assertEqual(payload["environment"], {"python": "3.12.1"})
            self.assertEqual(payload["code_revision"]["commit"], "a" * 40)
            self.assertEqual(payload["seeds"]["run_config"], {"random-seed": 19})
            self.assertEqual(
                payload["dataset"]["identity"],
                "stanfordnlp/imdb training split",
            )
            manifest_bytes = (public_dir / "manifest.json").read_bytes()
            self.assertEqual(
                payload["dataset"]["checksums"]["manifest.json"],
                "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
            )
            self.assertEqual(
                payload["dataset"]["private_client_shards"]["status"],
                "not_collected",
            )

    def test_manifest_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            with (
                patch(
                    "src.run_provenance.uuid.uuid4",
                    side_effect=[
                        "12345678-1234-4234-9234-123456789abc",
                        "22345678-1234-4234-9234-123456789abc",
                    ],
                ),
                patch(
                    "src.run_provenance._code_revision",
                    return_value={"commit": None, "dirty": None},
                ),
                patch("src.run_provenance._environment_metadata", return_value={}),
            ):
                manifest_path = write_run_provenance_manifest(path, {})
                with self.assertRaises(FileExistsError):
                    write_run_provenance_manifest(path, {})

            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"],
                "12345678-1234-4234-9234-123456789abc",
            )

    def test_distinct_runs_receive_distinct_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch(
                    "src.run_provenance._code_revision",
                    return_value={"commit": None, "dirty": None, "source": "test"},
                ),
                patch("src.run_provenance._environment_metadata", return_value={}),
            ):
                first = load_run_provenance_manifest(
                    write_run_provenance_manifest(root / "first", {})
                )
                second = load_run_provenance_manifest(
                    write_run_provenance_manifest(root / "second", {})
                )

            self.assertNotEqual(first["run_id"], second["run_id"])

    def test_invalid_run_config_is_rejected_before_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            with self.assertRaisesRegex(ValueError, "run configuration"):
                write_run_provenance_manifest(path, {"nested": {"unsafe": True}})

            self.assertFalse((path / "run_manifest.json").exists())

    def test_loader_rejects_incomplete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_manifest.json"
            path.write_text(
                json.dumps({"schema_version": ARTIFACT_SCHEMA_VERSION}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing"):
                load_run_provenance_manifest(path)


if __name__ == "__main__":
    unittest.main()
