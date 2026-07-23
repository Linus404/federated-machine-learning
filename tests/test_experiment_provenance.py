"""Tests for compact experiment-cell provenance finalization."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import experiment_provenance


class ExperimentProvenanceTests(unittest.TestCase):
    """Verify deterministic evidence capture and unsafe-output rejection."""

    def test_writes_canonical_manifest_and_checksum_sidecar(self) -> None:
        """Capture all inputs, outputs, revision state, seeds, and protocol bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            public = root / "public"
            evaluation = root / "evaluation"
            client = root / "client-0"
            output = root / "cell"
            for directory in (public, evaluation, client, output):
                directory.mkdir()
            files = {
                public / "manifest.json": b"public manifest\n",
                public / "vocab.txt": b"\n[UNK]\nword\n",
                evaluation / "manifest.json": b"evaluation manifest\n",
                evaluation / "test.jsonl": b'{"row_id":"test:0"}\n',
                client / "client_metadata.json": b"client metadata\n",
                client / "reviews.jsonl": b'{"row_id":"train:0"}\n',
                output / "results.json": b'{"accuracy":0.5}\n',
                output / "model.keras": b"model",
            }
            for path, content in files.items():
                path.write_bytes(content)
            protocol_path = root / "scientific-protocol-v1.toml"
            protocol_path.write_bytes(b"frozen protocol\n")
            protocol = {
                "protocol_version": 1,
                "provenance": {
                    "canonical_serialization": {
                        "encoding": "utf-8",
                        "ensure_ascii": True,
                        "sort_keys": True,
                        "separators": [",", ":"],
                        "allow_nan": False,
                    }
                },
                "dataset": {
                    "id": "dataset",
                    "config": "plain",
                    "revision": "revision",
                    "datasets_version": "1.0",
                    "splits": {
                        "train": {
                            "raw_parquet_sha256": "a" * 64,
                            "content_sha256": "b" * 64,
                        },
                        "test": {
                            "raw_parquet_sha256": "c" * 64,
                            "content_sha256": "d" * 64,
                        },
                    },
                },
            }
            public_snapshot = SimpleNamespace(
                vocabulary_path=public / "vocab.txt",
                manifest_bytes=files[public / "manifest.json"],
                vocabulary_bytes=files[public / "vocab.txt"],
            )
            evaluation_snapshot = SimpleNamespace(
                directory=evaluation,
                records=files[evaluation / "test.jsonl"],
            )
            client_snapshot = SimpleNamespace(
                directory=client,
                metadata={"client_id": 0},
                metadata_bytes=files[client / "client_metadata.json"],
                records_bytes=files[client / "reviews.jsonl"],
            )
            result = {
                "strategy": "fedavg",
                "config": {"seed": 67, "partition": "iid_stratified"},
                "seeds": {"global_model_initialization": 123},
                "clients": [{"seeds": {"dropout": 456}}],
            }
            revision = {"commit": "0" * 40, "dirty": False, "source": "git"}

            with (
                patch.object(
                    experiment_provenance, "SCIENTIFIC_PROTOCOL_PATH", protocol_path
                ),
                patch.object(
                    experiment_provenance, "_code_revision", return_value=revision
                ),
            ):
                path = experiment_provenance.write_experiment_provenance(
                    output,
                    result=result,
                    public_manifest=public_snapshot,
                    evaluation_snapshot=evaluation_snapshot,
                    client_snapshots=[client_snapshot],
                    protocol=protocol,
                )

            content = path.read_bytes()
            payload = json.loads(content)
            self.assertEqual(
                content, experiment_provenance._canonical_bytes(payload, protocol)
            )
            self.assertEqual(payload["code"], revision)
            self.assertEqual(payload["seeds"]["master"], 67)
            self.assertEqual(
                payload["seeds"]["derived"],
                [
                    {"path": "clients.0.seeds.dropout", "value": 456},
                    {"path": "seeds.global_model_initialization", "value": 123},
                ],
            )
            self.assertEqual(set(payload["outputs"]), {"model.keras", "results.json"})
            self.assertEqual(len(payload["inputs"]), 6)
            sidecar = (
                output / experiment_provenance.PROVENANCE_CHECKSUM_FILENAME
            ).read_text()
            self.assertEqual(
                sidecar,
                f"{hashlib.sha256(content).hexdigest()}  provenance.json\n",
            )

    def test_rejects_symlinked_output_artifact(self) -> None:
        """Refuse provenance when any declared cell output is a symlink."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "cell"
            output.mkdir()
            target = root / "outside"
            target.write_text("outside", encoding="utf-8")
            try:
                (output / "result-link").symlink_to(target)
            except OSError as error:
                self.skipTest(f"file symlinks are unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "contained regular file"):
                experiment_provenance._output_checksums(output.resolve())


if __name__ == "__main__":
    unittest.main()
