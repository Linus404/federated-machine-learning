import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.artifact_compatibility import ARTIFACT_SCHEMA_VERSION
from src.artifact_history import resolve_current_run_dir
from src.local_training import DEFAULT_VALIDATION_SEED, train
from src.run_provenance import (
    load_run_provenance_manifest,
    write_run_provenance_manifest,
)


class LocalTrainingTests(unittest.TestCase):
    def test_train_writes_immutable_run_manifest_before_loading_private_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = argparse.Namespace(
                batch_size=8,
                client_data_dir=root / "client",
                epochs=2,
                public_artifact_dir=root / "public",
                quiet=True,
                run_artifact_dir=root / "run",
                validation_split=0.25,
            )
            args.public_artifact_dir.mkdir()
            vocabulary = b"\n[UNK]\ngood\nbad\n"
            vocabulary_sha256 = hashlib.sha256(vocabulary).hexdigest()
            (args.public_artifact_dir / "vocab.txt").write_bytes(vocabulary)
            dataset = {
                "id": "example/imdb",
                "config": "plain_text",
                "revision": "frozen",
                "datasets_version": "1.0.0",
                "split": "train",
                "rows": 4,
                "raw_parquet_sha256": "1" * 64,
                "content_sha256": "2" * 64,
            }
            (args.public_artifact_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": ARTIFACT_SCHEMA_VERSION,
                        "embedding_dim": 8,
                        "sequence_length": 16,
                        "vocabulary_size": 4,
                        "vocabulary": {
                            "filename": "vocab.txt",
                            "sha256": vocabulary_sha256,
                            "size_bytes": len(vocabulary),
                        },
                        "dataset": dataset,
                    }
                ),
                encoding="utf-8",
            )
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
                    "vocabulary_size": 4,
                    "vocabulary_sha256": vocabulary_sha256,
                },
            }
            history = SimpleNamespace(
                history={
                    "loss": [0.4],
                    "accuracy": [0.8],
                    "val_loss": [0.5],
                    "val_accuracy": [0.75],
                }
            )
            model = MagicMock()
            model.save.side_effect = lambda path: Path(path).write_bytes(b"model")
            model.fit.return_value = history
            model.evaluate.return_value = (0.5, 0.75)
            training_data = ((MagicMock(), MagicMock()), (MagicMock(), MagicMock()))
            events: list[str] = []

            def write_manifest(*call_args, **call_kwargs):
                events.append("manifest")
                return write_run_provenance_manifest(*call_args, **call_kwargs)

            with (
                patch(
                    "src.artifact_history.write_run_provenance_manifest",
                    side_effect=write_manifest,
                ),
                patch(
                    "src.local_training.load_client_shard",
                    side_effect=lambda *args, **kwargs: (
                        events.append("private-data") or training_data
                    ),
                ) as load_shard,
                patch(
                    "src.local_training.build_model_from_manifest", return_value=model
                ),
                patch(
                    "src.app_manifest.load_scientific_protocol",
                    return_value=protocol,
                ),
            ):
                train(args)

            self.assertEqual(events, ["manifest", "private-data"])
            load_shard.assert_called_once()
            payload = load_run_provenance_manifest(
                resolve_current_run_dir(args.run_artifact_dir) / "run_manifest.json"
            )
            self.assertEqual(
                payload["run_config"],
                {
                    "artifact-retention-runs": 10,
                    "batch-size": 8,
                    "client-data-dir": str(args.client_data_dir),
                    "epochs": 2,
                    "public-artifact-dir": str(args.public_artifact_dir),
                    "quiet": True,
                    "run-artifact-dir": str(args.run_artifact_dir),
                    "validation-seed": DEFAULT_VALIDATION_SEED,
                    "validation-split": 0.25,
                },
            )


if __name__ == "__main__":
    unittest.main()
