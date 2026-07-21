import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.artifact_compatibility import ARTIFACT_SCHEMA_VERSION
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
            (args.public_artifact_dir / "vocab.txt").write_text(
                "\n[UNK]\ngood\nbad\n", encoding="utf-8"
            )
            (args.public_artifact_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": ARTIFACT_SCHEMA_VERSION,
                        "embedding_dim": 8,
                        "sequence_length": 16,
                        "vocabulary_size": 4,
                        "vocabulary": {"filename": "vocab.txt"},
                        "provenance": "test dataset",
                    }
                ),
                encoding="utf-8",
            )
            history = SimpleNamespace(
                history={
                    "loss": [0.4],
                    "accuracy": [0.8],
                    "val_loss": [0.5],
                    "val_accuracy": [0.75],
                }
            )
            model = MagicMock()
            model.fit.return_value = history
            model.evaluate.return_value = (0.5, 0.75)
            training_data = ((MagicMock(), MagicMock()), (MagicMock(), MagicMock()))
            events: list[str] = []

            def write_manifest(*call_args, **call_kwargs):
                events.append("manifest")
                return write_run_provenance_manifest(*call_args, **call_kwargs)

            with (
                patch(
                    "src.local_training.write_run_provenance_manifest",
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
            ):
                train(args)

            self.assertEqual(events, ["manifest", "private-data"])
            load_shard.assert_called_once()
            payload = load_run_provenance_manifest(
                args.run_artifact_dir / "run_manifest.json"
            )
            self.assertEqual(
                payload["run_config"],
                {
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
