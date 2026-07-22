import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data_prep import (
    _validate_loaded_dataset,
    build_vectorizer,
    load_verified_imdb_dataset,
    prepare_all,
)
from src.evaluation_artifact import (
    EVALUATION_MANIFEST_FILENAME,
    EVALUATION_RECORDS_FILENAME,
    canonical_source_row_bytes,
    load_evaluation_artifact_snapshot,
    publish_evaluation_artifact,
)
from src.local_training import build_model


def test_protocol(rows: list[dict[str, object]]) -> dict[str, object]:
    """Build a small frozen-protocol fixture for artifact tests.

    Parameters
    ----------
    rows : list of dict
        Test rows in official order.

    Returns
    -------
    dict of str to object
        Minimal protocol accepted by the evaluation artifact boundary.
    """
    content = b"".join(
        canonical_source_row_bytes(str(row["text"]), int(row["label"])) for row in rows
    )
    counts = [sum(row["label"] == label for row in rows) for label in range(2)]
    return {
        "protocol_version": 1,
        "status": "frozen",
        "dataset": {
            "id": "stanfordnlp/imdb",
            "config": "plain_text",
            "revision": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
            "datasets_version": "4.8.5",
            "splits": {
                "test": {
                    "rows": len(rows),
                    "label_counts": counts,
                    "raw_parquet_sha256": "a" * 64,
                    "content_sha256": hashlib.sha256(content).hexdigest(),
                }
            },
        },
    }


class EvaluationArtifactTests(unittest.TestCase):
    def test_publication_rejects_parent_replacement_at_every_commit_boundary(
        self,
    ) -> None:
        """Never return a path through a renamed or replaced visible parent."""
        from src import evaluation_artifact

        for boundary in ("before rename", "after rename", "during fsync", "return"):
            with (
                self.subTest(boundary=boundary),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                parent = root / "parent"
                parent.mkdir()
                output = parent / "evaluation"
                parked = root / "parked-parent"
                parent_identity = (parent.stat().st_dev, parent.stat().st_ino)
                real_rename = os.rename
                real_fsync = os.fsync
                real_inventory = evaluation_artifact._verify_evaluation_inventory
                replaced = False
                inventory_calls = 0

                def replace_parent() -> None:
                    nonlocal replaced
                    if replaced:
                        return
                    replaced = True
                    real_rename(parent, parked)
                    parent.mkdir()

                def replace_around_rename(source, destination, **kwargs):
                    if destination == "evaluation":
                        if boundary == "before rename":
                            replace_parent()
                        result = real_rename(source, destination, **kwargs)
                        if boundary == "after rename":
                            replace_parent()
                        return result
                    return real_rename(source, destination, **kwargs)

                def replace_during_fsync(descriptor):
                    result = real_fsync(descriptor)
                    file_stat = os.fstat(descriptor)
                    if (
                        boundary == "during fsync"
                        and (file_stat.st_dev, file_stat.st_ino) == parent_identity
                        and output.is_dir()
                    ):
                        replace_parent()
                    return result

                def replace_before_return(descriptor, files):
                    nonlocal inventory_calls
                    result = real_inventory(descriptor, files)
                    inventory_calls += 1
                    if boundary == "return" and inventory_calls == 2:
                        replace_parent()
                    return result

                with (
                    patch.object(
                        evaluation_artifact,
                        "require_secure_artifact_platform",
                        return_value=None,
                    ),
                    patch.object(
                        evaluation_artifact.os,
                        "rename",
                        side_effect=replace_around_rename,
                    ),
                    patch.object(
                        evaluation_artifact.os,
                        "fsync",
                        side_effect=replace_during_fsync,
                    ),
                    patch.object(
                        evaluation_artifact,
                        "_verify_evaluation_inventory",
                        side_effect=replace_before_return,
                    ),
                    self.assertRaisesRegex(ValueError, "parent changed"),
                ):
                    publish_evaluation_artifact(
                        self.rows, output, protocol=self.protocol
                    )

                self.assertTrue(replaced)
                self.assertFalse(output.exists())
                self.assertFalse((parked / "evaluation").exists())
                self.assertEqual(
                    publish_evaluation_artifact(
                        self.rows, output, protocol=self.protocol
                    ),
                    output,
                )

    def test_post_rename_failures_remove_owned_destination_and_allow_retry(
        self,
    ) -> None:
        """Roll back the exact installed directory across every late failure."""
        from src import evaluation_artifact

        for failure in ("verification", "fsync", "final reachability"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                output = root / "evaluation"
                parent_identity = (root.stat().st_dev, root.stat().st_ino)
                real_matches = evaluation_artifact._directory_entry_matches_descriptor
                real_fsync = os.fsync
                real_verify_parent = evaluation_artifact._verify_evaluation_parent
                verification_failed = False
                fsync_failed = False
                reachability_calls = 0

                def fail_verification(parent_descriptor, name, descriptor):
                    nonlocal verification_failed
                    if (
                        failure == "verification"
                        and name == "evaluation"
                        and not verification_failed
                    ):
                        verification_failed = True
                        return False
                    return real_matches(parent_descriptor, name, descriptor)

                def fail_fsync(descriptor):
                    nonlocal fsync_failed
                    file_stat = os.fstat(descriptor)
                    if (
                        failure == "fsync"
                        and (file_stat.st_dev, file_stat.st_ino) == parent_identity
                        and output.is_dir()
                        and not fsync_failed
                    ):
                        fsync_failed = True
                        raise OSError("injected parent fsync failure")
                    return real_fsync(descriptor)

                def fail_final_reachability(parent):
                    nonlocal reachability_calls
                    if output.is_dir():
                        reachability_calls += 1
                    if failure == "final reachability" and reachability_calls == 3:
                        raise ValueError("evaluation parent changed during publication")
                    return real_verify_parent(parent)

                with (
                    patch.object(
                        evaluation_artifact,
                        "_directory_entry_matches_descriptor",
                        side_effect=fail_verification,
                    ),
                    patch.object(
                        evaluation_artifact.os, "fsync", side_effect=fail_fsync
                    ),
                    patch.object(
                        evaluation_artifact,
                        "_verify_evaluation_parent",
                        side_effect=fail_final_reachability,
                    ),
                    self.assertRaises((OSError, ValueError)),
                ):
                    publish_evaluation_artifact(
                        self.rows, output, protocol=self.protocol
                    )

                self.assertFalse(output.exists())
                self.assertEqual(
                    publish_evaluation_artifact(
                        self.rows, output, protocol=self.protocol
                    ),
                    output,
                )

    def test_publication_rejects_staging_entry_substitution(self) -> None:
        """Reject renamed substitutes while leaving every unowned entry intact."""
        from src import evaluation_artifact

        for entry_type in ("valid directory", "file", "symlink", "directory"):
            with (
                self.subTest(entry_type=entry_type),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                output = root / "evaluation"
                parked = root / "parked-owned-staging"
                real_rename = os.rename

                def substitute(source, destination, **kwargs):
                    source_path = root / str(source)
                    if destination == "evaluation" and str(source).endswith(".staging"):
                        real_rename(source, parked.name, **kwargs)
                        if entry_type == "valid directory":
                            import shutil

                            shutil.copytree(parked, source_path)
                        elif entry_type == "file":
                            source_path.write_bytes(b"unowned")
                        elif entry_type == "symlink":
                            source_path.symlink_to("missing")
                        else:
                            source_path.mkdir()
                        return real_rename(source, destination, **kwargs)
                    return real_rename(source, destination, **kwargs)

                with (
                    patch.object(
                        evaluation_artifact,
                        "require_secure_artifact_platform",
                        return_value=None,
                    ),
                    patch.object(
                        evaluation_artifact.os, "rename", side_effect=substitute
                    ),
                    self.assertRaisesRegex(ValueError, "destination changed"),
                ):
                    publish_evaluation_artifact(
                        self.rows, output, protocol=self.protocol
                    )

                self.assertTrue(parked.is_dir())
                if entry_type == "file":
                    self.assertEqual(output.read_bytes(), b"unowned")
                elif entry_type == "symlink":
                    self.assertTrue(output.is_symlink())
                else:
                    self.assertTrue(output.is_dir())

    def test_publication_rejects_post_rename_content_mutation(self) -> None:
        """Validate exact installed content after the parent durability barrier."""
        from src import evaluation_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "evaluation"
            real_fsync = os.fsync
            mutated = False

            def mutate_after_parent_sync(descriptor):
                nonlocal mutated
                result = real_fsync(descriptor)
                if not mutated and output.is_dir():
                    mutated = True
                    (output / EVALUATION_RECORDS_FILENAME).write_bytes(b"changed\n")
                return result

            with (
                patch.object(
                    evaluation_artifact.os,
                    "fsync",
                    side_effect=mutate_after_parent_sync,
                ),
                self.assertRaisesRegex(ValueError, "changed during publication"),
            ):
                publish_evaluation_artifact(self.rows, output, protocol=self.protocol)

            self.assertFalse(output.exists())
            self.assertEqual(
                publish_evaluation_artifact(self.rows, output, protocol=self.protocol),
                output,
            )

    def test_publication_rejects_unsupported_platform_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "missing" / "evaluation"
            with (
                patch("src.artifact_compatibility.sys.platform", "win32"),
                self.assertRaisesRegex(RuntimeError, "require Linux"),
            ):
                publish_evaluation_artifact(self.rows, output, protocol=self.protocol)

            self.assertFalse(output.parent.exists())

    def setUp(self) -> None:
        self.rows = [
            {"text": "first\nreview", "label": 0},
            {"text": "café", "label": 0},
            {"text": "third", "label": 1},
            {"text": "last", "label": 1},
        ]
        self.protocol = test_protocol(self.rows)

    def test_publication_is_deterministic_ordered_and_fully_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = publish_evaluation_artifact(
                self.rows, root / "first", protocol=self.protocol
            )
            second = publish_evaluation_artifact(
                self.rows, root / "second", protocol=self.protocol
            )

            first_snapshot = load_evaluation_artifact_snapshot(
                first, protocol=self.protocol
            )
            second_snapshot = load_evaluation_artifact_snapshot(
                second, protocol=self.protocol
            )
            self.assertEqual(first_snapshot.records, second_snapshot.records)
            self.assertEqual(
                (first / EVALUATION_MANIFEST_FILENAME).read_bytes(),
                (second / EVALUATION_MANIFEST_FILENAME).read_bytes(),
            )
            records = [json.loads(line) for line in first_snapshot.records.splitlines()]
            self.assertEqual(
                [record["row_id"] for record in records],
                ["test:0", "test:1", "test:2", "test:3"],
            )
            self.assertEqual(
                [record["text"] for record in records],
                [row["text"] for row in self.rows],
            )
            self.assertTrue(first_snapshot.records.endswith(b"\n"))
            with self.assertRaises(TypeError):
                first_snapshot.manifest["dataset"]["id"] = "mutated"
            with self.assertRaises(TypeError):
                first_snapshot.manifest["records"]["fields"][0] = "mutated"

    def test_publication_rejects_existing_and_symlink_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing replacement"):
                publish_evaluation_artifact(self.rows, existing, protocol=self.protocol)

            target = root / "target"
            target.mkdir()
            symlink = root / "symlink"
            symlink.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(FileExistsError, "refusing replacement"):
                publish_evaluation_artifact(self.rows, symlink, protocol=self.protocol)

    def test_publication_cleans_owned_crash_residue_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            residue = root / ".evaluation.crashed.staging"
            residue.mkdir()
            (residue / "partial.jsonl").write_bytes(b"partial")

            artifact = publish_evaluation_artifact(
                self.rows, root / "evaluation", protocol=self.protocol
            )

            self.assertTrue(artifact.is_dir())
            self.assertFalse(residue.exists())

    def test_publication_rejects_symlinked_ancestor_before_external_cleanup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            external = root / "external"
            external.mkdir()
            residue = external / ".evaluation.hostile.staging"
            residue.mkdir()
            marker = residue / "must-survive"
            marker.write_text("external", encoding="utf-8")
            linked_parent = root / "linked-parent"
            try:
                linked_parent.symlink_to(external, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            with self.assertRaisesRegex(ValueError, "path component"):
                publish_evaluation_artifact(
                    self.rows,
                    linked_parent / "evaluation",
                    protocol=self.protocol,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "external")
            self.assertFalse((external / "evaluation").exists())
            self.assertFalse((external / ".evaluation.run.lock").exists())

    def test_publication_rejects_linked_residue_without_external_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            external = root / "external"
            external.mkdir()
            marker = external / "must-survive"
            marker.write_text("external", encoding="utf-8")
            residue = root / ".evaluation.hostile.staging"
            try:
                residue.symlink_to(external, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            with self.assertRaisesRegex(ValueError, "residue is unsafe"):
                publish_evaluation_artifact(
                    self.rows, root / "evaluation", protocol=self.protocol
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "external")
            self.assertTrue(residue.is_symlink())

    def test_loader_rejects_tampering_noncanonical_rows_and_unsafe_files(self) -> None:
        mutations = {
            "checksum": lambda path: path.write_bytes(path.read_bytes() + b"x"),
            "noncanonical": lambda path: path.write_bytes(
                path.read_bytes().replace(b'"label":0', b'"label": 0', 1)
            ),
            "identity": lambda path: path.write_bytes(
                path.read_bytes().replace(b'"test:0"', b'"test:9"', 1)
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                artifact = publish_evaluation_artifact(
                    self.rows, Path(tmpdir) / "evaluation", protocol=self.protocol
                )
                records_path = artifact / EVALUATION_RECORDS_FILENAME
                mutate(records_path)
                manifest = json.loads(
                    (artifact / EVALUATION_MANIFEST_FILENAME).read_text(
                        encoding="utf-8"
                    )
                )
                if name != "checksum":
                    manifest["checksums"][EVALUATION_RECORDS_FILENAME] = (
                        "sha256:"
                        + hashlib.sha256(records_path.read_bytes()).hexdigest()
                    )
                    (artifact / EVALUATION_MANIFEST_FILENAME).write_text(
                        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                    )
                with self.assertRaises(ValueError):
                    load_evaluation_artifact_snapshot(artifact, protocol=self.protocol)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = publish_evaluation_artifact(
                self.rows, root / "evaluation", protocol=self.protocol
            )
            records = artifact / EVALUATION_RECORDS_FILENAME
            outside = root / "outside.jsonl"
            records.replace(outside)
            records.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "contained regular file"):
                load_evaluation_artifact_snapshot(artifact, protocol=self.protocol)

        if os.name != "nt":
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                artifact = publish_evaluation_artifact(
                    self.rows, root / "evaluation", protocol=self.protocol
                )
                os.link(
                    artifact / EVALUATION_RECORDS_FILENAME,
                    root / "hardlink.jsonl",
                )
                with self.assertRaisesRegex(ValueError, "contained regular file"):
                    load_evaluation_artifact_snapshot(artifact, protocol=self.protocol)

    def test_loader_rejects_manifest_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = publish_evaluation_artifact(
                self.rows, Path(tmpdir) / "evaluation", protocol=self.protocol
            )
            manifest_path = artifact / EVALUATION_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["unexpected"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid field set"):
                load_evaluation_artifact_snapshot(artifact, protocol=self.protocol)

    def test_loader_rejects_unmanifested_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = publish_evaluation_artifact(
                self.rows, Path(tmpdir) / "evaluation", protocol=self.protocol
            )
            (artifact / "unexpected.txt").write_text("unbound", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unexpected files"):
                load_evaluation_artifact_snapshot(artifact, protocol=self.protocol)

    def test_dataset_validation_hard_fails_each_frozen_property(self) -> None:
        class Info:
            dataset_name = "imdb"
            config_name = "plain_text"

        class Split:
            info = Info()
            column_names = ["text", "label"]

            def __init__(self, rows):
                self.rows = rows

            def __iter__(self):
                return iter(self.rows)

            def __len__(self):
                return len(self.rows)

        rows = [
            {"text": "negative", "label": 0},
            {"text": "positive", "label": 1},
        ]
        raw = b"raw parquet bytes"
        content = b"".join(
            canonical_source_row_bytes(row["text"], row["label"]) for row in rows
        )
        split_spec = {
            "rows": 2,
            "label_counts": [1, 1],
            "label_index_ranges": [[0, 0], [1, 1]],
            "raw_parquet_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": hashlib.sha256(content).hexdigest(),
        }
        protocol = {
            "dataset": {
                "id": "stanfordnlp/imdb",
                "config": "plain_text",
                "splits": {"train": split_spec},
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "train.parquet"
            raw_path.write_bytes(raw)
            dataset = {"train": Split(rows)}
            _validate_loaded_dataset(dataset, protocol, {"train": raw_path})

            hostile_cases = {
                "raw Parquet": (dataset, b"tampered", protocol),
                "identity": (
                    {"train": Split(rows)},
                    raw,
                    {
                        **protocol,
                        "dataset": {**protocol["dataset"], "config": "wrong"},
                    },
                ),
                "row count": ({"train": Split(rows[:1])}, raw, protocol),
                "labels": ({"train": Split(list(reversed(rows)))}, raw, protocol),
                "canonical content": (
                    {"train": Split([rows[0], {"text": "changed", "label": 1}])},
                    raw,
                    protocol,
                ),
            }
            for message, (
                candidate,
                raw_bytes,
                candidate_protocol,
            ) in hostile_cases.items():
                with self.subTest(message=message):
                    raw_path.write_bytes(raw_bytes)
                    with self.assertRaisesRegex(ValueError, message):
                        _validate_loaded_dataset(
                            candidate,
                            candidate_protocol,
                            {"train": raw_path},
                        )

    def test_cached_dataset_matches_every_frozen_identity_and_checksum(self) -> None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            for split in ("train", "test", "unsupervised"):
                hf_hub_download(
                    repo_id="stanfordnlp/imdb",
                    filename=f"plain_text/{split}-00000-of-00001.parquet",
                    repo_type="dataset",
                    revision="e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
                    local_files_only=True,
                )
        except LocalEntryNotFoundError:
            self.skipTest("immutable IMDB revision is not available in the local cache")

        dataset = load_verified_imdb_dataset(local_files_only=True)
        vectorizer = build_vectorizer(dataset["train"]["text"])
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = publish_evaluation_artifact(
                dataset["test"], Path(tmpdir) / "evaluation"
            )
            snapshot = load_evaluation_artifact_snapshot(artifact)

        self.assertEqual(
            {split: len(rows) for split, rows in dataset.items()},
            {"train": 25000, "test": 25000, "unsupervised": 50000},
        )
        self.assertEqual(snapshot.manifest["records"]["row_count"], 25000)
        self.assertTrue(snapshot.records.startswith(b'{"label":0,"row_id":"test:0"'))
        self.assertEqual(len(vectorizer.get_vocabulary()), 20000)

    def test_registered_datasets_version_mismatch_hard_fails_before_loading(
        self,
    ) -> None:
        with (
            patch("src.data_prep.importlib.metadata.version", return_value="0.0.0"),
            patch("datasets.load_dataset") as load_dataset,
        ):
            with self.assertRaisesRegex(ValueError, "datasets version differs"):
                load_verified_imdb_dataset()

        load_dataset.assert_not_called()

    def test_vectorizer_rejects_hostile_framework_versions_before_construction(
        self,
    ) -> None:
        import keras
        import tensorflow as tf

        with (
            patch.object(np, "__version__", "99.0.0"),
            patch.object(keras, "__version__", "99.0.0"),
            patch.object(tf, "__version__", "99.0.0"),
            patch.object(keras.layers, "TextVectorization") as vectorizer,
            self.assertRaisesRegex(ValueError, "framework versions differ"),
        ):
            build_vectorizer(["must not be preprocessed"])

        vectorizer.assert_not_called()

    def test_model_rejects_hostile_runtime_before_construction(self) -> None:
        import keras

        with (
            patch.object(np, "__version__", "99.0.0"),
            patch.object(keras, "Input") as model_input,
            self.assertRaisesRegex(ValueError, "framework versions differ"),
        ):
            build_model(20_000, 500, 100)

        model_input.assert_not_called()

    def test_preparation_rejects_hostile_frameworks_before_dataset_loading(
        self,
    ) -> None:
        import keras
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch.object(np, "__version__", "99.0.0"),
                patch.object(keras, "__version__", "99.0.0"),
                patch.object(tf, "__version__", "99.0.0"),
                patch("src.data_prep.load_verified_imdb_dataset") as load_dataset,
                self.assertRaisesRegex(ValueError, "framework versions differ"),
            ):
                prepare_all(
                    2,
                    root / "clients",
                    root / "public",
                    root / "evaluation",
                )

        load_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
