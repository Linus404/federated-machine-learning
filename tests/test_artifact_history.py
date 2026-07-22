import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.app_manifest import AppManifest, expected_train_dataset, load_app_manifest
from src.artifact_history import (
    load_current_run_snapshot,
    prune_run_history,
    publish_completed_run,
    resolve_current_run_dir,
)
from src.artifact_compatibility import (
    canonical_json_bytes,
    server_artifact_binding,
    sha256_bytes,
    write_server_artifact_manifest,
)
from src.run_provenance import write_run_provenance_manifest
from tests.test_run_provenance import (
    runtime_environment,
    write_public_dataset_contract,
)


RUN_IDS = (
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
)


def create_public_snapshot(
    root: Path, name: str, *, metadata: str | None = None
) -> AppManifest:
    """Create and load one valid public-manifest snapshot.

    Parameters
    ----------
    root : pathlib.Path
        Test artifact root.
    name : str
        Unique public artifact directory name.
    metadata : str or None, optional
        Additive manifest metadata used to create a distinct valid snapshot.

    Returns
    -------
    src.app_manifest.AppManifest
        Validated immutable public artifact snapshot.
    """
    public_dir = root / name
    public_dir.mkdir()
    protocol = write_public_dataset_contract(public_dir)
    if metadata is not None:
        path = public_dir / "manifest.json"
        payload = json.loads(path.read_bytes())
        payload["metadata"] = metadata
        path.write_bytes(canonical_json_bytes(payload))
    return load_app_manifest(public_artifact_dir=public_dir, protocol=protocol)


def create_run(
    root: Path,
    run_id: str,
    created_at: str,
    *,
    provenance_app_manifest: AppManifest | None = None,
    artifact_app_manifest: AppManifest | None = None,
) -> Path:
    """Create a complete run candidate for history tests.

    Parameters
    ----------
    root : pathlib.Path
        Artifact history root.
    run_id : str
        Canonical UUID4 directory identity.
    created_at : str
        Deterministic provenance timestamp.
    provenance_app_manifest : src.app_manifest.AppManifest or None, optional
        Public snapshot recorded by run provenance.
    artifact_app_manifest : src.app_manifest.AppManifest or None, optional
        Public snapshot bound by the server artifact manifest.

    Returns
    -------
    pathlib.Path
        Created run directory.
    """
    provenance_manifest = provenance_app_manifest or create_public_snapshot(
        root, f"public-{run_id}"
    )
    artifact_manifest = artifact_app_manifest or provenance_manifest
    run_dir = root / "runs" / run_id
    with (
        patch(
            "src.run_provenance._code_revision",
            return_value={"commit": None, "dirty": None, "source": "unavailable"},
        ),
        patch(
            "src.run_provenance._environment_metadata",
            return_value=runtime_environment(),
        ),
    ):
        write_run_provenance_manifest(
            run_dir,
            {},
            app_manifest=provenance_manifest,
            created_at=created_at,
            run_id=run_id,
        )
    (run_dir / "global_model.keras").write_bytes(f"model-{run_id}".encode())
    (run_dir / "metrics.csv").write_text(
        "round,loss,accuracy\n1,0.5,0.75\n", encoding="utf-8"
    )
    write_server_artifact_manifest(run_dir, app_manifest=artifact_manifest)
    return run_dir


class ArtifactHistoryTests(unittest.TestCase):
    def test_publish_selects_run_and_load_boundary_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            second = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")

            publish_completed_run(root, first)
            publish_completed_run(root, second)

            self.assertEqual(resolve_current_run_dir(root), second.resolve())
            self.assertTrue(first.is_dir())
            (second / "global_model.keras").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum does not match"):
                resolve_current_run_dir(root)

    def test_retention_is_deterministic_and_protects_current_and_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            oldest = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            current = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
            active = create_run(root, RUN_IDS[2], "2026-01-03T00:00:00Z")
            publish_completed_run(root, current)

            deleted = prune_run_history(root, 1, active_run_dir=active)

            self.assertEqual(deleted, [oldest.resolve()])
            self.assertTrue(current.is_dir())
            self.assertTrue(active.is_dir())

    def test_retention_does_not_delete_unvalidated_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            unknown = root / "runs" / "unknown"
            unknown.mkdir()
            (unknown / "keep.txt").write_text("keep", encoding="utf-8")

            prune_run_history(root, 1, active_run_dir=valid)

            self.assertTrue(unknown.is_dir())

    def test_legacy_flat_layout_remains_selectable_without_an_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            self.assertEqual(resolve_current_run_dir(root), root.resolve())

    def test_current_index_cannot_escape_the_runs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "current.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "../outside",
                        "artifact_manifest_checksum": "sha256:" + "0" * 64,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid run_id"):
                resolve_current_run_dir(root)

    def test_current_index_rejects_boolean_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "current.json").write_text(
                json.dumps(
                    {
                        "schema_version": True,
                        "run_id": RUN_IDS[0],
                        "artifact_manifest_checksum": "sha256:" + "0" * 64,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid schema_version"):
                resolve_current_run_dir(root)

    def test_current_index_rejects_duplicate_security_fields(self) -> None:
        checksum = "sha256:" + "0" * 64
        indexes = (
            '{"schema_version":true,"schema_version":1,'
            f'"run_id":"{RUN_IDS[0]}",'
            f'"artifact_manifest_checksum":"{checksum}"}}',
            '{"schema_version":1,"run_id":"../outside",'
            f'"run_id":"{RUN_IDS[0]}",'
            f'"artifact_manifest_checksum":"{checksum}"}}',
            '{"schema_version":1,'
            f'"run_id":"{RUN_IDS[0]}",'
            '"artifact_manifest_checksum":"invalid",'
            f'"artifact_manifest_checksum":"{checksum}"}}',
        )
        for index in indexes:
            with (
                self.subTest(index=index),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                (root / "current.json").write_text(index, encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "invalid current-run index"):
                    resolve_current_run_dir(root)

    def test_current_index_rejects_non_finite_numbers_and_accepts_finite_exponents(
        self,
    ) -> None:
        checksum = "sha256:" + "0" * 64
        for constant in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            with (
                self.subTest(constant=constant),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                (root / "current.json").write_text(
                    '{"schema_version":1,'
                    f'"run_id":"{RUN_IDS[0]}",'
                    f'"artifact_manifest_checksum":"{checksum}",'
                    f'"producer":{constant}}}',
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "invalid current-run index"):
                    resolve_current_run_dir(root)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            publish_completed_run(root, run)
            pointer = root / "current.json"
            valid = pointer.read_text(encoding="utf-8")
            pointer.write_text(
                valid[:-2] + ',\n  "producer": [1e308, -1e308]\n}\n',
                encoding="utf-8",
            )

            self.assertEqual(resolve_current_run_dir(root), run.resolve())

    def test_dangling_current_index_symlink_is_not_legacy_absence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pointer = root / "current.json"
            try:
                pointer.symlink_to("missing-current.json")
            except OSError as error:
                self.skipTest(f"file symlinks are unavailable: {error}")

            with self.assertRaisesRegex(ValueError, "invalid current-run index"):
                resolve_current_run_dir(root)

    def test_retention_refuses_a_symlinked_runs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            outside = Path(tmpdir) / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                (root / "runs").symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            with self.assertRaisesRegex(ValueError, "must not be symlinks"):
                prune_run_history(root, 1)

    def test_retention_aborts_when_current_run_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            oldest = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            current = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
            publish_completed_run(root, current)
            (current / "metrics.csv").write_text("tampered", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checksum does not match"):
                prune_run_history(root, 1)

            self.assertTrue(oldest.is_dir())
            self.assertTrue(current.is_dir())

    def test_retention_aborts_when_current_index_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            oldest = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            newest = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")

            self.assertEqual(prune_run_history(root, 1), [])
            self.assertTrue(oldest.is_dir())
            self.assertTrue(newest.is_dir())

    def test_nonexistent_active_run_does_not_consume_retention_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            newest = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
            publish_completed_run(root, current)

            deleted = prune_run_history(
                root, 2, active_run_dir=root / "runs" / RUN_IDS[2]
            )

            self.assertEqual(deleted, [])
            self.assertTrue(newest.is_dir())

    def test_current_snapshot_keeps_verified_bytes_after_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            publish_completed_run(root, current)

            snapshot = load_current_run_snapshot(root)
            (current / "metrics.csv").write_bytes(b"attacker-controlled")

            self.assertEqual(
                snapshot.files["metrics.csv"],
                b"round,loss,accuracy\n1,0.5,0.75\n",
            )

    def test_finalization_rejects_artifact_symlink_outside_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            outside = root / "outside.keras"
            outside.write_bytes(b"outside")
            (run / "global_model.keras").unlink()
            try:
                (run / "global_model.keras").symlink_to(outside)
            except OSError as error:
                self.skipTest(f"file symlinks are unavailable: {error}")

            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                publish_completed_run(root, run)

    def test_completed_run_cannot_be_finalized_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            publish_completed_run(root, run)

            with self.assertRaisesRegex(ValueError, "cannot be finalized again"):
                publish_completed_run(root, run)

    def test_publication_rejects_distinct_valid_public_manifest_bindings(self) -> None:
        for existing_current in (False, True):
            with (
                self.subTest(existing_current=existing_current),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                first = create_public_snapshot(root, "public-first")
                second = create_public_snapshot(
                    root, "public-second", metadata="distinct"
                )
                if existing_current:
                    current_run = create_run(
                        root,
                        RUN_IDS[0],
                        "2026-01-01T00:00:00Z",
                        provenance_app_manifest=first,
                        artifact_app_manifest=first,
                    )
                    publish_completed_run(root, current_run)
                    current_bytes = (root / "current.json").read_bytes()

                candidate = create_run(
                    root,
                    RUN_IDS[1],
                    "2026-01-02T00:00:00Z",
                    provenance_app_manifest=first,
                    artifact_app_manifest=second,
                )
                with self.assertRaisesRegex(
                    ValueError, "public manifest does not match"
                ):
                    publish_completed_run(root, candidate)

                if existing_current:
                    self.assertEqual(
                        (root / "current.json").read_bytes(), current_bytes
                    )
                else:
                    self.assertFalse((root / "current.json").exists())

    def test_publication_rejects_vocabulary_checksum_binding_mismatch(self) -> None:
        for existing_current in (False, True):
            with (
                self.subTest(existing_current=existing_current),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                public = create_public_snapshot(root, "public")
                if existing_current:
                    current_run = create_run(
                        root,
                        RUN_IDS[0],
                        "2026-01-01T00:00:00Z",
                        provenance_app_manifest=public,
                        artifact_app_manifest=public,
                    )
                    publish_completed_run(root, current_run)
                    current_bytes = (root / "current.json").read_bytes()

                candidate = create_run(
                    root,
                    RUN_IDS[1],
                    "2026-01-02T00:00:00Z",
                    provenance_app_manifest=public,
                    artifact_app_manifest=public,
                )
                provenance_path = candidate / "run_manifest.json"
                provenance = json.loads(provenance_path.read_bytes())
                provenance["dataset"]["checksums"]["vocab.txt"] = "sha256:" + "0" * 64
                provenance_path.write_bytes(canonical_json_bytes(provenance))

                with self.assertRaisesRegex(ValueError, "vocabulary does not match"):
                    publish_completed_run(root, candidate)

                if existing_current:
                    self.assertEqual(
                        (root / "current.json").read_bytes(), current_bytes
                    )
                else:
                    self.assertFalse((root / "current.json").exists())

    def test_current_snapshot_rejects_distinct_valid_public_manifest_bindings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = create_public_snapshot(root, "public-first")
            second = create_public_snapshot(root, "public-second", metadata="distinct")
            run = create_run(
                root,
                RUN_IDS[0],
                "2026-01-01T00:00:00Z",
                provenance_app_manifest=first,
                artifact_app_manifest=first,
            )
            publish_completed_run(root, run)
            artifact_manifest_path = run / "artifact_manifest.json"
            artifact_manifest = json.loads(artifact_manifest_path.read_bytes())
            artifact_manifest["binding"] = server_artifact_binding(second)
            artifact_manifest_path.write_bytes(canonical_json_bytes(artifact_manifest))
            current_path = root / "current.json"
            current = json.loads(current_path.read_bytes())
            current["artifact_manifest_checksum"] = sha256_bytes(
                artifact_manifest_path.read_bytes()
            )
            current_path.write_bytes(canonical_json_bytes(current))

            with self.assertRaisesRegex(ValueError, "public manifest does not match"):
                load_current_run_snapshot(root)

    def test_current_snapshot_rejects_vocabulary_checksum_binding_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            publish_completed_run(root, run)
            provenance_path = run / "run_manifest.json"
            provenance = json.loads(provenance_path.read_bytes())
            provenance["dataset"]["checksums"]["vocab.txt"] = "sha256:" + "0" * 64
            provenance_path.write_bytes(canonical_json_bytes(provenance))
            artifact_manifest_path = run / "artifact_manifest.json"
            artifact_manifest = json.loads(artifact_manifest_path.read_bytes())
            artifact_manifest["checksums"]["run_manifest.json"] = sha256_bytes(
                provenance_path.read_bytes()
            )
            artifact_manifest_path.write_bytes(canonical_json_bytes(artifact_manifest))
            current_path = root / "current.json"
            current = json.loads(current_path.read_bytes())
            current["artifact_manifest_checksum"] = sha256_bytes(
                artifact_manifest_path.read_bytes()
            )
            current_path.write_bytes(canonical_json_bytes(current))

            with self.assertRaisesRegex(ValueError, "vocabulary does not match"):
                load_current_run_snapshot(root)

    def test_finalization_rejects_hostile_provenance_json(self) -> None:
        for hostile_case in (
            "duplicate schema",
            "duplicate run identity",
            "duplicate dataset identity",
            "NaN",
            "Infinity",
            "-Infinity",
            "1e999",
            "-1e999",
        ):
            with (
                self.subTest(hostile_case=hostile_case),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                provenance_path = run / "run_manifest.json"
                valid = provenance_path.read_text(encoding="utf-8")
                if hostile_case == "duplicate schema":
                    document = valid.replace(
                        '"schema_version": 1,',
                        '"schema_version": 0,\n  "schema_version": 1,',
                        1,
                    )
                elif hostile_case == "duplicate run identity":
                    document = valid.replace(
                        f'"run_id": "{RUN_IDS[0]}",',
                        f'"run_id": "{RUN_IDS[1]}",\n  "run_id": "{RUN_IDS[0]}",',
                        1,
                    )
                elif hostile_case == "duplicate dataset identity":
                    document = valid.replace(
                        '    "identity": ',
                        '    "identity": "attacker",\n    "identity": ',
                        1,
                    )
                else:
                    document = valid[:-2] + f',\n  "producer": {hostile_case}\n}}\n'
                provenance_path.write_text(document, encoding="utf-8")

                with self.assertRaisesRegex(
                    ValueError, "invalid run provenance manifest"
                ):
                    publish_completed_run(root, run)

    def test_finalization_rejects_each_frozen_dataset_identity_change(self) -> None:
        identity = expected_train_dataset()
        hostile_values = {
            "id": "attacker/imdb",
            "config": "attacker_config",
            "revision": "0" * 40,
            "datasets_version": "0.0.0",
            "split": "test",
            "rows": 24999,
            "raw_parquet_sha256": "0" * 64,
            "content_sha256": "0" * 64,
        }
        for field, hostile_value in hostile_values.items():
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                provenance_path = run / "run_manifest.json"
                provenance = json.loads(provenance_path.read_bytes())
                provenance["dataset"] = {
                    **provenance["dataset"],
                    "identity": json.dumps(
                        {**identity, field: hostile_value},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "checksums": {"manifest.json": "sha256:" + "0" * 64},
                    "public_manifest": {
                        "filename": "manifest.json",
                        "size_bytes": 1,
                        "checksum": "sha256:" + "0" * 64,
                    },
                    "status": "available",
                }
                provenance_path.write_bytes(canonical_json_bytes(provenance))

                with self.assertRaisesRegex(ValueError, "dataset.identity"):
                    publish_completed_run(root, run)

    def test_finalization_rejects_hostile_private_shard_provenance(self) -> None:
        identity = expected_train_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            provenance_path = run / "run_manifest.json"
            provenance = json.loads(provenance_path.read_bytes())
            public_manifest = {
                "filename": "manifest.json",
                "size_bytes": 1,
                "checksum": "sha256:" + "3" * 64,
            }
            provenance["dataset"] = {
                "identity": json.dumps(
                    identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "checksums": {
                    "manifest.json": public_manifest["checksum"],
                    "vocab.txt": "sha256:" + "4" * 64,
                },
                "public_manifest": public_manifest,
                "status": "available",
                "private_client_shards": {
                    "status": "available",
                    "identity": {
                        "client_id": 0,
                        "dataset": {**identity, "revision": "0" * 40},
                        "source_split": "train",
                        "row_identity": ("train:{zero_based_official_split_row_index}"),
                        "sample_count": 2,
                        "label_histogram": {"0": 1, "1": 1},
                        "public_manifest": public_manifest,
                    },
                    "checksums": {
                        "client_metadata.json": "sha256:" + "1" * 64,
                        "reviews.jsonl": "sha256:" + "2" * 64,
                    },
                },
            }
            provenance_path.write_bytes(canonical_json_bytes(provenance))

            with self.assertRaisesRegex(ValueError, "private_client_shards"):
                publish_completed_run(root, run)

    def test_finalization_rejects_mutation_after_provenance_validation(self) -> None:
        from src.run_provenance import load_run_provenance_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            provenance_path = run / "run_manifest.json"
            mutated = False

            def validate_then_mutate(path, *, manifest_bytes=None):
                nonlocal mutated
                payload = load_run_provenance_manifest(
                    path, manifest_bytes=manifest_bytes
                )
                if not mutated:
                    mutated = True
                    provenance_path.write_bytes(b'{"attacker":true}\n')
                return payload

            with (
                patch(
                    "src.artifact_history.load_run_provenance_manifest",
                    side_effect=validate_then_mutate,
                ),
                self.assertRaisesRegex(ValueError, "changed during finalization"),
            ):
                publish_completed_run(root, run)

            self.assertFalse((root / "current.json").exists())

    def test_finalization_rejects_same_content_inode_replacement(self) -> None:
        from src.run_provenance import load_run_provenance_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            provenance_path = run / "run_manifest.json"
            replaced = False

            def validate_then_replace(path, *, manifest_bytes=None):
                nonlocal replaced
                payload = load_run_provenance_manifest(
                    path, manifest_bytes=manifest_bytes
                )
                if not replaced:
                    replaced = True
                    replacement = run / "replacement.tmp"
                    replacement.write_bytes(provenance_path.read_bytes())
                    os.replace(replacement, provenance_path)
                return payload

            with (
                patch(
                    "src.artifact_history.load_run_provenance_manifest",
                    side_effect=validate_then_replace,
                ),
                self.assertRaisesRegex(ValueError, "changed during finalization"),
            ):
                publish_completed_run(root, run)

            self.assertFalse((root / "current.json").exists())

    def test_current_snapshot_validates_exact_completed_provenance_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            publish_completed_run(root, run)
            provenance_path = run / "run_manifest.json"
            provenance_path.write_bytes(b'{"attacker":true}\n')
            artifact_manifest_path = run / "artifact_manifest.json"
            artifact_manifest = json.loads(artifact_manifest_path.read_bytes())
            artifact_manifest["sizes"]["run_manifest.json"] = (
                provenance_path.stat().st_size
            )
            artifact_manifest["checksums"]["run_manifest.json"] = (
                "sha256:" + hashlib.sha256(provenance_path.read_bytes()).hexdigest()
            )
            artifact_manifest_path.write_bytes(canonical_json_bytes(artifact_manifest))
            current_path = root / "current.json"
            current = json.loads(current_path.read_bytes())
            current["artifact_manifest_checksum"] = (
                "sha256:"
                + hashlib.sha256(artifact_manifest_path.read_bytes()).hexdigest()
            )
            current_path.write_bytes(canonical_json_bytes(current))

            with self.assertRaisesRegex(ValueError, "run provenance manifest"):
                load_current_run_snapshot(root)

    def test_finalization_recovers_after_current_index_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")

            with (
                patch(
                    "src.artifact_history.write_json_atomically",
                    side_effect=OSError("simulated current-index write failure"),
                ),
                self.assertRaisesRegex(OSError, "current-index write failure"),
            ):
                publish_completed_run(root, run)

            manifest_path = run / "artifact_manifest.json"
            manifest_bytes = manifest_path.read_bytes()
            metrics_path = run / "metrics.csv"
            metrics_bytes = metrics_path.read_bytes()
            self.assertFalse((root / "current.json").exists())

            metrics_path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum does not match"):
                publish_completed_run(root, run)
            metrics_path.write_bytes(metrics_bytes)
            publish_completed_run(root, run)

            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)
            self.assertEqual(resolve_current_run_dir(root), run.resolve())
            with self.assertRaisesRegex(ValueError, "cannot be finalized again"):
                publish_completed_run(root, run)

    def test_concurrent_finalization_has_exactly_one_publisher(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            start = threading.Barrier(2)
            first_write = threading.Event()
            release_write = threading.Event()
            errors: list[BaseException] = []

            from src import artifact_compatibility

            write_json_atomically = artifact_compatibility.write_json_atomically

            def delayed_write(path, payload, *, overwrite=True):
                if (
                    path.name == "artifact_manifest.json"
                    and payload.get("lifecycle") == "complete"
                    and not first_write.is_set()
                ):
                    first_write.set()
                    release_write.wait(timeout=2)
                return write_json_atomically(path, payload, overwrite=overwrite)

            def publish() -> Path | None:
                start.wait()
                try:
                    return publish_completed_run(root, run)
                except BaseException as error:
                    errors.append(error)
                    return None

            with (
                patch(
                    "src.artifact_compatibility.write_json_atomically",
                    side_effect=delayed_write,
                ),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                futures = [executor.submit(publish) for _ in range(2)]
                self.assertTrue(first_write.wait(timeout=2))
                time.sleep(0.05)
                release_write.set()
                results = [future.result(timeout=2) for future in futures]

            self.assertEqual(sum(result is not None for result in results), 1)
            self.assertEqual(len(errors), 1)
            self.assertRegex(str(errors[0]), "cannot be finalized again")
            manifest = json.loads(
                (run / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["lifecycle"], "complete")
            self.assertEqual(resolve_current_run_dir(root), run.resolve())

    def test_retention_refuses_hardlinked_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            current = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
            publish_completed_run(root, current)
            try:
                os.link(candidate / "run_manifest.json", root / "provenance-copy.json")
            except OSError as error:
                self.skipTest(f"hard links are unavailable: {error}")

            self.assertEqual(prune_run_history(root, 1), [])
            self.assertTrue(candidate.is_dir())
            self.assertTrue(current.is_dir())


if __name__ == "__main__":
    unittest.main()
