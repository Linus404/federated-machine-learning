import base64
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.app_manifest import AppManifest, expected_train_dataset, load_app_manifest
from src.artifact_history import (
    load_current_run_snapshot,
    prune_run_history,
    publish_completed_run as _publish_completed_run,
    resolve_current_run_dir,
)
from src.artifact_compatibility import (
    canonical_json_bytes,
    load_server_artifact_snapshot,
    server_artifact_binding,
    sha256_bytes,
    write_server_artifact_manifest,
)
from src.contracts import DEFAULT_SPLIT_SEED, DEFAULT_VALIDATION_SEED
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
_RUN_PUBLIC_MANIFESTS: dict[Path, AppManifest] = {}
_TEST_VOCABULARY = b"\n[UNK]\ngood\nbad\n"
_TEST_VOCABULARY_SHA256 = hashlib.sha256(_TEST_VOCABULARY).hexdigest()
_TEST_PROTOCOL = {
    "dataset": {
        **{
            key: expected_train_dataset()[key]
            for key in ("id", "config", "revision", "datasets_version")
        },
        "splits": {
            "train": {
                key: expected_train_dataset()[key]
                for key in ("rows", "raw_parquet_sha256", "content_sha256")
            }
        },
    },
    "preprocessing": {
        "vocabulary_size": 4,
        "max_tokens": 4,
        "output_sequence_length": 500,
        "vocabulary_sha256": _TEST_VOCABULARY_SHA256,
    },
    "model": {
        "vocabulary_size": 4,
        "sequence_length": 500,
        "embedding_dimension": 100,
    },
}


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
    _RUN_PUBLIC_MANIFESTS[run_dir] = provenance_manifest
    return run_dir


def publish_completed_run(root: Path, run_dir: Path) -> Path:
    """Publish a test run with the public snapshot used for its provenance.

    Parameters
    ----------
    root : pathlib.Path
        Artifact history root.
    run_dir : pathlib.Path
        Candidate run directory created by :func:`create_run`.

    Returns
    -------
    pathlib.Path
        Published current-run index path.
    """
    return _publish_completed_run(
        root, run_dir, app_manifest=_RUN_PUBLIC_MANIFESTS[run_dir]
    )


def rewrite_current_artifact_manifest(root: Path, run_dir: Path) -> None:
    """Rebind the current index to a deliberately rewritten artifact manifest.

    Parameters
    ----------
    root : pathlib.Path
        Artifact history root.
    run_dir : pathlib.Path
        Current completed run whose artifact manifest was rewritten.

    Returns
    -------
    None
    """
    manifest_path = run_dir / "artifact_manifest.json"
    current_path = root / "current.json"
    current = json.loads(current_path.read_bytes())
    current["artifact_manifest_checksum"] = sha256_bytes(manifest_path.read_bytes())
    current_path.write_bytes(canonical_json_bytes(current))


class ArtifactHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        """Use the small frozen public contract created by history fixtures.

        Returns
        -------
        None
        """
        protocol = patch(
            "src.app_manifest.load_scientific_protocol",
            return_value=_TEST_PROTOCOL,
        )
        protocol.start()
        self.addCleanup(protocol.stop)

    def test_publication_rejects_coordinated_false_public_values(self) -> None:
        for field in (
            "public_manifest_checksum",
            "vocabulary_checksum",
            "model_dimensions",
        ):
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                artifact_path = run / "artifact_manifest.json"
                artifact = json.loads(artifact_path.read_bytes())
                provenance_path = run / "run_manifest.json"
                provenance = json.loads(provenance_path.read_bytes())
                if field == "public_manifest_checksum":
                    invented = "sha256:" + "0" * 64
                    artifact["binding"][field] = invented
                    provenance["dataset"]["checksums"]["manifest.json"] = invented
                    provenance["dataset"]["public_manifest"]["checksum"] = invented
                elif field == "vocabulary_checksum":
                    invented = "sha256:" + "1" * 64
                    artifact["binding"][field] = invented
                    provenance["dataset"]["checksums"]["vocab.txt"] = invented
                else:
                    artifact["binding"][field]["embedding_dim"] = 1
                artifact_path.write_bytes(canonical_json_bytes(artifact))
                provenance_path.write_bytes(canonical_json_bytes(provenance))

                with self.assertRaisesRegex(ValueError, "does not match"):
                    publish_completed_run(root, run)

                self.assertFalse((root / "current.json").exists())

    def test_publication_retains_and_manifests_legitimate_writer_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            (run / "client_metrics.csv").write_text(
                "round,client_id,loss,accuracy,samples\n",
                encoding="utf-8",
            )

            publish_completed_run(root, run)

            manifest = json.loads((run / "artifact_manifest.json").read_bytes())
            self.assertIn("client_metrics.csv", manifest["checksums"])
            self.assertEqual(
                {entry.name for entry in run.iterdir()},
                {"artifact_manifest.json", *manifest["checksums"]},
            )

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix sockets require POSIX")
    def test_publication_rejects_non_regular_extra_entries(self) -> None:
        for entry_type in ("hardlink", "symlink", "directory", "fifo", "socket"):
            if entry_type == "fifo" and not hasattr(os, "mkfifo"):
                continue
            with (
                self.subTest(entry_type=entry_type),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                extra = run / "extra"
                open_socket = None
                if entry_type == "hardlink":
                    os.link(run / "metrics.csv", extra)
                elif entry_type == "symlink":
                    extra.symlink_to(run / "metrics.csv")
                elif entry_type == "directory":
                    extra.mkdir()
                elif entry_type == "fifo":
                    os.mkfifo(extra)
                else:
                    open_socket = socket.socket(socket.AF_UNIX)
                    open_socket.bind(str(extra))
                self.addCleanup(open_socket.close if open_socket else lambda: None)

                with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                    publish_completed_run(root, run)

                self.assertFalse((root / "current.json").exists())

    def test_publication_retry_rejects_missing_or_changed_public_evidence(
        self,
    ) -> None:
        for filename in ("manifest.json", "vocab.txt"):
            for mutation in ("missing", "changed"):
                with (
                    self.subTest(filename=filename, mutation=mutation),
                    tempfile.TemporaryDirectory() as tmpdir,
                ):
                    root = Path(tmpdir)
                    run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                    with (
                        patch(
                            "src.artifact_history._write_current_index_at",
                            side_effect=OSError("current-index failure"),
                        ),
                        self.assertRaisesRegex(OSError, "current-index failure"),
                    ):
                        publish_completed_run(root, run)
                    evidence = run / filename
                    if mutation == "missing":
                        evidence.unlink()
                    else:
                        evidence.write_bytes(evidence.read_bytes() + b"changed")

                    with self.assertRaisesRegex(
                        ValueError, "inventory|checksum does not match"
                    ):
                        publish_completed_run(root, run)

                    self.assertFalse((root / "current.json").exists())

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix sockets require POSIX")
    def test_publication_retry_rejects_every_unmanifested_entry_type(self) -> None:
        for entry_type in (
            "regular",
            "hardlink",
            "symlink",
            "directory",
            "fifo",
            "socket",
        ):
            if entry_type == "fifo" and not hasattr(os, "mkfifo"):
                continue
            with (
                self.subTest(entry_type=entry_type),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                with (
                    patch(
                        "src.artifact_history._write_current_index_at",
                        side_effect=OSError("current-index failure"),
                    ),
                    self.assertRaisesRegex(OSError, "current-index failure"),
                ):
                    publish_completed_run(root, run)
                extra = run / "extra"
                open_socket = None
                if entry_type == "regular":
                    extra.write_bytes(b"extra")
                elif entry_type == "hardlink":
                    os.link(run / "metrics.csv", extra)
                elif entry_type == "symlink":
                    extra.symlink_to(run / "metrics.csv")
                elif entry_type == "directory":
                    extra.mkdir()
                elif entry_type == "fifo":
                    os.mkfifo(extra)
                else:
                    open_socket = socket.socket(socket.AF_UNIX)
                    open_socket.bind(str(extra))
                self.addCleanup(open_socket.close if open_socket else lambda: None)

                with self.assertRaisesRegex(ValueError, "inventory"):
                    publish_completed_run(root, run)

                self.assertFalse((root / "current.json").exists())

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
                    ValueError, "public-manifest.*does not match"
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

                with self.assertRaisesRegex(ValueError, "retained public evidence"):
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

            with self.assertRaisesRegex(ValueError, "public-manifest.*does not match"):
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

            with self.assertRaisesRegex(ValueError, "retained public evidence"):
                load_current_run_snapshot(root)

    def test_current_snapshot_rejects_coordinated_false_manifest_dimensions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            publish_completed_run(root, run)
            retained_manifest_path = run / "manifest.json"
            retained_manifest = json.loads(retained_manifest_path.read_bytes())
            retained_manifest["embedding_dim"] = 1
            retained_manifest_bytes = canonical_json_bytes(retained_manifest)
            retained_manifest_path.write_bytes(retained_manifest_bytes)

            provenance_path = run / "run_manifest.json"
            provenance = json.loads(provenance_path.read_bytes())
            manifest_checksum = sha256_bytes(retained_manifest_bytes)
            provenance["dataset"]["checksums"]["manifest.json"] = manifest_checksum
            provenance["dataset"]["public_manifest"] = {
                "filename": "manifest.json",
                "size_bytes": len(retained_manifest_bytes),
                "checksum": manifest_checksum,
            }
            provenance_bytes = canonical_json_bytes(provenance)
            provenance_path.write_bytes(provenance_bytes)

            artifact_path = run / "artifact_manifest.json"
            artifact = json.loads(artifact_path.read_bytes())
            artifact["binding"]["public_manifest_checksum"] = manifest_checksum
            artifact["binding"]["model_dimensions"]["embedding_dim"] = 1
            artifact["sizes"]["manifest.json"] = len(retained_manifest_bytes)
            artifact["checksums"]["manifest.json"] = manifest_checksum
            artifact["sizes"]["run_manifest.json"] = len(provenance_bytes)
            artifact["checksums"]["run_manifest.json"] = sha256_bytes(provenance_bytes)
            artifact_path.write_bytes(canonical_json_bytes(artifact))
            rewrite_current_artifact_manifest(root, run)

            with self.assertRaisesRegex(ValueError, "protocol dimensions"):
                load_current_run_snapshot(root)

    def test_current_snapshot_rejects_coordinated_false_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            publish_completed_run(root, run)
            retained_vocabulary = b"\n[UNK]\nevil\nbad\n"
            vocabulary_checksum = sha256_bytes(retained_vocabulary)
            (run / "vocab.txt").write_bytes(retained_vocabulary)
            retained_manifest_path = run / "manifest.json"
            retained_manifest = json.loads(retained_manifest_path.read_bytes())
            retained_manifest["vocabulary"]["sha256"] = vocabulary_checksum[7:]
            retained_manifest["vocabulary"]["size_bytes"] = len(retained_vocabulary)
            retained_manifest_bytes = canonical_json_bytes(retained_manifest)
            retained_manifest_path.write_bytes(retained_manifest_bytes)
            manifest_checksum = sha256_bytes(retained_manifest_bytes)

            provenance_path = run / "run_manifest.json"
            provenance = json.loads(provenance_path.read_bytes())
            provenance["dataset"]["checksums"] = {
                "manifest.json": manifest_checksum,
                "vocab.txt": vocabulary_checksum,
            }
            provenance["dataset"]["public_manifest"] = {
                "filename": "manifest.json",
                "size_bytes": len(retained_manifest_bytes),
                "checksum": manifest_checksum,
            }
            provenance_bytes = canonical_json_bytes(provenance)
            provenance_path.write_bytes(provenance_bytes)

            artifact_path = run / "artifact_manifest.json"
            artifact = json.loads(artifact_path.read_bytes())
            artifact["binding"]["public_manifest_checksum"] = manifest_checksum
            artifact["binding"]["vocabulary_checksum"] = vocabulary_checksum
            for name, content in (
                ("manifest.json", retained_manifest_bytes),
                ("vocab.txt", retained_vocabulary),
                ("run_manifest.json", provenance_bytes),
            ):
                artifact["sizes"][name] = len(content)
                artifact["checksums"][name] = sha256_bytes(content)
            artifact_path.write_bytes(canonical_json_bytes(artifact))
            rewrite_current_artifact_manifest(root, run)

            with self.assertRaisesRegex(ValueError, "frozen protocol"):
                load_current_run_snapshot(root)

    def test_completed_loaders_reject_missing_or_changed_public_evidence(self) -> None:
        for loader_name in ("current", "historical"):
            for filename in ("manifest.json", "vocab.txt"):
                for mutation in ("missing", "changed"):
                    with (
                        self.subTest(
                            loader=loader_name,
                            filename=filename,
                            mutation=mutation,
                        ),
                        tempfile.TemporaryDirectory() as tmpdir,
                    ):
                        root = Path(tmpdir)
                        run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                        publish_completed_run(root, run)
                        evidence = run / filename
                        if mutation == "missing":
                            evidence.unlink()
                        else:
                            evidence.write_bytes(evidence.read_bytes() + b"changed")

                        with self.assertRaisesRegex(
                            ValueError, "inventory|checksum does not match"
                        ):
                            if loader_name == "current":
                                load_current_run_snapshot(root)
                            else:
                                load_server_artifact_snapshot(run)

    def test_completed_loaders_reject_rechecksummed_provenance_tampering(self) -> None:
        for loader_name in ("current", "historical"):
            for mutation in ("wrong run_id", "wrong public checksum"):
                with (
                    self.subTest(loader=loader_name, mutation=mutation),
                    tempfile.TemporaryDirectory() as tmpdir,
                ):
                    root = Path(tmpdir)
                    run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                    publish_completed_run(root, run)
                    provenance_path = run / "run_manifest.json"
                    provenance = json.loads(provenance_path.read_bytes())
                    if mutation == "wrong run_id":
                        provenance["run_id"] = RUN_IDS[1]
                    else:
                        provenance["dataset"]["checksums"]["vocab.txt"] = (
                            "sha256:" + "0" * 64
                        )
                    provenance_bytes = canonical_json_bytes(provenance)
                    provenance_path.write_bytes(provenance_bytes)
                    artifact_path = run / "artifact_manifest.json"
                    artifact = json.loads(artifact_path.read_bytes())
                    artifact["sizes"]["run_manifest.json"] = len(provenance_bytes)
                    artifact["checksums"]["run_manifest.json"] = sha256_bytes(
                        provenance_bytes
                    )
                    artifact_path.write_bytes(canonical_json_bytes(artifact))
                    rewrite_current_artifact_manifest(root, run)

                    with self.assertRaisesRegex(
                        ValueError, "run_id|retained public evidence"
                    ):
                        if loader_name == "current":
                            load_current_run_snapshot(root)
                        else:
                            load_server_artifact_snapshot(run)

    def test_completed_loaders_require_canonical_code_defaults(self) -> None:
        code_default_mutations = {
            "changed value": {
                "client_validation_split": DEFAULT_VALIDATION_SEED + 1,
                "data_partition": DEFAULT_SPLIT_SEED,
            },
            "missing key": {"data_partition": DEFAULT_SPLIT_SEED},
            "extra key": {
                "client_validation_split": DEFAULT_VALIDATION_SEED,
                "data_partition": DEFAULT_SPLIT_SEED,
                "unexpected": 1,
            },
            "boolean substitution": {
                "client_validation_split": True,
                "data_partition": DEFAULT_SPLIT_SEED,
            },
            "type substitution": {
                "client_validation_split": str(DEFAULT_VALIDATION_SEED),
                "data_partition": DEFAULT_SPLIT_SEED,
            },
        }
        for loader_name in ("current", "historical"):
            for mutation, code_defaults in code_default_mutations.items():
                with (
                    self.subTest(loader=loader_name, mutation=mutation),
                    tempfile.TemporaryDirectory() as tmpdir,
                ):
                    root = Path(tmpdir)
                    run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                    publish_completed_run(root, run)
                    loader = (
                        load_current_run_snapshot
                        if loader_name == "current"
                        else load_server_artifact_snapshot
                    )
                    self.assertEqual(
                        loader(root if loader_name == "current" else run).directory,
                        run.resolve(),
                    )

                    provenance_path = run / "run_manifest.json"
                    provenance = json.loads(provenance_path.read_bytes())
                    provenance["seeds"]["code_defaults"] = code_defaults
                    provenance_bytes = canonical_json_bytes(provenance)
                    provenance_path.write_bytes(provenance_bytes)
                    artifact_path = run / "artifact_manifest.json"
                    artifact = json.loads(artifact_path.read_bytes())
                    artifact["sizes"]["run_manifest.json"] = len(provenance_bytes)
                    artifact["checksums"]["run_manifest.json"] = sha256_bytes(
                        provenance_bytes
                    )
                    artifact_path.write_bytes(canonical_json_bytes(artifact))
                    rewrite_current_artifact_manifest(root, run)

                    with self.assertRaisesRegex(ValueError, "seeds.code_defaults"):
                        loader(root if loader_name == "current" else run)

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix sockets require POSIX")
    def test_completed_loaders_reject_every_extra_entry_type(self) -> None:
        for loader_name in ("current", "historical"):
            for entry_type in (
                "regular",
                "hardlink",
                "symlink",
                "directory",
                "fifo",
                "socket",
            ):
                if entry_type == "fifo" and not hasattr(os, "mkfifo"):
                    continue
                with (
                    self.subTest(loader=loader_name, entry_type=entry_type),
                    tempfile.TemporaryDirectory() as tmpdir,
                ):
                    root = Path(tmpdir)
                    run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                    publish_completed_run(root, run)
                    extra = run / "extra"
                    open_socket = None
                    if entry_type == "regular":
                        extra.write_bytes(b"extra")
                    elif entry_type == "hardlink":
                        os.link(run / "metrics.csv", extra)
                    elif entry_type == "symlink":
                        extra.symlink_to(run / "metrics.csv")
                    elif entry_type == "directory":
                        extra.mkdir()
                    elif entry_type == "fifo":
                        os.mkfifo(extra)
                    else:
                        open_socket = socket.socket(socket.AF_UNIX)
                        open_socket.bind(str(extra))
                    self.addCleanup(open_socket.close if open_socket else lambda: None)

                    with self.assertRaisesRegex(ValueError, "inventory"):
                        if loader_name == "current":
                            load_current_run_snapshot(root)
                        else:
                            load_server_artifact_snapshot(run)

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
                    "src.artifact_history._write_current_index_at",
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

    def test_publication_rejects_root_and_runs_directory_redirection(self) -> None:
        for redirected in ("root", "runs"):
            with (
                self.subTest(redirected=redirected),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                base = Path(tmpdir)
                root = base / "artifacts"
                root.mkdir()
                run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                root_identity = (root.stat().st_dev, root.stat().st_ino)
                runs_identity = (
                    (root / "runs").stat().st_dev,
                    (root / "runs").stat().st_ino,
                )
                parked = base / f"parked-{redirected}"
                redirected_once = False
                real_fsync = os.fsync

                def redirect_after_barrier(descriptor):
                    nonlocal redirected_once
                    result = real_fsync(descriptor)
                    identity = (
                        os.fstat(descriptor).st_dev,
                        os.fstat(descriptor).st_ino,
                    )
                    if redirected_once or identity != runs_identity:
                        return result
                    redirected_once = True
                    if redirected == "root":
                        root.rename(parked)
                        root.mkdir()
                        (root / "runs").mkdir()
                    else:
                        (root / "runs").rename(parked)
                        (root / "runs").mkdir()
                    return result

                with (
                    patch("src.artifact_history.os.fsync", redirect_after_barrier),
                    self.assertRaisesRegex(ValueError, "changed during finalization"),
                ):
                    publish_completed_run(root, run)

                self.assertTrue(redirected_once)
                parked_identity = (parked.stat().st_dev, parked.stat().st_ino)
                self.assertEqual(
                    parked_identity,
                    root_identity if redirected == "root" else runs_identity,
                )
                self.assertFalse((root / "current.json").exists())
                self.assertFalse((parked / "current.json").exists())

    def test_publication_rejects_artifact_mutation_after_runs_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            runs_stat = (root / "runs").stat()
            runs_identity = (runs_stat.st_dev, runs_stat.st_ino)
            metrics_path = run / "metrics.csv"
            mutated = False
            real_fsync = os.fsync

            def mutate_after_barrier(descriptor):
                nonlocal mutated
                result = real_fsync(descriptor)
                file_stat = os.fstat(descriptor)
                if (
                    not mutated
                    and (file_stat.st_dev, file_stat.st_ino) == runs_identity
                ):
                    mutated = True
                    metrics_path.write_bytes(b"attacker-controlled")
                return result

            with (
                patch("src.artifact_history.os.fsync", mutate_after_barrier),
                self.assertRaisesRegex(ValueError, "changed during finalization"),
            ):
                publish_completed_run(root, run)

            self.assertTrue(mutated)
            self.assertFalse((root / "current.json").exists())

    def test_publication_rejects_run_directory_replacement_after_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            runs_stat = (root / "runs").stat()
            runs_identity = (runs_stat.st_dev, runs_stat.st_ino)
            parked_run = root / "runs" / "parked-run"
            replaced = False
            real_fsync = os.fsync

            def replace_after_barrier(descriptor):
                nonlocal replaced
                result = real_fsync(descriptor)
                file_stat = os.fstat(descriptor)
                if (
                    not replaced
                    and (file_stat.st_dev, file_stat.st_ino) == runs_identity
                ):
                    replaced = True
                    run.rename(parked_run)
                    run.mkdir()
                return result

            with (
                patch("src.artifact_history.os.fsync", replace_after_barrier),
                self.assertRaisesRegex(ValueError, "changed during finalization"),
            ):
                publish_completed_run(root, run)

            self.assertTrue(replaced)
            self.assertFalse((root / "current.json").exists())

    def test_exact_retry_recovers_after_post_replacement_root_fsync_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            root_stat = root.stat()
            root_identity = (root_stat.st_dev, root_stat.st_ino)
            failed = False
            real_fsync = os.fsync

            def fail_pointer_barrier(descriptor):
                nonlocal failed
                file_stat = os.fstat(descriptor)
                if (
                    not failed
                    and (file_stat.st_dev, file_stat.st_ino) == root_identity
                    and (root / "current.json").exists()
                ):
                    failed = True
                    raise OSError("simulated artifact-root fsync failure")
                return real_fsync(descriptor)

            with (
                patch("src.artifact_history.os.fsync", fail_pointer_barrier),
                self.assertRaisesRegex(OSError, "artifact-root fsync failure"),
            ):
                publish_completed_run(root, run)

            current_bytes = (root / "current.json").read_bytes()
            self.assertTrue(failed)
            self.assertEqual(publish_completed_run(root, run), root / "current.json")
            self.assertEqual((root / "current.json").read_bytes(), current_bytes)
            self.assertEqual(resolve_current_run_dir(root), run.resolve())

    def test_successful_exact_recovery_does_not_allow_repeated_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            root_identity = (root.stat().st_dev, root.stat().st_ino)
            failed = False
            real_fsync = os.fsync

            def fail_pointer_barrier(descriptor):
                nonlocal failed
                file_stat = os.fstat(descriptor)
                if (
                    not failed
                    and (file_stat.st_dev, file_stat.st_ino) == root_identity
                    and (root / "current.json").exists()
                ):
                    failed = True
                    raise OSError("simulated artifact-root fsync failure")
                return real_fsync(descriptor)

            with (
                patch("src.artifact_history.os.fsync", fail_pointer_barrier),
                self.assertRaises(OSError),
            ):
                publish_completed_run(root, run)
            publish_completed_run(root, run)

            with self.assertRaisesRegex(ValueError, "cannot be finalized again"):
                publish_completed_run(root, run)

    def test_exact_retry_rejects_mismatched_existing_pointer(self) -> None:
        for mismatch in ("run_id", "artifact_manifest_checksum"):
            with (
                self.subTest(mismatch=mismatch),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                root_identity = (root.stat().st_dev, root.stat().st_ino)
                failed = False
                real_fsync = os.fsync

                def fail_pointer_barrier(descriptor):
                    nonlocal failed
                    file_stat = os.fstat(descriptor)
                    if (
                        not failed
                        and (file_stat.st_dev, file_stat.st_ino) == root_identity
                        and (root / "current.json").exists()
                    ):
                        failed = True
                        raise OSError("simulated artifact-root fsync failure")
                    return real_fsync(descriptor)

                with (
                    patch("src.artifact_history.os.fsync", fail_pointer_barrier),
                    self.assertRaises(OSError),
                ):
                    publish_completed_run(root, run)

                current_path = root / "current.json"
                current = json.loads(current_path.read_bytes())
                current[mismatch] = (
                    RUN_IDS[1] if mismatch == "run_id" else "sha256:" + "0" * 64
                )
                current_path.write_bytes(canonical_json_bytes(current))

                with self.assertRaisesRegex(ValueError, "cannot be finalized again"):
                    publish_completed_run(root, run)

    def test_pending_retry_restores_publication_over_exact_previous_pointer(
        self,
    ) -> None:
        from src import artifact_history

        for failure in (
            "completed files",
            "run directory",
            "runs directory",
            "pointer write",
            "pointer permissions",
            "pointer file fsync",
            "pointer rename",
        ):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                previous = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                candidate = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
                publish_completed_run(root, previous)
                previous_bytes = (root / "current.json").read_bytes()
                run_identity = (candidate.stat().st_dev, candidate.stat().st_ino)
                runs_identity = (
                    (root / "runs").stat().st_dev,
                    (root / "runs").stat().st_ino,
                )
                real_sync_files = artifact_history._sync_retained_files
                real_sync_directory = artifact_history._sync_retained_directory
                real_write_all = artifact_history._write_all
                real_fchmod = os.fchmod
                real_fsync = os.fsync
                real_rename = os.rename
                sync_files_calls = 0

                def is_pointer_temporary(descriptor: int) -> bool:
                    return ".current.json." in os.readlink(
                        f"/proc/self/fd/{descriptor}"
                    )

                def fail_sync_files(inventory):
                    nonlocal sync_files_calls
                    sync_files_calls += 1
                    if failure == "completed files" and sync_files_calls == 2:
                        raise OSError("injected completed-files failure")
                    return real_sync_files(inventory)

                def fail_sync_directory(directory):
                    identity = (directory.device, directory.inode)
                    if failure == "run directory" and identity == run_identity:
                        raise OSError("injected run-directory failure")
                    if failure == "runs directory" and identity == runs_identity:
                        raise OSError("injected runs-directory failure")
                    return real_sync_directory(directory)

                def fail_write(descriptor, content):
                    if failure == "pointer write" and is_pointer_temporary(descriptor):
                        raise OSError("injected pointer-write failure")
                    return real_write_all(descriptor, content)

                def fail_fchmod(descriptor, mode):
                    if failure == "pointer permissions" and is_pointer_temporary(
                        descriptor
                    ):
                        raise OSError("injected pointer-permissions failure")
                    return real_fchmod(descriptor, mode)

                def fail_fsync(descriptor):
                    if (
                        failure == "pointer file fsync"
                        and stat.S_ISREG(os.fstat(descriptor).st_mode)
                        and is_pointer_temporary(descriptor)
                    ):
                        raise OSError("injected pointer-fsync failure")
                    return real_fsync(descriptor)

                def fail_rename(source, destination, **kwargs):
                    if failure == "pointer rename" and destination == "current.json":
                        raise OSError("injected pointer-rename failure")
                    return real_rename(source, destination, **kwargs)

                with (
                    patch.object(
                        artifact_history,
                        "require_secure_artifact_platform",
                        return_value=None,
                    ),
                    patch(
                        "src.artifact_compatibility.require_secure_artifact_platform",
                        return_value=None,
                    ),
                    patch.object(
                        artifact_history,
                        "_sync_retained_files",
                        side_effect=fail_sync_files,
                    ),
                    patch.object(
                        artifact_history,
                        "_sync_retained_directory",
                        side_effect=fail_sync_directory,
                    ),
                    patch.object(
                        artifact_history, "_write_all", side_effect=fail_write
                    ),
                    patch.object(os, "fchmod", side_effect=fail_fchmod),
                    patch.object(os, "fsync", side_effect=fail_fsync),
                    patch.object(os, "rename", side_effect=fail_rename),
                    self.assertRaisesRegex(OSError, "injected"),
                ):
                    publish_completed_run(root, candidate)

                self.assertEqual((root / "current.json").read_bytes(), previous_bytes)
                state_path = root / f".{RUN_IDS[1]}.finalize.state"
                state = json.loads(state_path.read_bytes())
                previous_binding = state["previous_pointer"]
                self.assertEqual(state["state"], "pending")
                self.assertEqual(previous_binding["run_id"], RUN_IDS[0])
                self.assertEqual(
                    previous_binding["bytes_base64"],
                    base64.b64encode(previous_bytes).decode("ascii"),
                )
                self.assertEqual(
                    previous_binding["bytes_checksum"], sha256_bytes(previous_bytes)
                )

                publish_completed_run(root, candidate)
                self.assertEqual(resolve_current_run_dir(root), candidate.resolve())
                with self.assertRaisesRegex(ValueError, "cannot be finalized again"):
                    publish_completed_run(root, candidate)

    def test_pending_publication_recovers_in_a_new_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            previous = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            candidate = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
            publish_completed_run(root, previous)

            with (
                patch(
                    "src.artifact_history._write_current_index_at",
                    side_effect=OSError("injected pointer failure"),
                ),
                self.assertRaisesRegex(OSError, "pointer failure"),
            ):
                publish_completed_run(root, candidate)

            script = """
import sys
from pathlib import Path
from unittest.mock import patch

from src.app_manifest import load_app_manifest
from src.artifact_history import publish_completed_run
from tests.test_artifact_history import _TEST_PROTOCOL

root = Path(sys.argv[1])
run = Path(sys.argv[2])
with patch("src.app_manifest.load_scientific_protocol", return_value=_TEST_PROTOCOL):
    manifest = load_app_manifest(
        public_artifact_dir=root / f"public-{run.name}",
        protocol=_TEST_PROTOCOL,
    )
    publish_completed_run(root, run, app_manifest=manifest)
"""
            retry = subprocess.run(
                [sys.executable, "-c", script, str(root), str(candidate)],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(resolve_current_run_dir(root), candidate.resolve())
            with self.assertRaisesRegex(ValueError, "cannot be finalized again"):
                publish_completed_run(root, candidate)

    def test_pending_retry_rejects_divergent_pointer_and_preserves_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            previous = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            candidate = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
            divergent = create_run(root, RUN_IDS[2], "2026-01-03T00:00:00Z")
            publish_completed_run(root, previous)
            with (
                patch(
                    "src.artifact_history._write_current_index_at",
                    side_effect=OSError("injected pointer failure"),
                ),
                self.assertRaises(OSError),
            ):
                publish_completed_run(root, candidate)
            publish_completed_run(root, divergent)
            divergent_bytes = (root / "current.json").read_bytes()

            with self.assertRaisesRegex(ValueError, "cannot be finalized again"):
                publish_completed_run(root, candidate)
            self.assertEqual((root / "current.json").read_bytes(), divergent_bytes)

    def test_atomic_state_failures_preserve_exact_pending_state(self) -> None:
        from src import artifact_history

        for failure in ("write", "permissions", "file fsync", "rename", "root fsync"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                root.mkdir(exist_ok=True)
                index = {
                    "schema_version": 1,
                    "run_id": RUN_IDS[0],
                    "artifact_manifest_checksum": "sha256:" + "1" * 64,
                }
                candidate = artifact_history._PointerBinding(
                    RUN_IDS[0],
                    "sha256:" + "1" * 64,
                    canonical_json_bytes(index),
                )
                pending = artifact_history._FinalizationState(
                    "pending", candidate, None
                )
                retained_root = artifact_history._open_retained_directory(root)
                try:
                    artifact_history._record_finalization_state(
                        retained_root, RUN_IDS[0], pending
                    )
                    state_path = root / f".{RUN_IDS[0]}.finalize.state"
                    pending_bytes = state_path.read_bytes()
                    real_write_all = artifact_history._write_all
                    real_fchmod = os.fchmod
                    real_fsync = os.fsync
                    real_rename = os.rename

                    def fail_write(descriptor, content):
                        if failure == "write":
                            raise OSError("injected state-write failure")
                        return real_write_all(descriptor, content)

                    def fail_fchmod(descriptor, mode):
                        if failure == "permissions":
                            raise OSError("injected state-permissions failure")
                        return real_fchmod(descriptor, mode)

                    def fail_fsync(descriptor):
                        is_root = (
                            os.fstat(descriptor).st_ino == retained_root.inode
                            and os.fstat(descriptor).st_dev == retained_root.device
                        )
                        if failure == "file fsync" and not is_root:
                            raise OSError("injected state-file-fsync failure")
                        if failure == "root fsync" and is_root:
                            raise OSError("injected state-root-fsync failure")
                        return real_fsync(descriptor)

                    def fail_rename(source, destination, **kwargs):
                        if failure == "rename":
                            raise OSError("injected state-rename failure")
                        return real_rename(source, destination, **kwargs)

                    with (
                        patch.object(
                            artifact_history, "_write_all", side_effect=fail_write
                        ),
                        patch.object(os, "fchmod", side_effect=fail_fchmod),
                        patch.object(os, "fsync", side_effect=fail_fsync),
                        patch.object(os, "rename", side_effect=fail_rename),
                        self.assertRaisesRegex(OSError, "injected"),
                    ):
                        artifact_history._record_finalization_state(
                            retained_root, RUN_IDS[0], pending.completed()
                        )

                    self.assertEqual(state_path.read_bytes(), pending_bytes)
                    self.assertEqual(state_path.stat().st_mode & 0o777, 0o600)
                    self.assertEqual(
                        [path for path in root.iterdir() if path != state_path], []
                    )
                finally:
                    os.close(retained_root.descriptor)

    def test_finalization_state_requires_canonical_private_regular_file(self) -> None:
        from src import artifact_history

        for mutation in (
            "noncanonical",
            "extra field",
            "wrong checksum",
            "permissions",
            "symlink",
            "directory",
            "hard link",
        ):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                index = {
                    "schema_version": 1,
                    "run_id": RUN_IDS[0],
                    "artifact_manifest_checksum": "sha256:" + "1" * 64,
                }
                candidate = artifact_history._PointerBinding(
                    RUN_IDS[0],
                    "sha256:" + "1" * 64,
                    canonical_json_bytes(index),
                )
                pending = artifact_history._FinalizationState(
                    "pending", candidate, None
                )
                retained_root = artifact_history._open_retained_directory(root)
                state_path = root / f".{RUN_IDS[0]}.finalize.state"
                try:
                    artifact_history._record_finalization_state(
                        retained_root, RUN_IDS[0], pending
                    )
                    payload = json.loads(state_path.read_bytes())
                    if mutation == "noncanonical":
                        state_path.write_text(json.dumps(payload), encoding="utf-8")
                    elif mutation == "extra field":
                        payload["unexpected"] = True
                        state_path.write_bytes(canonical_json_bytes(payload))
                    elif mutation == "wrong checksum":
                        payload["candidate_pointer"]["bytes_checksum"] = (
                            "sha256:" + "0" * 64
                        )
                        state_path.write_bytes(canonical_json_bytes(payload))
                    elif mutation == "permissions":
                        state_path.chmod(0o644)
                    else:
                        state_path.unlink()
                        if mutation == "symlink":
                            state_path.symlink_to("missing-state")
                        elif mutation == "directory":
                            state_path.mkdir()
                        else:
                            source = root / "state-source"
                            source.write_bytes(canonical_json_bytes(payload))
                            source.chmod(0o600)
                            os.link(source, state_path)

                    with self.assertRaisesRegex(
                        ValueError, "run finalization state is unsafe"
                    ):
                        artifact_history._load_finalization_state_at(
                            retained_root.descriptor, RUN_IDS[0]
                        )
                finally:
                    os.close(retained_root.descriptor)

    def test_replace_bytes_never_removes_unowned_name_collisions(self) -> None:
        from src import artifact_history

        collision_id = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
        success_id = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
        for entry_type in ("file", "symlink", "directory"):
            with (
                self.subTest(entry_type=entry_type),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                collision = root / f".target.{collision_id.hex}.tmp"
                if entry_type == "file":
                    collision.write_bytes(b"owned elsewhere")
                elif entry_type == "symlink":
                    collision.symlink_to("missing")
                else:
                    collision.mkdir()
                descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    with patch.object(
                        artifact_history.uuid,
                        "uuid4",
                        side_effect=(collision_id, success_id),
                    ):
                        artifact_history._replace_bytes_at(
                            descriptor, "target", b"replacement"
                        )
                finally:
                    os.close(descriptor)
                self.assertTrue(collision.exists() or collision.is_symlink())
                self.assertEqual((root / "target").read_bytes(), b"replacement")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            collision = root / f".target.{collision_id.hex}.tmp"
            collision.write_bytes(b"owned elsewhere")
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with (
                    patch.object(
                        artifact_history.uuid, "uuid4", return_value=collision_id
                    ),
                    self.assertRaises(FileExistsError),
                ):
                    artifact_history._replace_bytes_at(
                        descriptor, "target", b"replacement"
                    )
            finally:
                os.close(descriptor)
            self.assertEqual(collision.read_bytes(), b"owned elsewhere")
            self.assertFalse((root / "target").exists())

    def test_replace_bytes_rejects_parked_temp_source_substitutes(self) -> None:
        """Retain the exclusive source and never clean a replacement by its name."""
        from src import artifact_history

        for entry_type in ("valid file", "file", "symlink", "directory"):
            with (
                self.subTest(entry_type=entry_type),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                parked = root / "parked-owned-temp"
                descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                real_rename = os.rename

                def substitute(source, destination, **kwargs):
                    source_path = root / str(source)
                    if destination == "target":
                        real_rename(source, parked.name, **kwargs)
                        if entry_type == "valid file":
                            source_path.write_bytes(b"prior pointer")
                        elif entry_type == "file":
                            source_path.write_bytes(b"unowned")
                        elif entry_type == "symlink":
                            source_path.symlink_to("missing")
                        else:
                            source_path.mkdir()
                    return real_rename(source, destination, **kwargs)

                try:
                    with (
                        patch.object(os, "rename", side_effect=substitute),
                        self.assertRaises(ValueError),
                    ):
                        artifact_history._replace_bytes_at(
                            descriptor, "target", b"candidate"
                        )
                finally:
                    os.close(descriptor)

                self.assertEqual(parked.read_bytes(), b"candidate")
                target = root / "target"
                if entry_type == "valid file":
                    self.assertEqual(target.read_bytes(), b"prior pointer")
                elif entry_type == "file":
                    self.assertEqual(target.read_bytes(), b"unowned")
                elif entry_type == "symlink":
                    self.assertTrue(target.is_symlink())
                else:
                    self.assertTrue(target.is_dir())

    def test_root_replacement_at_complete_transition_never_returns_success(
        self,
    ) -> None:
        """Restore pending state on the retained root at both complete boundaries."""
        from src import artifact_history

        for boundary in ("before", "after"):
            with (
                self.subTest(boundary=boundary),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir) / "root"
                root.mkdir()
                run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                parked = root.parent / "parked-root"
                real_record = artifact_history._record_finalization_state
                replaced = False

                def replace_root():
                    nonlocal replaced
                    if replaced:
                        return
                    replaced = True
                    root.rename(parked)
                    root.mkdir()
                    (root / "runs").mkdir()

                def record(retained_root, run_id, state):
                    if state.status == "complete" and boundary == "before":
                        replace_root()
                    result = real_record(retained_root, run_id, state)
                    if state.status == "complete" and boundary == "after":
                        replace_root()
                    return result

                with (
                    patch.object(
                        artifact_history,
                        "_record_finalization_state",
                        side_effect=record,
                    ),
                    self.assertRaisesRegex(ValueError, "history changed"),
                ):
                    publish_completed_run(root, run)

                state_path = parked / f".{RUN_IDS[0]}.finalize.state"
                self.assertEqual(
                    json.loads(state_path.read_bytes())["state"], "pending"
                )
                self.assertFalse((root / "current.json").exists())

    def test_pruning_removes_only_safe_finalization_state(self) -> None:
        """Remove complete and obsolete state while preserving recoverable state."""
        from src import artifact_history

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            oldest = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            current = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
            publish_completed_run(root, oldest)
            publish_completed_run(root, current)
            oldest_state = root / f".{RUN_IDS[0]}.finalize.state"

            self.assertEqual(prune_run_history(root, 1), [oldest.resolve()])
            self.assertFalse(oldest_state.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            pending = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
            publish_completed_run(root, current)
            with (
                patch.object(
                    artifact_history,
                    "_write_current_index_at",
                    side_effect=OSError("injected pointer failure"),
                ),
                self.assertRaises(OSError),
            ):
                publish_completed_run(root, pending)
            pending_state = root / f".{RUN_IDS[1]}.finalize.state"

            self.assertEqual(prune_run_history(root, 1), [])
            self.assertTrue(pending.is_dir())
            self.assertTrue(pending_state.is_file())

            replacement = create_run(root, RUN_IDS[2], "2026-01-03T00:00:00Z")
            publish_completed_run(root, replacement)
            self.assertEqual(
                prune_run_history(root, 1), [current.resolve(), pending.resolve()]
            )
            self.assertFalse(pending_state.exists())

    def test_pruning_preserves_unsafe_state_and_state_after_delete_failure(
        self,
    ) -> None:
        """Never remove malformed state or state for a run whose deletion failed."""
        from src import artifact_history

        for failure in ("unsafe state", "delete failure"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                oldest = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                current = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
                publish_completed_run(root, oldest)
                publish_completed_run(root, current)
                state_path = root / f".{RUN_IDS[0]}.finalize.state"
                if failure == "unsafe state":
                    state_path.write_bytes(b"not-json")
                    self.assertEqual(prune_run_history(root, 1), [])
                else:
                    real_rmtree = artifact_history.shutil.rmtree

                    def fail_oldest(path, *args, **kwargs):
                        if str(path) == RUN_IDS[0]:
                            raise OSError("injected prune failure")
                        return real_rmtree(path, *args, **kwargs)

                    with (
                        patch.object(
                            artifact_history.shutil,
                            "rmtree",
                            side_effect=fail_oldest,
                        ),
                        self.assertRaisesRegex(OSError, "prune failure"),
                    ):
                        prune_run_history(root, 1)
                self.assertTrue(oldest.is_dir())
                self.assertTrue(state_path.exists())

    def test_pruning_recovers_state_after_run_deletion_failures(self) -> None:
        """Retry state cleanup when deletion crossed a failed durability step."""
        from src import artifact_history

        for failure in ("runs fsync", "state cleanup"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                oldest = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                current = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
                publish_completed_run(root, oldest)
                publish_completed_run(root, current)
                state_path = root / f".{RUN_IDS[0]}.finalize.state"
                runs_identity = (
                    (root / "runs").stat().st_dev,
                    (root / "runs").stat().st_ino,
                )
                root_identity = (root.stat().st_dev, root.stat().st_ino)
                real_sync = artifact_history._sync_retained_directory
                failed = False

                def fail_sync(directory):
                    nonlocal failed
                    if (
                        failure == "runs fsync"
                        and (directory.device, directory.inode) == runs_identity
                        and not failed
                    ):
                        failed = True
                        raise OSError("injected runs-directory fsync failure")
                    if (
                        failure == "state cleanup"
                        and (directory.device, directory.inode) == root_identity
                        and not state_path.exists()
                        and not failed
                    ):
                        failed = True
                        raise OSError("injected state-cleanup failure")
                    return real_sync(directory)

                with (
                    patch.object(
                        artifact_history,
                        "_sync_retained_directory",
                        side_effect=fail_sync,
                    ),
                    self.assertRaisesRegex(OSError, "injected"),
                ):
                    prune_run_history(root, 1)

                self.assertFalse(oldest.exists())
                self.assertTrue(state_path.is_file())
                self.assertEqual(prune_run_history(root, 1), [])
                self.assertFalse(state_path.exists())

    def test_pruning_cleans_only_safe_state_for_absent_runs(self) -> None:
        """Clean complete and obsolete pending state but retain recovery evidence."""
        from src import artifact_history

        for state_kind in ("obsolete pending", "recoverable pending", "unsafe"):
            with (
                self.subTest(state_kind=state_kind),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                previous = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                candidate = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
                publish_completed_run(root, previous)
                with (
                    patch.object(
                        artifact_history,
                        "_write_current_index_at",
                        side_effect=OSError("injected pointer failure"),
                    ),
                    self.assertRaisesRegex(OSError, "pointer failure"),
                ):
                    publish_completed_run(root, candidate)
                state_path = root / f".{RUN_IDS[1]}.finalize.state"
                if state_kind == "obsolete pending":
                    replacement = create_run(root, RUN_IDS[2], "2026-01-03T00:00:00Z")
                    publish_completed_run(root, replacement)
                elif state_kind == "unsafe":
                    state_path.write_bytes(b"not-json")
                artifact_history.shutil.rmtree(candidate)

                self.assertEqual(prune_run_history(root, 3), [])
                self.assertEqual(state_path.exists(), state_kind != "obsolete pending")

    def test_pruning_orphan_recovery_serializes_concurrent_publication(self) -> None:
        """Hold the root lock through orphan discovery and durable state removal."""
        from src import artifact_history

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            previous = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            orphan = create_run(root, RUN_IDS[1], "2026-01-02T00:00:00Z")
            replacement = create_run(root, RUN_IDS[2], "2026-01-03T00:00:00Z")
            concurrent_id = "44444444-4444-4444-8444-444444444444"
            concurrent = create_run(root, concurrent_id, "2026-01-04T00:00:00Z")
            publish_completed_run(root, previous)
            with (
                patch.object(
                    artifact_history,
                    "_write_current_index_at",
                    side_effect=OSError("injected pointer failure"),
                ),
                self.assertRaises(OSError),
            ):
                publish_completed_run(root, orphan)
            publish_completed_run(root, replacement)
            artifact_history.shutil.rmtree(orphan)
            orphan_state = root / f".{RUN_IDS[1]}.finalize.state"
            real_recover = artifact_history._recover_absent_run_states
            recovering = threading.Event()
            release = threading.Event()
            published = threading.Event()

            def pause_recovery(*args):
                recovering.set()
                self.assertTrue(release.wait(timeout=2))
                return real_recover(*args)

            def publish_concurrently() -> None:
                publish_completed_run(root, concurrent)
                published.set()

            with (
                patch.object(
                    artifact_history,
                    "_recover_absent_run_states",
                    side_effect=pause_recovery,
                ),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                prune = executor.submit(prune_run_history, root, 4)
                self.assertTrue(recovering.wait(timeout=2))
                publication = executor.submit(publish_concurrently)
                self.assertFalse(published.wait(timeout=0.1))
                release.set()
                self.assertEqual(prune.result(timeout=2), [])
                publication.result(timeout=2)

            self.assertFalse(orphan_state.exists())
            self.assertEqual(resolve_current_run_dir(root), concurrent.resolve())

    def test_replaced_legacy_lock_entry_cannot_split_root_serialization(self) -> None:
        from src import artifact_history

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            second_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            legacy_lock = root / f".{RUN_IDS[0]}.finalize.lock"
            legacy_lock.write_bytes(b"old")
            acquired = threading.Event()

            def acquire_second() -> None:
                with artifact_history._finalization_lock(second_descriptor):
                    acquired.set()

            try:
                with artifact_history._finalization_lock(first_descriptor):
                    parked = root / "parked-lock"
                    legacy_lock.rename(parked)
                    legacy_lock.write_bytes(b"replacement")
                    thread = threading.Thread(target=acquire_second)
                    thread.start()
                    self.assertFalse(acquired.wait(timeout=0.1))
                self.assertTrue(acquired.wait(timeout=1))
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive())
            finally:
                os.close(first_descriptor)
                os.close(second_descriptor)

    def test_finalization_closes_retained_descriptors_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            runs_identity = (
                (root / "runs").stat().st_dev,
                (root / "runs").stat().st_ino,
            )
            baseline = len(os.listdir("/proc/self/fd"))
            real_fsync = os.fsync

            def fail_runs_barrier(descriptor):
                file_stat = os.fstat(descriptor)
                if (file_stat.st_dev, file_stat.st_ino) == runs_identity:
                    raise OSError("simulated runs fsync failure")
                return real_fsync(descriptor)

            for _ in range(3):
                with (
                    patch("src.artifact_history.os.fsync", fail_runs_barrier),
                    self.assertRaisesRegex(OSError, "runs fsync failure"),
                ):
                    publish_completed_run(root, run)
                self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)

    def test_publication_crosses_durability_barriers_before_returning_current(
        self,
    ) -> None:
        from src import artifact_history

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            events: list[str] = []
            root_identity = (root.stat().st_dev, root.stat().st_ino)
            run_identity = (run.stat().st_dev, run.stat().st_ino)
            runs_identity = (
                (root / "runs").stat().st_dev,
                (root / "runs").stat().st_ino,
            )
            real_sync_files = artifact_history._sync_retained_files
            real_sync_directory = artifact_history._sync_retained_directory
            real_write_current = artifact_history._write_current_index_at

            def sync_files(inventory):
                events.append("files:sync")
                return real_sync_files(inventory)

            def sync_directory(directory):
                identity = (directory.device, directory.inode)
                labels = {
                    root_identity: "root",
                    runs_identity: "runs",
                    run_identity: "run",
                }
                events.append(f"{labels[identity]}:sync")
                return real_sync_directory(directory)

            def write_current(*args, **kwargs):
                events.append("current:start")
                result = real_write_current(*args, **kwargs)
                events.append("current:return")
                return result

            with (
                patch.object(
                    artifact_history,
                    "_sync_retained_files",
                    side_effect=sync_files,
                ),
                patch.object(
                    artifact_history,
                    "_sync_retained_directory",
                    side_effect=sync_directory,
                ),
                patch.object(
                    artifact_history,
                    "_write_current_index_at",
                    side_effect=write_current,
                ),
            ):
                publish_completed_run(root, run)

            files_sync = max(
                index for index, event in enumerate(events) if event == "files:sync"
            )
            completed_run_sync = events.index("run:sync", files_sync)
            runs_root_sync = events.index("runs:sync", completed_run_sync)
            current_start = events.index("current:start")
            root_sync = events.index("root:sync", current_start)
            current_return = events.index("current:return")
            self.assertLess(files_sync, completed_run_sync)
            self.assertLess(completed_run_sync, runs_root_sync)
            self.assertLess(runs_root_sync, current_start)
            self.assertLess(current_start, root_sync)
            self.assertLess(root_sync, current_return)
            self.assertEqual(resolve_current_run_dir(root), run.resolve())

    def test_durability_checkpoint_failures_do_not_select_and_are_recoverable(
        self,
    ) -> None:
        from src import artifact_history

        for checkpoint in (
            "captured files",
            "completed manifest directory",
            "runs directory",
        ):
            with (
                self.subTest(checkpoint=checkpoint),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                if checkpoint == "captured files":
                    durability_patch = patch.object(
                        artifact_history,
                        "_sync_retained_files",
                        side_effect=OSError("captured-file fsync failure"),
                    )
                else:
                    real_sync = artifact_history._sync_retained_directory
                    target_path = (
                        run
                        if checkpoint == "completed manifest directory"
                        else root / "runs"
                    )
                    target_identity = (
                        target_path.stat().st_dev,
                        target_path.stat().st_ino,
                    )

                    def fail_directory_sync(directory):
                        if (directory.device, directory.inode) == target_identity:
                            raise OSError(f"{checkpoint} fsync failure")
                        return real_sync(directory)

                    durability_patch = patch.object(
                        artifact_history,
                        "_sync_retained_directory",
                        side_effect=fail_directory_sync,
                    )

                with durability_patch, self.assertRaisesRegex(OSError, "fsync"):
                    publish_completed_run(root, run)

                self.assertFalse((root / "current.json").exists())
                publish_completed_run(root, run)
                self.assertEqual(resolve_current_run_dir(root), run.resolve())

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
