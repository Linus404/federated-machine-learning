import hashlib
import json
import os
import socket
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
                            "src.artifact_history.write_json_atomically",
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
                        "src.artifact_history.write_json_atomically",
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

    def test_publication_crosses_durability_barriers_before_returning_current(
        self,
    ) -> None:
        from src import artifact_compatibility, artifact_history

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
            events: list[str] = []
            real_sync_files = artifact_history.sync_server_artifact_files
            real_sync_directory = artifact_compatibility.sync_directory
            real_run_sync = artifact_history.sync_directory
            real_write_current = artifact_history.write_json_atomically

            def sync_files(*args, **kwargs):
                events.append("captured-files:sync")
                return real_sync_files(*args, **kwargs)

            def sync_written_directory(path):
                events.append(f"atomic-directory:{Path(path).resolve()}")
                return real_sync_directory(path)

            def sync_completed_run(path):
                events.append("completed-run:sync")
                return real_run_sync(path)

            def write_current(path, payload, *, overwrite=True):
                events.append("current:start")
                result = real_write_current(path, payload, overwrite=overwrite)
                events.append("current:return")
                return result

            with (
                patch.object(
                    artifact_history,
                    "sync_server_artifact_files",
                    side_effect=sync_files,
                ),
                patch.object(
                    artifact_compatibility,
                    "sync_directory",
                    side_effect=sync_written_directory,
                ),
                patch.object(
                    artifact_history,
                    "sync_directory",
                    side_effect=sync_completed_run,
                ),
                patch.object(
                    artifact_history,
                    "write_json_atomically",
                    side_effect=write_current,
                ),
            ):
                publish_completed_run(root, run)

            files_sync = events.index("captured-files:sync")
            completed_manifest_sync = events.index(
                f"atomic-directory:{run.resolve()}", files_sync
            )
            completed_run_sync = events.index("completed-run:sync")
            current_start = events.index("current:start")
            root_sync = events.index(
                f"atomic-directory:{root.resolve()}", current_start
            )
            current_return = events.index("current:return")
            self.assertLess(files_sync, completed_manifest_sync)
            self.assertLess(completed_manifest_sync, completed_run_sync)
            self.assertLess(completed_run_sync, current_start)
            self.assertLess(current_start, root_sync)
            self.assertLess(root_sync, current_return)
            self.assertEqual(resolve_current_run_dir(root), run.resolve())

    def test_durability_checkpoint_failures_do_not_select_and_are_recoverable(
        self,
    ) -> None:
        from src import artifact_compatibility, artifact_history

        for checkpoint in ("captured files", "completed manifest directory"):
            with (
                self.subTest(checkpoint=checkpoint),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                run = create_run(root, RUN_IDS[0], "2026-01-01T00:00:00Z")
                if checkpoint == "captured files":
                    durability_patch = patch.object(
                        artifact_history,
                        "sync_server_artifact_files",
                        side_effect=OSError("captured-file fsync failure"),
                    )
                else:
                    real_sync = artifact_compatibility.sync_directory
                    run_sync_count = 0

                    def fail_completed_manifest_sync(path):
                        nonlocal run_sync_count
                        if Path(path).resolve() == run.resolve():
                            run_sync_count += 1
                            if run_sync_count == 3:
                                raise OSError("completed-directory fsync failure")
                        return real_sync(path)

                    durability_patch = patch.object(
                        artifact_compatibility,
                        "sync_directory",
                        side_effect=fail_completed_manifest_sync,
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
