import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.artifact_history import (
    load_current_run_snapshot,
    prune_run_history,
    publish_completed_run,
    resolve_current_run_dir,
)
from src.artifact_compatibility import write_server_artifact_manifest
from src.run_provenance import write_run_provenance_manifest
from tests.artifact_helpers import fake_app_manifest
from tests.test_run_provenance import runtime_environment


RUN_IDS = (
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
)


def create_run(root: Path, run_id: str, created_at: str) -> Path:
    """Create a complete run candidate for history tests.

    Parameters
    ----------
    root : pathlib.Path
        Artifact history root.
    run_id : str
        Canonical UUID4 directory identity.
    created_at : str
        Deterministic provenance timestamp.

    Returns
    -------
    pathlib.Path
        Created run directory.
    """
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
        write_run_provenance_manifest(run_dir, {}, created_at=created_at, run_id=run_id)
    (run_dir / "global_model.keras").write_bytes(f"model-{run_id}".encode())
    (run_dir / "metrics.csv").write_text(
        "round,loss,accuracy\n1,0.5,0.75\n", encoding="utf-8"
    )
    write_server_artifact_manifest(run_dir, app_manifest=fake_app_manifest())
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
