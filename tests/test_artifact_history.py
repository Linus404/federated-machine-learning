import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.artifact_history import (
    prune_run_history,
    publish_completed_run,
    resolve_current_run_dir,
)
from src.run_provenance import write_run_provenance_manifest
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


if __name__ == "__main__":
    unittest.main()
