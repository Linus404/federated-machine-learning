import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.paths import (
    SERVER_ARTIFACT_DIR_ENV,
    client_metrics_path,
    clear_artifact_dir,
    default_server_artifact_dir,
    global_model_path,
    metrics_path,
)


class ClearArtifactDirTests(unittest.TestCase):
    def test_clear_artifact_dir_removes_existing_contents_and_keeps_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            nested_dir = artifact_dir / "nested"
            nested_dir.mkdir(parents=True)
            (nested_dir / "old.txt").write_text("old", encoding="utf-8")
            (artifact_dir / "metrics.csv").write_text("stale", encoding="utf-8")

            resolved = clear_artifact_dir(artifact_dir)

            self.assertEqual(resolved, artifact_dir.resolve())
            self.assertTrue(artifact_dir.is_dir())
            self.assertEqual(list(artifact_dir.iterdir()), [])

    def test_clear_artifact_dir_creates_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"

            resolved = clear_artifact_dir(artifact_dir)

            self.assertEqual(resolved, artifact_dir.resolve())
            self.assertTrue(artifact_dir.is_dir())

    def test_clear_artifact_dir_refuses_current_working_directory(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)

                with self.assertRaises(ValueError):
                    clear_artifact_dir(".")
            finally:
                os.chdir(original_cwd)

    def test_clear_artifact_dir_refuses_protected_path_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            public_dir = Path(tmpdir) / "public"
            public_dir.mkdir()
            (public_dir / "vocab.txt").write_text("vocab", encoding="utf-8")

            with self.assertRaises(ValueError):
                clear_artifact_dir(public_dir, protected_paths=[public_dir])

            self.assertTrue((public_dir / "vocab.txt").exists())

    def test_clear_artifact_dir_refuses_when_protected_path_is_inside(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            public_dir = artifact_dir / "public"
            public_dir.mkdir(parents=True)
            (public_dir / "vocab.txt").write_text("vocab", encoding="utf-8")

            with self.assertRaises(ValueError):
                clear_artifact_dir(artifact_dir, protected_paths=[public_dir])

            self.assertTrue((public_dir / "vocab.txt").exists())

    def test_clear_artifact_dir_refuses_symlinked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "target"
            target_dir.mkdir()
            (target_dir / "old.txt").write_text("old", encoding="utf-8")
            artifact_dir = Path(tmpdir) / "artifacts"
            try:
                artifact_dir.symlink_to(target_dir, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            with self.assertRaises(ValueError):
                clear_artifact_dir(artifact_dir)

            self.assertTrue((target_dir / "old.txt").exists())

    def test_clear_artifact_dir_refuses_broken_symlinked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "missing-target"
            artifact_dir = Path(tmpdir) / "artifacts"
            try:
                artifact_dir.symlink_to(target_dir, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            with self.assertRaises(ValueError):
                clear_artifact_dir(artifact_dir)

            self.assertFalse(target_dir.exists())
            self.assertTrue(artifact_dir.is_symlink())

    def test_clear_artifact_dir_refuses_protected_symlink_inside(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            public_target = Path(tmpdir) / "public-target"
            artifact_dir.mkdir()
            public_target.mkdir()
            (public_target / "vocab.txt").write_text("vocab", encoding="utf-8")
            public_link = artifact_dir / "public"
            try:
                public_link.symlink_to(public_target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            with self.assertRaises(ValueError):
                clear_artifact_dir(artifact_dir, protected_paths=[public_link])

            self.assertTrue(public_link.is_symlink())
            self.assertTrue((public_target / "vocab.txt").exists())


class ArtifactPathContractTests(unittest.TestCase):
    def test_server_artifact_paths_follow_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "selected-artifacts"

            with patch.dict(os.environ, {SERVER_ARTIFACT_DIR_ENV: str(artifact_dir)}):
                resolved_artifact_dir = default_server_artifact_dir()

            self.assertEqual(resolved_artifact_dir, artifact_dir.resolve())
            self.assertEqual(
                global_model_path(resolved_artifact_dir),
                artifact_dir.resolve() / "global_model.keras",
            )
            self.assertEqual(
                metrics_path(resolved_artifact_dir),
                artifact_dir.resolve() / "metrics.csv",
            )
            self.assertEqual(
                client_metrics_path(resolved_artifact_dir),
                artifact_dir.resolve() / "client_metrics.csv",
            )

    def test_default_server_artifact_dir_prefers_server_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            server_artifact_dir = Path(tmpdir) / "server-artifacts"

            with patch.dict(
                os.environ,
                {
                    SERVER_ARTIFACT_DIR_ENV: str(server_artifact_dir),
                },
                clear=True,
            ):
                resolved_artifact_dir = default_server_artifact_dir()

            self.assertEqual(resolved_artifact_dir, server_artifact_dir.resolve())


if __name__ == "__main__":
    unittest.main()
