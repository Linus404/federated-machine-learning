import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from src.paths import (
    EVALUATION_ARTIFACT_DIR_ENV,
    PREPARED_CURRENT_FILENAME,
    PREPARED_GENERATIONS_DIRECTORY,
    PREPARED_GENERATION_SCHEMA_VERSION,
    SERVER_ARTIFACT_DIR_ENV,
    acquire_run_artifact_lock,
    client_metrics_path,
    default_evaluation_artifact_dir,
    default_server_artifact_dir,
    global_model_path,
    metrics_path,
    resolve_prepared_artifact_dir,
    run_manifest_path,
)


class RunArtifactLockTests(unittest.TestCase):
    def test_lock_excludes_a_second_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "server"
            child_code = textwrap.dedent(
                """
                import sys
                from pathlib import Path

                from src.paths import acquire_run_artifact_lock

                lock = acquire_run_artifact_lock(Path(sys.argv[1]))
                print("locked", flush=True)
                sys.stdin.readline()
                lock.release()
                """
            )
            process = subprocess.Popen(
                [sys.executable, "-c", child_code, str(artifact_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                if process.stdout is None:
                    self.fail("child stdout was not captured")
                self.assertEqual(process.stdout.readline().strip(), "locked")
                with self.assertRaisesRegex(RuntimeError, "already writing"):
                    acquire_run_artifact_lock(artifact_dir)
            finally:
                if process.stdin is not None:
                    process.stdin.write("\n")
                    process.stdin.flush()
                _, stderr = process.communicate(timeout=5)

            self.assertEqual(process.returncode, 0, stderr)
            lock = acquire_run_artifact_lock(artifact_dir)
            lock.release()


class ArtifactPathContractTests(unittest.TestCase):
    def test_prepared_index_checks_each_schema_version_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generation_id = str(uuid.uuid4())
            generation = root / PREPARED_GENERATIONS_DIRECTORY / generation_id
            for artifact_kind in ("client", "public", "evaluation"):
                (generation / artifact_kind).mkdir(parents=True)
            try:
                (root / PREPARED_CURRENT_FILENAME).symlink_to(
                    Path(PREPARED_GENERATIONS_DIRECTORY) / generation_id,
                    target_is_directory=True,
                )
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            valid_index = {
                "schema_version": PREPARED_GENERATION_SCHEMA_VERSION,
                "generation_id": generation_id,
                "logical_roots": {
                    "client": "clients",
                    "public": "public",
                    "evaluation": "evaluation",
                },
            }
            missing = object()
            versions = (
                missing,
                None,
                False,
                True,
                float(PREPARED_GENERATION_SCHEMA_VERSION),
                PREPARED_GENERATION_SCHEMA_VERSION - 1,
                PREPARED_GENERATION_SCHEMA_VERSION,
                PREPARED_GENERATION_SCHEMA_VERSION + 1,
            )

            for version in versions:
                payload = dict(valid_index)
                if version is missing:
                    payload.pop("schema_version")
                else:
                    payload["schema_version"] = version
                (generation / "index.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

                with self.subTest(version="missing" if version is missing else version):
                    if (
                        type(version) is int
                        and version == PREPARED_GENERATION_SCHEMA_VERSION
                    ):
                        self.assertEqual(
                            resolve_prepared_artifact_dir(root / "public", "public"),
                            (generation / "public").resolve(),
                        )
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "field set|schema_version"
                        ):
                            resolve_prepared_artifact_dir(root / "public", "public")

    def test_default_evaluation_artifact_dir_prefers_evaluation_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluation_dir = Path(tmpdir) / "evaluation-artifacts"

            with patch.dict(
                os.environ,
                {EVALUATION_ARTIFACT_DIR_ENV: str(evaluation_dir)},
                clear=True,
            ):
                resolved = default_evaluation_artifact_dir()

            self.assertEqual(resolved, evaluation_dir.resolve())

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
            self.assertEqual(
                run_manifest_path(resolved_artifact_dir),
                artifact_dir.resolve() / "run_manifest.json",
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
