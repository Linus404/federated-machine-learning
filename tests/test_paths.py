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

from src.artifact_compatibility import canonical_json_bytes
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


def write_prepared_generation(
    root: Path,
    *,
    pointer_generation_id: str | None = None,
    index_generation_id: str | None = None,
) -> tuple[Path, str]:
    """Create one selected prepared generation for resolver tests.

    Parameters
    ----------
    root : pathlib.Path
        Shared parent of the logical and immutable prepared roots.
    pointer_generation_id : str or None, optional
        Directory identity selected by the pointer.
    index_generation_id : str or None, optional
        Identity serialized inside ``index.json``.

    Returns
    -------
    tuple of pathlib.Path and str
        Generation directory and pointer identity.
    """
    selected_id = pointer_generation_id or str(uuid.uuid4())
    serialized_id = index_generation_id or selected_id
    generation = root / PREPARED_GENERATIONS_DIRECTORY / selected_id
    generation.mkdir(parents=True)
    logical_roots = {
        "client": "clients",
        "evaluation": "evaluation",
        "public": "public",
    }
    for kind, logical_name in logical_roots.items():
        (generation / kind).mkdir()
        (root / logical_name).symlink_to(
            Path(PREPARED_CURRENT_FILENAME) / kind,
            target_is_directory=True,
        )
    (generation / "index.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": PREPARED_GENERATION_SCHEMA_VERSION,
                "generation_id": serialized_id,
                "logical_roots": logical_roots,
                "preparation_request": {"partitions": 4},
            }
        )
    )
    (root / PREPARED_CURRENT_FILENAME).symlink_to(
        Path(PREPARED_GENERATIONS_DIRECTORY) / selected_id,
        target_is_directory=True,
    )
    return generation, selected_id


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
    def test_prepared_generation_requires_matching_canonical_identities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generation, _ = write_prepared_generation(root)

            self.assertEqual(
                resolve_prepared_artifact_dir(root / "public", "public"),
                generation / "public",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_prepared_generation(root, index_generation_id=str(uuid.uuid4()))

            with self.assertRaisesRegex(ValueError, "identities differ"):
                resolve_prepared_artifact_dir(root / "public", "public")

    def test_prepared_generation_rejects_noncanonical_or_transplanted_index(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generation, _ = write_prepared_generation(root)
            index_path = generation / "index.json"
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            index_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not canonical"):
                resolve_prepared_artifact_dir(root / "public", "public")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_id = str(uuid.uuid4())
            second_id = str(uuid.uuid4())
            generation, _ = write_prepared_generation(
                root,
                pointer_generation_id=second_id,
                index_generation_id=first_id,
            )
            self.assertEqual(
                json.loads((generation / "index.json").read_text(encoding="utf-8"))[
                    "generation_id"
                ],
                first_id,
            )

            with self.assertRaisesRegex(ValueError, "identities differ"):
                resolve_prepared_artifact_dir(root / "clients", "client")

    def test_prepared_generation_rejects_older_and_newer_request_schemas(self) -> None:
        for version in (1, PREPARED_GENERATION_SCHEMA_VERSION + 1):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                generation, _ = write_prepared_generation(root)
                index = generation / "index.json"
                payload = json.loads(index.read_text(encoding="utf-8"))
                payload["schema_version"] = version
                if version == 1:
                    payload.pop("preparation_request")
                index.write_bytes(canonical_json_bytes(payload))

                with self.assertRaisesRegex(ValueError, "schema|field set"):
                    resolve_prepared_artifact_dir(root / "public", "public")

    def test_prepared_generation_rejects_reordered_bool_and_duplicate_fields(
        self,
    ) -> None:
        mutations = (
            lambda payload: {
                "generation_id": payload["generation_id"],
                "schema_version": payload["schema_version"],
                "logical_roots": payload["logical_roots"],
                "preparation_request": payload["preparation_request"],
            },
            lambda payload: {
                **payload,
                "logical_roots": {
                    "client": "clients",
                    "public": "public",
                    "evaluation": "evaluation",
                },
            },
            lambda payload: {**payload, "schema_version": True},
            lambda payload: {
                **payload,
                "preparation_request": {"partitions": True},
            },
            lambda payload: {
                **payload,
                "preparation_request": {"partitions": 0},
            },
        )
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                generation, _ = write_prepared_generation(root)
                index = generation / "index.json"
                payload = json.loads(index.read_text(encoding="utf-8"))
                index.write_bytes(canonical_json_bytes(mutation(payload)))

                with self.assertRaises(ValueError):
                    resolve_prepared_artifact_dir(root / "public", "public")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generation, generation_id = write_prepared_generation(root)
            (generation / "index.json").write_text(
                '{"schema_version":2,"schema_version":2,'
                f'"generation_id":"{generation_id}",'
                '"logical_roots":{"client":"clients",'
                '"evaluation":"evaluation","public":"public"},'
                '"preparation_request":{"partitions":4}}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid or unsafe"):
                resolve_prepared_artifact_dir(root / "public", "public")

    def test_prepared_generation_rejects_pointer_swap_and_hostile_entries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, generation_id = write_prepared_generation(root)
            pointer = root / PREPARED_CURRENT_FILENAME
            pointer.unlink()
            pointer.symlink_to(
                f"./{PREPARED_GENERATIONS_DIRECTORY}/{generation_id}",
                target_is_directory=True,
            )

            with self.assertRaisesRegex(ValueError, "pointer identity"):
                resolve_prepared_artifact_dir(root / "evaluation", "evaluation")

        for hostile_entry in (
            "pointer",
            "logical",
            "generations",
            "generation",
            "index",
            "selected",
        ):
            with (
                self.subTest(hostile_entry=hostile_entry),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                generation, _ = write_prepared_generation(root)
                external = root / "external"
                external.mkdir()
                if hostile_entry == "pointer":
                    pointer = root / PREPARED_CURRENT_FILENAME
                    pointer.unlink()
                    pointer.write_text(generation.name, encoding="utf-8")
                    expected_error = "atomic directory link"
                elif hostile_entry == "logical":
                    logical = root / "public"
                    logical.unlink()
                    logical.mkdir()
                    expected_error = "logical root"
                elif hostile_entry == "generations":
                    generations = root / PREPARED_GENERATIONS_DIRECTORY
                    renamed = root / "real-generations"
                    generations.rename(renamed)
                    generations.symlink_to(renamed, target_is_directory=True)
                    expected_error = "missing or unsafe"
                elif hostile_entry == "generation":
                    for child in generation.iterdir():
                        if child.is_dir():
                            child.rmdir()
                        else:
                            child.unlink()
                    generation.rmdir()
                    generation.symlink_to(external, target_is_directory=True)
                    expected_error = "missing or unsafe"
                elif hostile_entry == "index":
                    index = generation / "index.json"
                    index.unlink()
                    index.symlink_to(external / "index.json")
                    (external / "index.json").write_text("{}", encoding="utf-8")
                    expected_error = "invalid or unsafe"
                else:
                    selected = generation / "public"
                    selected.rmdir()
                    selected.symlink_to(external, target_is_directory=True)
                    expected_error = "missing or unsafe"

                with self.assertRaisesRegex(ValueError, expected_error):
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
