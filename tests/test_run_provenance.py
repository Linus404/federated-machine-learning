import hashlib
import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.artifact_compatibility import (
    ARTIFACT_SCHEMA_VERSION,
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    canonical_json_bytes,
)
from src.run_provenance import (
    _code_revision,
    _dataset_metadata,
    _environment_metadata,
    load_run_provenance_manifest,
    write_run_provenance_manifest,
)


def write_public_dataset_contract(path: Path) -> dict[str, object]:
    """Write a small valid public contract and return its frozen protocol.

    Parameters
    ----------
    path : pathlib.Path
        Public artifact directory.

    Returns
    -------
    dict of str to object
        Frozen protocol matching the test artifact.
    """
    vocabulary = b"\n[UNK]\ngood\nbad\n"
    vocabulary_sha256 = hashlib.sha256(vocabulary).hexdigest()
    (path / "vocab.txt").write_bytes(vocabulary)
    dataset = {
        "id": "stanfordnlp/imdb",
        "config": "plain_text",
        "revision": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
        "datasets_version": "4.8.5",
        "split": "train",
        "rows": 25000,
        "raw_parquet_sha256": "db47d16b" + "0" * 56,
        "content_sha256": "4639bf10" + "0" * 56,
    }
    (path / "manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
                "embedding_dim": 100,
                "sequence_length": 500,
                "vocabulary_size": 4,
                "vocabulary": {
                    "filename": "vocab.txt",
                    "sha256": vocabulary_sha256,
                    "size_bytes": len(vocabulary),
                },
                "dataset": dataset,
            }
        )
    )
    return {
        "dataset": {
            **{
                key: dataset[key]
                for key in (
                    "id",
                    "config",
                    "revision",
                    "datasets_version",
                )
            },
            "splits": {
                "train": {
                    key: dataset[key]
                    for key in (
                        "rows",
                        "raw_parquet_sha256",
                        "content_sha256",
                    )
                }
            },
        },
        "preprocessing": {
            "vocabulary_size": 4,
            "max_tokens": 4,
            "output_sequence_length": 500,
            "vocabulary_sha256": vocabulary_sha256,
        },
        "model": {
            "vocabulary_size": 4,
            "sequence_length": 500,
            "embedding_dimension": 100,
        },
    }


def runtime_environment() -> dict[str, object]:
    """Return complete deterministic runtime metadata.

    Returns
    -------
    dict of str to object
        Valid environment metadata for provenance tests.
    """
    return {
        "python_version": "3.12.1",
        "python_implementation": "CPython",
        "operating_system": "Linux",
        "operating_system_release": "6.1",
        "machine": "x86_64",
        "packages": {
            "datasets": "4.8.5",
            "flwr": "1.29.0",
            "keras": "3.14.0",
            "numpy": "2.2.0",
            "tensorflow": "2.20.0",
        },
    }


class RunProvenanceTests(unittest.TestCase):
    def test_supplied_public_snapshot_is_not_reopened_for_provenance(self) -> None:
        from src.app_manifest import load_app_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            public_dir = Path(tmpdir) / "public"
            public_dir.mkdir()
            protocol = write_public_dataset_contract(public_dir)
            snapshot = load_app_manifest(
                public_artifact_dir=public_dir, protocol=protocol
            )
            with patch(
                "src.run_provenance.load_app_manifest",
                side_effect=AssertionError("public pointer reopened"),
            ):
                metadata = _dataset_metadata(public_dir, app_manifest=snapshot)

        self.assertEqual(metadata["status"], "available")
        self.assertEqual(
            metadata["checksums"]["manifest.json"],
            "sha256:" + hashlib.sha256(snapshot.manifest_bytes).hexdigest(),
        )

    def test_public_checksums_use_the_same_snapshot_as_validation(self) -> None:
        from src.app_manifest import load_app_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            public_dir = Path(tmpdir) / "public"
            public_dir.mkdir()
            protocol = write_public_dataset_contract(public_dir)
            manifest_path = public_dir / "manifest.json"
            vocabulary_path = public_dir / "vocab.txt"
            manifest_bytes = manifest_path.read_bytes()
            vocabulary_bytes = vocabulary_path.read_bytes()

            def load_then_mutate(**kwargs):
                snapshot = load_app_manifest(protocol=protocol, **kwargs)
                mutated_manifest = manifest_path.read_bytes().replace(
                    b'"rows": 25000', b'"rows": 35000'
                )
                self.assertEqual(len(mutated_manifest), len(manifest_bytes))
                manifest_path.write_bytes(mutated_manifest)
                vocabulary_path.write_bytes(
                    vocabulary_path.read_bytes().replace(b"good", b"evil")
                )
                return snapshot

            with patch(
                "src.run_provenance.load_app_manifest",
                side_effect=load_then_mutate,
            ):
                metadata = _dataset_metadata(public_dir)

        self.assertEqual(
            metadata["checksums"]["manifest.json"],
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
        )
        self.assertEqual(
            metadata["checksums"]["vocab.txt"],
            "sha256:" + hashlib.sha256(vocabulary_bytes).hexdigest(),
        )
        self.assertEqual(json.loads(metadata["identity"])["rows"], 25000)

    def test_linux_environment_uses_tensorflow_cpu_distribution_version(self) -> None:
        versions = {
            "datasets": "4.8.5",
            "flwr": "1.29.0",
            "keras": "3.14.0",
            "numpy": "2.2.0",
            "tensorflow-cpu": "2.20.0",
        }

        def distribution_version(name: str) -> str:
            if name not in versions:
                raise importlib.metadata.PackageNotFoundError(name)
            return versions[name]

        with (
            patch("src.run_provenance.platform.system", return_value="Linux"),
            patch(
                "src.run_provenance.importlib.metadata.version",
                side_effect=distribution_version,
            ) as version,
        ):
            environment = _environment_metadata()

        self.assertEqual(environment["packages"]["tensorflow"], "2.20.0")
        self.assertIn(unittest.mock.call("tensorflow-cpu"), version.call_args_list)

    def test_non_linux_environment_does_not_use_tensorflow_cpu_distribution(
        self,
    ) -> None:
        def distribution_version(name: str) -> str:
            if name == "tensorflow":
                raise importlib.metadata.PackageNotFoundError(name)
            return "1.0"

        with (
            patch("src.run_provenance.platform.system", return_value="Darwin"),
            patch(
                "src.run_provenance.importlib.metadata.version",
                side_effect=distribution_version,
            ) as version,
        ):
            environment = _environment_metadata()

        self.assertIsNone(environment["packages"]["tensorflow"])
        self.assertNotIn(unittest.mock.call("tensorflow-cpu"), version.call_args_list)

    def test_manifest_records_run_inputs_and_public_dataset_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            public_dir = root / "public"
            server_dir = root / "server"
            public_dir.mkdir()
            protocol = write_public_dataset_contract(public_dir)
            run_config = {
                "num-server-rounds": 3,
                "public-artifact-dir": public_dir,
                "random-seed": 19,
                "use-huber": True,
            }

            with (
                patch(
                    "src.run_provenance.uuid.uuid4",
                    return_value="12345678-1234-4234-9234-123456789abc",
                ),
                patch(
                    "src.run_provenance._code_revision",
                    return_value={
                        "commit": "a" * 40,
                        "dirty": False,
                        "source": "git",
                    },
                ),
                patch(
                    "src.run_provenance._environment_metadata",
                    return_value=runtime_environment(),
                ),
                patch(
                    "src.app_manifest.load_scientific_protocol",
                    return_value=protocol,
                ),
            ):
                manifest_path = write_run_provenance_manifest(
                    server_dir,
                    run_config,
                    public_artifact_dir=public_dir,
                    flower_run_id=42,
                    created_at="2026-07-21T20:00:00Z",
                )

            payload = load_run_provenance_manifest(manifest_path)
            self.assertEqual(payload["schema_version"], ARTIFACT_SCHEMA_VERSION)
            self.assertEqual(payload["run_id"], "12345678-1234-4234-9234-123456789abc")
            self.assertEqual(payload["flower_run_id"], 42)
            self.assertEqual(payload["created_at"], "2026-07-21T20:00:00Z")
            self.assertEqual(
                payload["run_config"],
                {
                    "num-server-rounds": 3,
                    "public-artifact-dir": str(public_dir),
                    "random-seed": 19,
                    "use-huber": True,
                },
            )
            self.assertEqual(payload["environment"], runtime_environment())
            self.assertEqual(payload["code_revision"]["commit"], "a" * 40)
            self.assertEqual(payload["seeds"]["run_config"], {"random-seed": 19})
            self.assertEqual(
                payload["dataset"]["identity"],
                json.dumps(
                    {
                        "id": "stanfordnlp/imdb",
                        "config": "plain_text",
                        "revision": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
                        "datasets_version": "4.8.5",
                        "split": "train",
                        "rows": 25000,
                        "raw_parquet_sha256": "db47d16b" + "0" * 56,
                        "content_sha256": "4639bf10" + "0" * 56,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
            manifest_bytes = (public_dir / "manifest.json").read_bytes()
            self.assertEqual(
                payload["dataset"]["checksums"]["manifest.json"],
                "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
            )
            self.assertEqual(
                payload["dataset"]["private_client_shards"]["status"],
                "not_collected",
            )

    def test_hostile_public_identity_is_rejected_and_not_recorded(self) -> None:
        for field, value in (("split", "test"), ("id", "attacker/test-derived")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                public_dir = root / "public"
                server_dir = root / "server"
                public_dir.mkdir()
                protocol = write_public_dataset_contract(public_dir)
                manifest_path = public_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["dataset"][field] = value
                manifest_path.write_bytes(canonical_json_bytes(manifest))

                with (
                    patch(
                        "src.app_manifest.load_scientific_protocol",
                        return_value=protocol,
                    ),
                    self.assertRaisesRegex(ValueError, "dataset identity"),
                ):
                    write_run_provenance_manifest(
                        server_dir,
                        {},
                        public_artifact_dir=public_dir,
                    )

                self.assertFalse((server_dir / "run_manifest.json").exists())

    def test_hostile_public_vocabulary_is_rejected_and_not_recorded(self) -> None:
        mutations = {
            "path": lambda manifest, vocabulary: manifest["vocabulary"].__setitem__(
                "filename", "../vocab.txt"
            ),
            "length": lambda manifest, vocabulary: manifest["vocabulary"].__setitem__(
                "size_bytes", vocabulary.stat().st_size + 1
            ),
            "declared checksum": lambda manifest, vocabulary: manifest[
                "vocabulary"
            ].__setitem__("sha256", "f" * 64),
            "artifact checksum": lambda manifest, vocabulary: vocabulary.write_bytes(
                vocabulary.read_bytes() + b"attacker"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                public_dir = root / "public"
                server_dir = root / "server"
                public_dir.mkdir()
                protocol = write_public_dataset_contract(public_dir)
                manifest_path = public_dir / "manifest.json"
                vocabulary_path = public_dir / "vocab.txt"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest, vocabulary_path)
                manifest_path.write_bytes(canonical_json_bytes(manifest))

                with (
                    patch(
                        "src.app_manifest.load_scientific_protocol",
                        return_value=protocol,
                    ),
                    self.assertRaises(ValueError),
                ):
                    write_run_provenance_manifest(
                        server_dir,
                        {},
                        public_artifact_dir=public_dir,
                    )

                self.assertFalse((server_dir / "run_manifest.json").exists())

    def test_manifest_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            with (
                patch(
                    "src.run_provenance.uuid.uuid4",
                    side_effect=[
                        "12345678-1234-4234-9234-123456789abc",
                        "22345678-1234-4234-9234-123456789abc",
                    ],
                ),
                patch(
                    "src.run_provenance._code_revision",
                    return_value={
                        "commit": None,
                        "dirty": None,
                        "source": "unavailable",
                    },
                ),
                patch(
                    "src.run_provenance._environment_metadata",
                    return_value=runtime_environment(),
                ),
            ):
                manifest_path = write_run_provenance_manifest(path, {})
                with self.assertRaises(FileExistsError):
                    write_run_provenance_manifest(path, {})

            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"],
                "12345678-1234-4234-9234-123456789abc",
            )

    def test_distinct_runs_receive_distinct_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch(
                    "src.run_provenance._code_revision",
                    return_value={
                        "commit": None,
                        "dirty": None,
                        "source": "unavailable",
                    },
                ),
                patch(
                    "src.run_provenance._environment_metadata",
                    return_value=runtime_environment(),
                ),
            ):
                first = load_run_provenance_manifest(
                    write_run_provenance_manifest(root / "first", {})
                )
                second = load_run_provenance_manifest(
                    write_run_provenance_manifest(root / "second", {})
                )

            self.assertNotEqual(first["run_id"], second["run_id"])

    def test_invalid_run_config_is_rejected_before_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            with self.assertRaisesRegex(ValueError, "run configuration"):
                write_run_provenance_manifest(path, {"nested": {"unsafe": True}})

            self.assertFalse((path / "run_manifest.json").exists())

    def test_loader_rejects_incomplete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_manifest.json"
            path.write_text(
                json.dumps({"schema_version": ARTIFACT_SCHEMA_VERSION}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing"):
                load_run_provenance_manifest(path)

    def test_loader_rejects_hostile_json_and_accepts_finite_additions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch(
                    "src.run_provenance._code_revision",
                    return_value={
                        "commit": None,
                        "dirty": None,
                        "source": "unavailable",
                    },
                ),
                patch(
                    "src.run_provenance._environment_metadata",
                    return_value=runtime_environment(),
                ),
            ):
                manifest_path = write_run_provenance_manifest(
                    root,
                    {},
                    run_id="12345678-1234-4234-9234-123456789abc",
                )
            valid = manifest_path.read_text(encoding="utf-8")
            hostile_documents = (
                valid.replace(
                    '"schema_version": 1,',
                    '"schema_version": 0,\n  "schema_version": 1,',
                    1,
                ),
                valid.replace(
                    '"run_id": "12345678-1234-4234-9234-123456789abc",',
                    '"run_id": "attacker",\n  '
                    '"run_id": "12345678-1234-4234-9234-123456789abc",',
                    1,
                ),
                valid.replace(
                    '"identity": null,',
                    '"identity": "attacker",\n    "identity": null,',
                    1,
                ),
                *(
                    valid[:-2] + f',\n  "producer": {constant}\n}}\n'
                    for constant in (
                        "NaN",
                        "Infinity",
                        "-Infinity",
                        "1e999",
                        "-1e999",
                    )
                ),
            )
            for document in hostile_documents:
                with self.subTest(document=document):
                    manifest_path.write_text(document, encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "invalid run provenance manifest"
                    ):
                        load_run_provenance_manifest(manifest_path)

            finite = (
                valid[:-2] + ',\n  "producer": {"upper": 1e308, "lower": -1e308}\n}\n'
            )
            manifest_path.write_text(finite, encoding="utf-8")
            self.assertEqual(
                load_run_provenance_manifest(manifest_path)["producer"],
                {"upper": 1e308, "lower": -1e308},
            )

    def test_loader_rejects_missing_and_invalid_nested_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch(
                    "src.run_provenance._code_revision",
                    return_value={
                        "commit": "a" * 40,
                        "dirty": False,
                        "source": "git",
                    },
                ),
                patch(
                    "src.run_provenance._environment_metadata",
                    return_value=runtime_environment(),
                ),
            ):
                payload = dict(
                    load_run_provenance_manifest(
                        write_run_provenance_manifest(root / "valid", {})
                    )
                )

            invalid_values = {
                "environment": {**payload["environment"], "packages": []},
                "code_revision": {**payload["code_revision"], "dirty": "false"},
                "seeds": {**payload["seeds"], "code_defaults": {}},
                "dataset": {
                    **payload["dataset"],
                    "private_client_shards": {"status": "not_collected"},
                },
            }
            for field, invalid_value in invalid_values.items():
                with self.subTest(field=field):
                    manifest_path = root / f"invalid-{field}.json"
                    manifest_path.write_text(
                        json.dumps({**payload, field: invalid_value}), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, field):
                        load_run_provenance_manifest(manifest_path)

    def test_git_revision_marks_untracked_files_dirty(self) -> None:
        completed = [
            type("Result", (), {"stdout": "a" * 40 + "\n"})(),
            type("Result", (), {"stdout": "?? untracked.txt\n"})(),
        ]
        with (
            patch.dict("src.run_provenance.os.environ", {}, clear=True),
            patch("src.run_provenance.subprocess.run", side_effect=completed) as run,
        ):
            revision = _code_revision()

        self.assertTrue(revision["dirty"])
        self.assertEqual(
            run.call_args_list[1].args[0], ["git", "status", "--porcelain"]
        )


if __name__ == "__main__":
    unittest.main()
