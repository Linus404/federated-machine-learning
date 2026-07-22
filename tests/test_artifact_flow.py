import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from collections import Counter
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.app_manifest import AppManifest, load_app_manifest
from src.artifact_compatibility import (
    CLIENT_SHARD_SCHEMA_VERSION,
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    RetainedDirectoryChain,
    canonical_json_bytes,
    sha256_bytes,
)
from src.contracts import canonical_client_row_bytes, client_shard_metadata
from src.data_prep import (
    _acquire_preparation_lock,
    _preflight_output_root,
    _publish_prepared_roots,
    _recover_prepared_migration,
    _recover_preparation_stage,
    _validate_recovery_generation,
    build_vectorizer,
    main,
    package_raw_client_shards,
    prepare_all,
    publish_public_artifacts,
)
from src.evaluation_artifact import (
    canonical_source_row_bytes,
    publish_evaluation_artifact,
)
from src.local_training import (
    build_model_from_manifest,
    load_client_shard,
    load_client_shard_snapshot,
)
from src.paths import (
    PREPARED_GENERATION_SCHEMA_VERSION,
    prepared_generation_inventory,
    resolve_prepared_artifact_dir,
)
from src.text_preprocessing import create_text_vectorizer, protocol_standardize


def write_public_artifacts(path: Path, sequence_length: int = 4) -> AppManifest:
    vocabulary = b"\n[UNK]\ngood\nbad\nmovie\n"
    vocabulary_sha256 = hashlib.sha256(vocabulary).hexdigest()
    (path / "vocab.txt").write_bytes(vocabulary)
    train_rows = [
        {"text": "bad", "label": 0},
        {"text": "good", "label": 1},
        {"text": "bad movie", "label": 0},
        {"text": "good movie", "label": 1},
    ]
    train_content = b"".join(
        canonical_source_row_bytes(str(row["text"]), int(row["label"]))
        for row in train_rows
    )
    payload = {
        "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "embedding_dim": 100,
        "sequence_length": sequence_length,
        "vocabulary_size": 5,
        "vocabulary": {
            "filename": "vocab.txt",
            "sha256": vocabulary_sha256,
            "size_bytes": len(vocabulary),
        },
        "dataset": {
            "id": "example/imdb",
            "config": "plain_text",
            "revision": "frozen",
            "datasets_version": "1.0.0",
            "split": "train",
            "rows": 4,
            "raw_parquet_sha256": "1" * 64,
            "content_sha256": hashlib.sha256(train_content).hexdigest(),
        },
    }
    manifest_path = path / "manifest.json"
    manifest_bytes = canonical_json_bytes(payload)
    manifest_path.write_bytes(manifest_bytes)
    return AppManifest(
        payload,
        path / "vocab.txt",
        manifest_bytes,
        vocabulary,
        tuple(vocabulary.decode("utf-8")[:-1].split("\n")),
    )


def public_protocol(
    sequence_length: int = 4, *, duplicate_split_content: bool = False
) -> dict[str, object]:
    """Return the small frozen public-artifact protocol used by flow tests.

    Parameters
    ----------
    sequence_length : int, optional
        Frozen model input length.
    duplicate_split_content : bool, optional
        Give one official test row the same content as an official train row.

    Returns
    -------
    dict of str to object
        Dataset and vocabulary identity matching ``write_public_artifacts``.
    """
    vocabulary = b"\n[UNK]\ngood\nbad\nmovie\n"
    negative_test_text = "bad" if duplicate_split_content else "untouched negative"
    evaluation_rows = [
        {"text": negative_test_text, "label": 0},
        {"text": "untouched positive", "label": 1},
    ]
    evaluation_content = b"".join(
        canonical_source_row_bytes(str(row["text"]), int(row["label"]))
        for row in evaluation_rows
    )
    train_rows = [
        {"text": "bad", "label": 0},
        {"text": "good", "label": 1},
        {"text": "bad movie", "label": 0},
        {"text": "good movie", "label": 1},
    ]
    train_content = b"".join(
        canonical_source_row_bytes(str(row["text"]), int(row["label"]))
        for row in train_rows
    )
    return {
        "dataset": {
            "id": "example/imdb",
            "config": "plain_text",
            "revision": "frozen",
            "datasets_version": "1.0.0",
            "splits": {
                "train": {
                    "rows": 4,
                    "label_counts": [2, 2],
                    "raw_parquet_sha256": "1" * 64,
                    "content_sha256": hashlib.sha256(train_content).hexdigest(),
                },
                "test": {
                    "rows": 2,
                    "label_counts": [1, 1],
                    "raw_parquet_sha256": "3" * 64,
                    "content_sha256": hashlib.sha256(evaluation_content).hexdigest(),
                },
            },
        },
        "preprocessing": {
            "vocabulary_size": 5,
            "max_tokens": 5,
            "output_sequence_length": sequence_length,
            "vocabulary_sha256": hashlib.sha256(vocabulary).hexdigest(),
        },
        "model": {
            "vocabulary_size": 5,
            "sequence_length": sequence_length,
            "embedding_dimension": 100,
        },
    }


def preparation_request(partitions: int = 1) -> dict[str, int]:
    """Return the durable artifact-affecting preparation request.

    Parameters
    ----------
    partitions : int, optional
        Requested number of client shards.

    Returns
    -------
    dict of str to int
        Canonical request fields persisted with a prepared generation.
    """
    return {"partitions": partitions}


def write_complete_prepared_stage(
    path: Path, *, partitions: int = 1, duplicate_split_content: bool = False
) -> dict[str, object]:
    """Write one complete prepared generation candidate for recovery tests.

    Parameters
    ----------
    path : pathlib.Path
        New generation staging or final directory.
    partitions : int, optional
        Exact number of non-empty client shards to write.
    duplicate_split_content : bool, optional
        Give one official test row the same content as an official train row.

    Returns
    -------
    dict of str to object
        Frozen-protocol fixture matching every generated artifact.
    """
    protocol = public_protocol(duplicate_split_content=duplicate_split_content)
    negative_test_text = "bad" if duplicate_split_content else "untouched negative"
    public = path / "public"
    public.mkdir(parents=True)
    manifest = write_public_artifacts(public)
    client = path / "client"
    client.mkdir()
    records = [
        {"text": "bad", "label": 0},
        {"text": "good", "label": 1},
        {"text": "bad movie", "label": 0},
        {"text": "good movie", "label": 1},
    ]
    if partitions not in {1, 4}:
        raise ValueError("flow fixture supports exactly one or four partitions")
    for client_id in range(partitions):
        source_indices = list(range(4)) if partitions == 1 else [client_id]
        write_client_artifacts(
            client / f"client-{client_id}",
            manifest,
            [records[index] for index in source_indices],
            client_id=client_id,
            source_indices=source_indices,
        )
    publish_evaluation_artifact(
        [
            {"text": negative_test_text, "label": 0},
            {"text": "untouched positive", "label": 1},
        ],
        path / "evaluation",
        protocol=protocol,
    )
    return protocol


def write_pending_prepared_recovery(
    root: Path,
    *,
    final: bool,
    legacy_schema: bool = False,
    partitions: int = 1,
    duplicate_split_content: bool = False,
) -> tuple[dict[str, Path], Path, dict[str, object]]:
    """Create one journaled stage or final generation recovery fixture.

    Parameters
    ----------
    root : pathlib.Path
        Empty temporary artifact parent.
    final : bool
        Write the candidate under its final UUID name instead of its stage name.
    legacy_schema : bool, optional
        Write schema 1 metadata without a durable preparation request.
    partitions : int, optional
        Exact number of client shards bound by the candidate request.
    duplicate_split_content : bool, optional
        Give one official test row the same content as an official train row.

    Returns
    -------
    tuple
        Logical roots, candidate generation path, and matching protocol fixture.
    """
    roots = {
        "client": root / "clients",
        "public": root / "public",
        "evaluation": root / "evaluation",
    }
    for name, logical_root in roots.items():
        logical_root.mkdir()
        (logical_root / "legacy.txt").write_text(name, encoding="utf-8")
    generation_id = "4c03285d-83b3-45d1-bc29-00a516eeef93"
    stage_name = ".prepare-recovery.staging"
    generations = root / ".prepared-generations"
    candidate = generations / (generation_id if final else stage_name)
    protocol = write_complete_prepared_stage(
        candidate,
        partitions=partitions,
        duplicate_split_content=duplicate_split_content,
    )
    schema_version = 1 if legacy_schema else PREPARED_GENERATION_SCHEMA_VERSION
    index = {
        "schema_version": schema_version,
        "generation_id": generation_id,
        **(
            {"inventory": prepared_generation_inventory(candidate)}
            if not legacy_schema
            else {}
        ),
        "logical_roots": {name: roots[name].name for name in sorted(roots)},
        **(
            {}
            if legacy_schema
            else {"preparation_request": preparation_request(partitions)}
        ),
    }
    (candidate / "index.json").write_bytes(canonical_json_bytes(index))
    index_bytes = (candidate / "index.json").read_bytes()
    parent_stat = root.stat()
    generations_stat = candidate.parent.stat()
    candidate_stat = candidate.stat()
    journal = {
        "schema_version": schema_version,
        "generation_id": generation_id,
        "stage_name": stage_name,
        **(
            {"generation_index_checksum": sha256_bytes(index_bytes)}
            if not legacy_schema
            else {}
        ),
        **(
            {
                "parent_device": parent_stat.st_dev,
                "parent_inode": parent_stat.st_ino,
                "generations_device": generations_stat.st_dev,
                "generations_inode": generations_stat.st_ino,
                "stage_device": candidate_stat.st_dev,
                "stage_inode": candidate_stat.st_ino,
            }
            if not legacy_schema
            else {}
        ),
        "logical_roots": {name: roots[name].name for name in sorted(roots)},
        **(
            {}
            if legacy_schema
            else {"preparation_request": preparation_request(partitions)}
        ),
        "legacy_roots": sorted(roots),
        "alias_roots": sorted(roots),
        "previous_pointer_target": None,
    }
    (root / ".prepared-migration.json").write_bytes(canonical_json_bytes(journal))
    return roots, candidate, protocol


CRASHING_PREPARED_PUBLICATION = textwrap.dedent(
    """
    import json
    import os
    import sys
    from pathlib import Path

    import src.data_prep as data_prep

    root = Path(sys.argv[1])
    phase = sys.argv[2]
    roots = {
        "client": root / "clients",
        "public": root / "public",
        "evaluation": root / "evaluation",
    }
    stage = root / ".prepared-generations" / ".prepare-crash.staging"
    stages = {name: stage / name for name in roots}
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    data_prep.load_scientific_protocol = lambda: protocol

    def crash_at(value):
        if value == phase:
            os._exit(91)

    data_prep._publication_checkpoint = crash_at
    data_prep._publish_prepared_roots(roots, stages, {"partitions": 1})
    """
)


def crash_prepared_publication(root: Path, phase: str) -> None:
    """Terminate prepared publication at one durable checkpoint.

    Parameters
    ----------
    root : pathlib.Path
        Disposable artifact parent containing a complete crash staging tree.
    phase : str
        Publication checkpoint at which the child must terminate.

    Returns
    -------
    None
    """
    completed = subprocess.run(
        [sys.executable, "-c", CRASHING_PREPARED_PUBLICATION, str(root), phase],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 91:
        raise AssertionError(
            f"publication did not crash at {phase}: {completed.stderr}"
        )


def write_client_artifacts(
    path: Path,
    manifest: AppManifest,
    records: list[dict[str, object]],
    *,
    client_id: int = 0,
    source_indices: list[int] | None = None,
) -> Path:
    """Write one canonical client shard bound to a public manifest snapshot.

    Parameters
    ----------
    path : pathlib.Path
        New shard directory.
    manifest : AppManifest
        Public artifact snapshot to bind.
    records : list of dict
        Review text and binary labels in official-index order.
    client_id : int, optional
        Expected client identity.
    source_indices : list of int or None, optional
        Official train row identities corresponding to ``records``.

    Returns
    -------
    pathlib.Path
        Written client shard directory.
    """
    path.mkdir()
    indices = source_indices or list(range(len(records)))
    records_bytes = b"".join(
        canonical_client_row_bytes(
            f"train:{source_index}", str(record["text"]), int(record["label"])
        )
        for source_index, record in zip(indices, records, strict=True)
    )
    (path / "reviews.jsonl").write_bytes(records_bytes)
    metadata = client_shard_metadata(
        client_id,
        [int(record["label"]) for record in records],
        records_bytes=records_bytes,
        public_manifest_bytes=manifest.manifest_bytes,
        dataset=manifest.payload["dataset"],
    )
    (path / "client_metadata.json").write_bytes(canonical_json_bytes(metadata))
    return path


def check_public_manifest_loads_the_model_shape(tmp_path: Path) -> None:
    write_public_artifacts(tmp_path, sequence_length=500)

    manifest = load_app_manifest(
        public_artifact_dir=tmp_path, protocol=public_protocol(500)
    )

    assert manifest.vocabulary_path == tmp_path.resolve() / "vocab.txt"
    assert manifest.payload["embedding_dim"] == 100


def check_raw_client_packaging_keeps_every_sample_private(tmp_path: Path) -> None:
    texts = np.asarray([f"review {index}\nline" for index in range(12)])
    labels = np.asarray([0, 1] * 6)

    public_dir = tmp_path / "public"
    public_dir.mkdir()
    manifest = write_public_artifacts(public_dir)
    shards = package_raw_client_shards(
        texts,
        labels,
        tmp_path,
        num_clients=4,
        manifest=manifest.payload,
        dataset=manifest.payload["dataset"],
    )

    packaged = []
    sample_count = 0
    for client_id, shard_path in enumerate(shards):
        metadata = json.loads(
            (shard_path / "client_metadata.json").read_text(encoding="utf-8")
        )
        records = [
            json.loads(line)
            for line in (shard_path / "reviews.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert metadata["client_id"] == client_id
        assert metadata["schema_version"] == CLIENT_SHARD_SCHEMA_VERSION
        assert metadata["sample_count"] == len(records)
        assert metadata["split_seed"] == 67
        assert metadata["alpha"] == 0.5
        sample_count += len(records)
        packaged.extend((record["text"], record["label"]) for record in records)

    assert sample_count == len(texts)
    assert Counter(packaged) == Counter(zip(texts.tolist(), labels.tolist()))


def check_client_tokenizes_its_raw_reviews_with_public_vocabulary(
    tmp_path: Path,
) -> None:
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    manifest = write_public_artifacts(public_dir)
    records = [
        {"text": "good movie", "label": 1},
        {"text": "bad movie", "label": 0},
        {"text": "unknown movie", "label": 0},
        {"text": "good", "label": 1},
    ]
    client_dir = write_client_artifacts(tmp_path / "client-0", manifest, records)

    (train_x, train_y), (val_x, val_y) = load_client_shard(
        client_dir, manifest, 0, validation_split=0.25
    )

    assert train_x.dtype == np.int32
    assert train_y.dtype == np.float32
    assert train_x.shape == (2, 4)
    assert val_x.shape == (2, 4)
    combined_x = np.concatenate([train_x, val_x])
    assert any(np.array_equal(row, [2, 4, 0, 0]) for row in combined_x)
    assert any(np.array_equal(row, [3, 4, 0, 0]) for row in combined_x)
    assert any(np.array_equal(row, [1, 4, 0, 0]) for row in combined_x)
    np.testing.assert_array_equal(np.sort(train_y), [0, 1])
    np.testing.assert_array_equal(np.sort(val_y), [0, 1])
    model = build_model_from_manifest(manifest)
    assert model.input_shape == (None, 4)
    assert model.get_layer("token_embedding").trainable


def check_client_loader_preserves_unicode_line_separator_in_review(
    tmp_path: Path,
) -> None:
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    manifest = write_public_artifacts(public_dir)
    records = [
        {"text": "good\u2028movie", "label": 1},
        {"text": "bad movie", "label": 0},
    ]
    client_dir = write_client_artifacts(tmp_path / "client-0", manifest, records)

    (train_x, train_y), (val_x, val_y) = load_client_shard(
        client_dir, manifest, 0, validation_split=0.5
    )

    assert train_x.shape == (1, 4)
    assert val_x.shape == (1, 4)
    np.testing.assert_array_equal(np.sort(np.concatenate([train_y, val_y])), [0, 1])


def check_client_loader_preserves_control_character_in_vocabulary(
    tmp_path: Path,
) -> None:
    public_dir = tmp_path / "public"
    public_dir.mkdir()
    manifest = write_public_artifacts(public_dir)
    (public_dir / "vocab.txt").write_text(
        "\n[UNK]\ngood\nbad\nmovie\n\x85\nrare", encoding="utf-8"
    )
    records = [
        {"text": "good movie", "label": 1},
        {"text": "bad movie", "label": 0},
    ]
    client_dir = write_client_artifacts(tmp_path / "client-0", manifest, records)

    (train_x, train_y), (val_x, val_y) = load_client_shard(
        client_dir, manifest, 0, validation_split=0.5
    )

    assert train_x.shape == (1, 4)
    assert val_x.shape == (1, 4)
    np.testing.assert_array_equal(np.sort(np.concatenate([train_y, val_y])), [0, 1])


def check_cli_prepares_all_artifacts_with_one_dataset_load(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    client_dir = tmp_path / "clients"
    evaluation_dir = tmp_path / "evaluation"
    public_dir.mkdir()
    client_dir.mkdir()
    evaluation_dir.mkdir()
    sentinels = {
        "public": public_dir / "legacy-test-export.txt",
        "client": client_dir / "legacy-client-export.txt",
        "evaluation": evaluation_dir / "legacy-evaluation-export.txt",
    }
    for sentinel in sentinels.values():
        sentinel.write_text("external test sentinel", encoding="utf-8")
    unbound_stage = tmp_path / ".prepared-generations" / ".prepare-unbound.staging"
    unbound_stage.mkdir(parents=True)
    (unbound_stage / "preserve.bin").write_bytes(b"unbound preparation residue")
    (client_dir / "client-0.tar.gz").write_bytes(b"legacy private shard")
    texts = np.asarray([f"Review {index} good movie" for index in range(12)])
    labels = np.asarray([index % 2 for index in range(12)], dtype="int32")
    test_rows = [
        {"text": "untouched negative review", "label": 0},
        {"text": "untouched positive review", "label": 1},
    ]

    class Split:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, key):
            return [row[key] for row in self.rows]

        def __iter__(self):
            return iter(self.rows)

    class Vectorizer:
        def get_vocabulary(self):
            return ["", "[UNK]", "review", "good", "movie"]

    content = b"".join(
        canonical_source_row_bytes(row["text"], row["label"]) for row in test_rows
    )
    train_content = b"".join(
        canonical_source_row_bytes(str(text), int(label))
        for text, label in zip(texts, labels, strict=True)
    )
    protocol = {
        "dataset": {
            "id": "stanfordnlp/imdb",
            "config": "plain_text",
            "revision": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
            "datasets_version": "4.8.5",
            "splits": {
                "train": {
                    "rows": 12,
                    "label_counts": [6, 6],
                    "raw_parquet_sha256": "1" * 64,
                    "content_sha256": hashlib.sha256(train_content).hexdigest(),
                },
                "test": {
                    "rows": 2,
                    "label_counts": [1, 1],
                    "raw_parquet_sha256": "3" * 64,
                    "content_sha256": hashlib.sha256(content).hexdigest(),
                },
            },
        },
        "preprocessing": {
            "vocabulary_size": 5,
            "max_tokens": 5,
            "output_sequence_length": 500,
            "vocabulary_sha256": hashlib.sha256(
                b"\n[UNK]\nreview\ngood\nmovie\n"
            ).hexdigest(),
        },
        "model": {
            "vocabulary_size": 5,
            "sequence_length": 500,
            "embedding_dimension": 100,
        },
        "framework": {
            "tensorflow_version": "2.20.0",
            "keras_version": "3.14.0",
            "numpy_version": "2.4.4",
        },
    }
    dataset = {
        "train": Split(
            [
                {"text": str(text), "label": int(label)}
                for text, label in zip(texts, labels)
            ]
        ),
        "test": Split(test_rows),
        "unsupervised": object(),
    }

    with (
        patch(
            "src.data_prep.load_verified_imdb_dataset", return_value=dataset
        ) as load_dataset,
        patch("src.data_prep.build_vectorizer", return_value=Vectorizer()),
        patch("src.data_prep.load_scientific_protocol", return_value=protocol),
    ):
        main(
            [
                "--partitions",
                "4",
                "--client-shard-dir",
                str(client_dir),
                "--public-artifact-dir",
                str(public_dir),
                "--evaluation-artifact-dir",
                str(evaluation_dir),
            ]
        )

    load_dataset.assert_called_once_with()
    assert (unbound_stage / "preserve.bin").read_bytes() == (
        b"unbound preparation residue"
    )
    public_generation = resolve_prepared_artifact_dir(public_dir, "public")
    client_generation = resolve_prepared_artifact_dir(client_dir, "client")
    evaluation_generation = resolve_prepared_artifact_dir(evaluation_dir, "evaluation")
    assert (
        public_generation.parent
        == client_generation.parent
        == evaluation_generation.parent
    )
    assert not list(tmp_path.rglob("*.npy"))
    manifest = json.loads(
        (public_generation / "manifest.json").read_text(encoding="utf-8")
    )
    vocabulary = (
        (public_generation / "vocab.txt").read_text(encoding="utf-8").splitlines()
    )
    assert manifest["vocabulary_size"] == len(vocabulary)
    assert manifest["schema_version"] == PUBLIC_ARTIFACT_SCHEMA_VERSION
    assert "glove" not in json.dumps(manifest).lower()
    assert "embedding_matrix" not in manifest
    assert not list(client_generation.glob("client-*.tar.gz"))
    assert all(
        (client_generation / f"client-{index}" / "reviews.jsonl").exists()
        for index in range(4)
    )
    client_records = [
        json.loads(line)
        for path in client_generation.glob("client-*/reviews.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    evaluation_records = [
        json.loads(line)
        for line in (evaluation_generation / "test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {record["row_id"] for record in client_records}.isdisjoint(
        record["row_id"] for record in evaluation_records
    )
    assert all(record["row_id"].startswith("train:") for record in client_records)
    assert all(record["row_id"].startswith("test:") for record in evaluation_records)
    untouched_texts = {row["text"] for row in test_rows}
    assert untouched_texts.isdisjoint(record["text"] for record in client_records)
    public_bytes = b"".join(path.read_bytes() for path in public_generation.iterdir())
    assert all(text.encode() not in public_bytes for text in untouched_texts)
    assert {path.name for path in public_generation.iterdir()} == {
        "manifest.json",
        "vocab.txt",
    }
    assert {path.name for path in client_generation.iterdir()} == {
        f"client-{index}" for index in range(4)
    }
    assert {path.name for path in evaluation_generation.iterdir()} == {
        "manifest.json",
        "test.jsonl",
    }
    assert not list((tmp_path / ".prepared-current").rglob("*export*"))
    archive = next((tmp_path / ".prepared-legacy").iterdir())
    for kind, sentinel in sentinels.items():
        archived_root = client_dir.name if kind == "client" else kind
        assert (archive / archived_root / sentinel.name).read_text(
            encoding="utf-8"
        ) == "external test sentinel"


class ArtifactFlowTests(unittest.TestCase):
    def test_preparation_recovers_only_state_bound_stage_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            generations = Path(tmpdir) / ".prepared-generations"
            nonce = "2" * 32
            stage = generations / f".prepare-{nonce}.staging"
            unbound = generations / ".prepare-unbound.staging"
            stage.mkdir(parents=True)
            unbound.mkdir()
            (stage / "partial").write_bytes(b"owned")
            (unbound / "partial").write_bytes(b"unbound")
            parent_stat = generations.stat()
            stage_stat = stage.stat()
            state = {
                "schema_version": 1,
                "operation": "build",
                "parent_device": parent_stat.st_dev,
                "parent_inode": parent_stat.st_ino,
                "nonce": nonce,
                "stage_name": stage.name,
                "device": stage_stat.st_dev,
                "inode": stage_stat.st_ino,
                "tombstone_name": None,
            }
            state_path = generations / ".prepare-stage.state"
            state_path.write_bytes(canonical_json_bytes(state))
            state_path.chmod(0o600)

            chain = RetainedDirectoryChain.open(generations, check_platform=False)
            try:
                _recover_preparation_stage(chain)
            finally:
                chain.close()

            self.assertFalse(stage.exists())
            self.assertFalse(state_path.exists())
            self.assertEqual((unbound / "partial").read_bytes(), b"unbound")

    def test_preparation_rejects_corrupt_state_without_residue_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            generations = Path(tmpdir) / ".prepared-generations"
            residue = generations / ".prepare-unbound.staging"
            residue.mkdir(parents=True)
            marker = residue / "preserve"
            marker.write_bytes(b"unbound")
            state = generations / ".prepare-stage.state"
            state.write_bytes(b'{"schema_version":1}\n')
            state.chmod(0o600)

            chain = RetainedDirectoryChain.open(generations, check_platform=False)
            try:
                with self.assertRaisesRegex(ValueError, "stage state"):
                    _recover_preparation_stage(chain)
            finally:
                chain.close()

            self.assertEqual(marker.read_bytes(), b"unbound")
            self.assertEqual(state.read_bytes(), b'{"schema_version":1}\n')

    def test_recovery_accepts_equal_content_with_distinct_split_identities(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots, candidate, protocol = write_pending_prepared_recovery(
                root, final=False, duplicate_split_content=True
            )
            train_record = next(
                row
                for row in (candidate / "client" / "client-0" / "reviews.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if json.loads(row)["text"] == "bad"
            )
            evaluation_record = next(
                row
                for row in (candidate / "evaluation" / "test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if json.loads(row)["text"] == "bad"
            )
            train_row = json.loads(train_record)
            evaluation_row = json.loads(evaluation_record)
            self.assertEqual(
                (train_row["text"], train_row["label"]),
                (evaluation_row["text"], evaluation_row["label"]),
            )
            self.assertTrue(train_row["row_id"].startswith("train:"))
            self.assertTrue(evaluation_row["row_id"].startswith("test:"))

            with patch("src.data_prep.load_scientific_protocol", return_value=protocol):
                self.assertTrue(
                    _recover_prepared_migration(roots, preparation_request())
                )

            self.assertTrue((root / ".prepared-current").is_symlink())

    def test_recovery_rejects_invalid_stage_and_final_without_visible_mutation(
        self,
    ) -> None:
        mutations = {
            "missing-index": (lambda candidate: candidate / "index.json", "missing"),
            "corrupt-index": (lambda candidate: candidate / "index.json", "corrupt"),
            "tampered-index": (
                lambda candidate: candidate / "index.json",
                "tampered",
            ),
            "missing-public": (
                lambda candidate: candidate / "public" / "manifest.json",
                "missing",
            ),
            "corrupt-public": (
                lambda candidate: candidate / "public" / "vocab.txt",
                "corrupt",
            ),
            "missing-client": (
                lambda candidate: (
                    candidate / "client" / "client-0" / "client_metadata.json"
                ),
                "missing",
            ),
            "corrupt-client": (
                lambda candidate: candidate / "client" / "client-0" / "reviews.jsonl",
                "corrupt",
            ),
            "missing-evaluation": (
                lambda candidate: candidate / "evaluation" / "test.jsonl",
                "missing",
            ),
            "corrupt-evaluation": (
                lambda candidate: candidate / "evaluation" / "test.jsonl",
                "corrupt",
            ),
            "extra-public": (
                lambda candidate: candidate / "public" / "legacy-test-export.txt",
                "extra",
            ),
            "extra-client": (
                lambda candidate: candidate / "client" / "legacy-client-export.txt",
                "extra",
            ),
            "extra-evaluation": (
                lambda candidate: (
                    candidate / "evaluation" / "legacy-evaluation-export.txt"
                ),
                "extra",
            ),
        }
        for final in (False, True):
            for mutation, (target_for, operation) in mutations.items():
                with (
                    self.subTest(
                        location="final" if final else "stage", mutation=mutation
                    ),
                    tempfile.TemporaryDirectory() as tmpdir,
                ):
                    root = Path(tmpdir)
                    roots, candidate, protocol = write_pending_prepared_recovery(
                        root, final=final
                    )
                    target = target_for(candidate)
                    original = target.read_bytes() if target.exists() else None
                    if operation == "missing":
                        target.unlink()
                    elif operation == "corrupt":
                        target.write_bytes(b"corrupt\n")
                    elif operation == "extra":
                        target.write_text("external test sentinel", encoding="utf-8")
                    else:
                        assert original is not None
                        tampered = json.loads(original)
                        tampered["generation_id"] = (
                            "e9c739c6-c67e-43e5-b570-6d5f48fe09b4"
                        )
                        target.write_bytes(canonical_json_bytes(tampered))

                    with (
                        patch(
                            "src.data_prep.load_scientific_protocol",
                            return_value=protocol,
                        ),
                        self.assertRaisesRegex(ValueError, "generation validation"),
                    ):
                        _recover_prepared_migration(roots, preparation_request())

                    self.assertTrue((root / ".prepared-migration.json").is_file())
                    self.assertFalse((root / ".prepared-current").exists())
                    for name, logical_root in roots.items():
                        self.assertTrue(logical_root.is_dir())
                        self.assertFalse(logical_root.is_symlink())
                        self.assertEqual(
                            (logical_root / "legacy.txt").read_text(encoding="utf-8"),
                            name,
                        )
                    self.assertTrue(candidate.exists())

                    if original is None:
                        target.unlink()
                    else:
                        target.write_bytes(original)
                    with patch(
                        "src.data_prep.load_scientific_protocol",
                        return_value=protocol,
                    ):
                        self.assertTrue(
                            _recover_prepared_migration(roots, preparation_request())
                        )

                    self.assertFalse((root / ".prepared-migration.json").exists())
                    self.assertTrue((root / ".prepared-current").is_symlink())
                    for name, logical_root in roots.items():
                        selected = resolve_prepared_artifact_dir(logical_root, name)
                        self.assertEqual(
                            selected.parent.name,
                            "4c03285d-83b3-45d1-bc29-00a516eeef93",
                        )

    def test_corrupt_first_migration_recovery_rolls_back_every_archive_alias_checkpoint(
        self,
    ) -> None:
        phases = [
            checkpoint
            for name in ("client", "evaluation", "public")
            for checkpoint in (
                f"archive:{name}:renamed",
                f"archive:{name}:fsynced",
                f"parent:archive-{name}-fsynced",
            )
        ]
        phases.extend(
            checkpoint
            for name in ("client", "evaluation", "public")
            for checkpoint in (
                f"alias:{name}:created",
                f"parent:alias-{name}-fsynced",
            )
        )

        for phase in phases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                roots = {
                    "client": root / "clients",
                    "public": root / "public",
                    "evaluation": root / "evaluation",
                }
                stage = root / ".prepared-generations" / ".prepare-crash.staging"
                protocol = write_complete_prepared_stage(stage)
                (root / "protocol.json").write_text(
                    json.dumps(protocol), encoding="utf-8"
                )
                for name, logical_root in roots.items():
                    logical_root.mkdir()
                    (logical_root / "legacy.txt").write_text(name, encoding="utf-8")

                crash_prepared_publication(root, phase)
                journal_path = root / ".prepared-migration.json"
                journal_bytes = journal_path.read_bytes()
                journal = json.loads(journal_bytes)
                candidate = root / ".prepared-generations" / journal["generation_id"]
                self.assertTrue(candidate.is_dir())
                self.assertFalse(stage.exists())
                records = candidate / "evaluation" / "test.jsonl"
                original_records = records.read_bytes()
                records.write_bytes(b"corrupt after process death\n")

                for retry in range(2):
                    with (
                        self.subTest(retry=retry),
                        patch(
                            "src.data_prep.load_scientific_protocol",
                            return_value=protocol,
                        ),
                        self.assertRaisesRegex(ValueError, "generation validation"),
                    ):
                        _recover_prepared_migration(roots, preparation_request())

                    self.assertEqual(journal_path.read_bytes(), journal_bytes)
                    self.assertTrue(candidate.is_dir())
                    self.assertEqual(
                        records.read_bytes(), b"corrupt after process death\n"
                    )
                    pointer = root / ".prepared-current"
                    self.assertFalse(pointer.exists() or pointer.is_symlink())
                    for name, logical_root in roots.items():
                        self.assertTrue(logical_root.is_dir())
                        self.assertFalse(logical_root.is_symlink())
                        self.assertEqual(
                            (logical_root / "legacy.txt").read_text(encoding="utf-8"),
                            name,
                        )

                records.write_bytes(original_records)
                with patch(
                    "src.data_prep.load_scientific_protocol", return_value=protocol
                ):
                    self.assertTrue(
                        _recover_prepared_migration(roots, preparation_request())
                    )
                self.assertTrue((root / ".prepared-current").is_symlink())
                self.assertFalse(journal_path.exists())

    def test_corrupt_process_death_recovery_preserves_absent_and_previous_pointers(
        self,
    ) -> None:
        cases = (
            (False, "parent:journal-fsynced", "stage"),
            (False, "generations:fsynced", "final"),
            (True, "parent:journal-fsynced", "stage"),
            (True, "generations:fsynced", "final"),
        )
        for has_previous, phase, candidate_form in cases:
            with (
                self.subTest(
                    has_previous=has_previous,
                    phase=phase,
                    candidate_form=candidate_form,
                ),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                if has_previous:
                    roots, _, protocol = write_pending_prepared_recovery(
                        root, final=False
                    )
                    with patch(
                        "src.data_prep.load_scientific_protocol",
                        return_value=protocol,
                    ):
                        self.assertTrue(
                            _recover_prepared_migration(roots, preparation_request())
                        )
                    pointer = root / ".prepared-current"
                    previous_target = os.readlink(pointer)
                    previous_roots = {
                        name: resolve_prepared_artifact_dir(logical_root, name)
                        for name, logical_root in roots.items()
                    }
                else:
                    roots = {
                        "client": root / "clients",
                        "public": root / "public",
                        "evaluation": root / "evaluation",
                    }
                    for name, logical_root in roots.items():
                        logical_root.mkdir()
                        (logical_root / "legacy.txt").write_text(name, encoding="utf-8")
                    previous_target = None
                    previous_roots = {}

                stage = root / ".prepared-generations" / ".prepare-crash.staging"
                protocol = write_complete_prepared_stage(stage)
                (root / "protocol.json").write_text(
                    json.dumps(protocol), encoding="utf-8"
                )
                crash_prepared_publication(root, phase)

                journal_path = root / ".prepared-migration.json"
                journal_bytes = journal_path.read_bytes()
                journal = json.loads(journal_bytes)
                final = root / ".prepared-generations" / journal["generation_id"]
                candidate = stage if candidate_form == "stage" else final
                self.assertTrue(candidate.is_dir())
                self.assertFalse(
                    (final if candidate_form == "stage" else stage).exists()
                )
                records = candidate / "evaluation" / "test.jsonl"
                original_records = records.read_bytes()
                records.write_bytes(b"corrupt after process death\n")

                for retry in range(2):
                    with (
                        self.subTest(retry=retry),
                        patch(
                            "src.data_prep.load_scientific_protocol",
                            return_value=protocol,
                        ),
                        self.assertRaisesRegex(ValueError, "generation validation"),
                    ):
                        _recover_prepared_migration(roots, preparation_request())

                    self.assertEqual(journal_path.read_bytes(), journal_bytes)
                    self.assertTrue(candidate.is_dir())
                    self.assertEqual(
                        records.read_bytes(), b"corrupt after process death\n"
                    )
                    pointer = root / ".prepared-current"
                    if previous_target is None:
                        self.assertFalse(pointer.exists() or pointer.is_symlink())
                        for name, logical_root in roots.items():
                            self.assertEqual(
                                (logical_root / "legacy.txt").read_text(
                                    encoding="utf-8"
                                ),
                                name,
                            )
                    else:
                        self.assertEqual(os.readlink(pointer), previous_target)
                        for name, logical_root in roots.items():
                            self.assertEqual(
                                resolve_prepared_artifact_dir(logical_root, name),
                                previous_roots[name],
                            )

                records.write_bytes(original_records)
                with patch(
                    "src.data_prep.load_scientific_protocol", return_value=protocol
                ):
                    self.assertTrue(
                        _recover_prepared_migration(roots, preparation_request())
                    )
                self.assertNotEqual(
                    os.readlink(root / ".prepared-current"), previous_target
                )
                self.assertFalse(journal_path.exists())

    def test_corrupt_process_death_recovery_refuses_ambiguous_rollback_ownership(
        self,
    ) -> None:
        cases = (
            ("alias:client:created", "logical-root"),
            ("alias:public:created", "control-residue"),
        )
        for phase, ambiguity in cases:
            with (
                self.subTest(phase=phase, ambiguity=ambiguity),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                roots = {
                    "client": root / "clients",
                    "public": root / "public",
                    "evaluation": root / "evaluation",
                }
                stage = root / ".prepared-generations" / ".prepare-crash.staging"
                protocol = write_complete_prepared_stage(stage)
                (root / "protocol.json").write_text(
                    json.dumps(protocol), encoding="utf-8"
                )
                for name, logical_root in roots.items():
                    logical_root.mkdir()
                    (logical_root / "legacy.txt").write_text(name, encoding="utf-8")
                crash_prepared_publication(root, phase)

                journal_path = root / ".prepared-migration.json"
                journal_bytes = journal_path.read_bytes()
                journal = json.loads(journal_bytes)
                candidate = root / ".prepared-generations" / journal["generation_id"]
                records = candidate / "evaluation" / "test.jsonl"
                records.write_bytes(b"corrupt after process death\n")
                if ambiguity == "logical-root":
                    external = roots["evaluation"]
                    external.mkdir()
                    ambiguous_path = external / "external.txt"
                else:
                    ambiguous_path = (
                        root / f"..prepared-current.{journal['generation_id']}.tmp"
                    )
                ambiguous_path.write_bytes(b"externally owned\n")

                for retry in range(2):
                    with (
                        self.subTest(retry=retry),
                        patch(
                            "src.data_prep.load_scientific_protocol",
                            return_value=protocol,
                        ),
                        self.assertRaisesRegex(
                            ValueError, "rollback ownership is ambiguous"
                        ),
                    ):
                        _recover_prepared_migration(roots, preparation_request())

                    self.assertEqual(ambiguous_path.read_bytes(), b"externally owned\n")
                    self.assertEqual(journal_path.read_bytes(), journal_bytes)
                    self.assertTrue(candidate.is_dir())
                    self.assertEqual(
                        records.read_bytes(), b"corrupt after process death\n"
                    )
                    pointer = root / ".prepared-current"
                    self.assertFalse(pointer.exists() or pointer.is_symlink())
                    for name in ("client", "public"):
                        self.assertEqual(
                            (roots[name] / "legacy.txt").read_text(encoding="utf-8"),
                            name,
                        )
                    if ambiguity == "logical-root":
                        archive = (
                            root
                            / ".prepared-legacy"
                            / journal["generation_id"]
                            / roots["evaluation"].name
                        )
                        self.assertEqual(
                            (archive / "legacy.txt").read_text(encoding="utf-8"),
                            "evaluation",
                        )
                    else:
                        self.assertEqual(
                            (roots["evaluation"] / "legacy.txt").read_text(
                                encoding="utf-8"
                            ),
                            "evaluation",
                        )

    def test_recovery_revalidates_immediately_before_pointer_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots, _, protocol = write_pending_prepared_recovery(root, final=False)
            generation = (
                root / ".prepared-generations" / "4c03285d-83b3-45d1-bc29-00a516eeef93"
            )
            calls = 0
            original_records: bytes | None = None

            def tamper_at_activation(path, transaction):
                """Corrupt evaluation bytes at the final recovery validation.

                Parameters
                ----------
                path : pathlib.Path
                    Final generation path supplied by recovery.
                transaction : mapping
                    Validated migration journal supplied by recovery.

                Returns
                -------
                None
                """
                nonlocal calls, original_records
                calls += 1
                if calls == 4:
                    records = path / "evaluation" / "test.jsonl"
                    original_records = records.read_bytes()
                    records.write_bytes(b"tampered\n")
                _validate_recovery_generation(path, transaction)

            with (
                patch(
                    "src.data_prep.load_scientific_protocol",
                    return_value=protocol,
                ),
                patch(
                    "src.data_prep._validate_recovery_generation",
                    side_effect=tamper_at_activation,
                ),
                self.assertRaisesRegex(ValueError, "generation validation"),
            ):
                _recover_prepared_migration(roots, preparation_request())

            self.assertEqual(calls, 4)
            self.assertIsNotNone(original_records)
            self.assertTrue((root / ".prepared-migration.json").is_file())
            self.assertFalse((root / ".prepared-current").exists())
            for name, logical_root in roots.items():
                self.assertTrue(logical_root.is_dir())
                self.assertFalse(logical_root.is_symlink())
                self.assertEqual(
                    (logical_root / "legacy.txt").read_text(encoding="utf-8"),
                    name,
                )

            (generation / "evaluation" / "test.jsonl").write_bytes(original_records)
            with patch(
                "src.data_prep.load_scientific_protocol",
                return_value=protocol,
            ):
                self.assertTrue(
                    _recover_prepared_migration(roots, preparation_request())
                )

            self.assertTrue((root / ".prepared-current").is_symlink())
            self.assertFalse((root / ".prepared-migration.json").exists())

    def test_recovery_rejects_self_consistent_poisoned_client_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots, candidate, protocol = write_pending_prepared_recovery(
                root, final=False
            )
            shard = candidate / "client" / "client-0"
            records_path = shard / "reviews.jsonl"
            metadata_path = shard / "client_metadata.json"
            rows = [json.loads(line) for line in records_path.read_bytes().splitlines()]
            rows[0]["text"] = "untouched negative"
            poisoned_records = b"".join(
                canonical_client_row_bytes(
                    str(row["row_id"]), str(row["text"]), int(row["label"])
                )
                for row in rows
            )
            records_path.write_bytes(poisoned_records)
            metadata = json.loads(metadata_path.read_bytes())
            metadata["records"]["checksum"] = sha256_bytes(poisoned_records)
            metadata_path.write_bytes(canonical_json_bytes(metadata))

            with (
                patch("src.data_prep.load_scientific_protocol", return_value=protocol),
                self.assertRaisesRegex(ValueError, "generation validation"),
            ):
                _recover_prepared_migration(roots, preparation_request())

            index_path = candidate / "index.json"
            index = json.loads(index_path.read_bytes())
            index["inventory"] = prepared_generation_inventory(candidate)
            index_path.write_bytes(canonical_json_bytes(index))
            with (
                patch("src.data_prep.load_scientific_protocol", return_value=protocol),
                self.assertRaisesRegex(ValueError, "generation validation"),
            ):
                _recover_prepared_migration(roots, preparation_request())

    def test_selected_generation_rejects_post_selection_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots, _, protocol = write_pending_prepared_recovery(root, final=False)
            with patch("src.data_prep.load_scientific_protocol", return_value=protocol):
                self.assertTrue(
                    _recover_prepared_migration(roots, preparation_request())
                )
            selected = root / ".prepared-current" / "client" / "client-0"
            (selected / "reviews.jsonl").write_bytes(b"attacker-controlled\n")

            with self.assertRaisesRegex(ValueError, "bound inventory"):
                resolve_prepared_artifact_dir(roots["client"], "client")

    def test_recovery_accepts_exact_one_and_four_partition_generations(self) -> None:
        for partitions in (1, 4):
            with (
                self.subTest(partitions=partitions),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                roots, _, protocol = write_pending_prepared_recovery(
                    root, final=False, partitions=partitions
                )
                with patch(
                    "src.data_prep.load_scientific_protocol", return_value=protocol
                ):
                    self.assertTrue(
                        _recover_prepared_migration(
                            roots, preparation_request(partitions)
                        )
                    )
                selected = resolve_prepared_artifact_dir(roots["client"], "client")
                self.assertEqual(
                    {path.name for path in selected.iterdir()},
                    {f"client-{index}" for index in range(partitions)},
                )

    def test_frozen_standardizer_matches_producer_and_every_consumer(self) -> None:
        import keras
        import tensorflow as tf

        import dashboard

        cases = [
            "<b>AZ</b>\tÄÖÜ ẞ Σ İ CAFÉ!",
            " A\tB\nC\rD\fE\vF  G\u00a0H\u2003I\u2028J\u2029K ",
        ]
        expected_standardized = [
            " az \tÄÖÜ ẞ Σ İ cafÉ",
            " a\tb\nc\rd\fe\vf  g\u00a0h\u2003i\u2028j\u2029k ",
        ]
        preliminary = create_text_vectorizer(
            sequence_length=500,
            max_tokens=20_000,
        )
        preliminary.adapt(tf.constant(cases))
        vocabulary = preliminary.get_vocabulary()
        vocabulary_bytes = b"".join(term.encode("utf-8") + b"\n" for term in vocabulary)
        protocol = {
            "dataset": {
                "id": "example/imdb",
                "config": "plain_text",
                "revision": "frozen",
                "datasets_version": "1.0.0",
                "splits": {
                    "train": {
                        "rows": 4,
                        "raw_parquet_sha256": "1" * 64,
                        "content_sha256": "2" * 64,
                    }
                },
            },
            "preprocessing": {
                "vocabulary_size": len(vocabulary),
                "max_tokens": len(vocabulary),
                "output_sequence_length": 500,
                "vocabulary_sha256": hashlib.sha256(vocabulary_bytes).hexdigest(),
            },
            "model": {
                "vocabulary_size": len(vocabulary),
                "sequence_length": 500,
                "embedding_dimension": 100,
            },
        }

        with (
            patch(
                "src.data_prep._validated_frameworks",
                return_value=(keras, tf, protocol),
            ),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            root = Path(tmpdir)
            producer = build_vectorizer(cases)
            publish_public_artifacts(producer, root, protocol=protocol)
            manifest = load_app_manifest(public_artifact_dir=root, protocol=protocol)
            records = [
                {"text": text, "label": label} for label in (0, 1) for text in cases
            ]
            client_dir = write_client_artifacts(root / "client-0", manifest, records)

            expected_ids = np.asarray(producer([record["text"] for record in records]))
            train, validation = load_client_shard(
                client_dir, manifest, 0, validation_split=0.5
            )
            consumed_ids = np.concatenate([train[0], validation[0]])
            self.assertEqual(
                Counter(map(tuple, consumed_ids)),
                Counter(map(tuple, expected_ids)),
            )

            dashboard_vectorizer = dashboard.load_vectorizer(manifest)
            np.testing.assert_array_equal(
                dashboard_vectorizer(tf.constant(cases)), producer(tf.constant(cases))
            )

        self.assertEqual(
            [
                value.decode("utf-8")
                for value in protocol_standardize(tf.constant(cases)).numpy()
            ],
            expected_standardized,
        )
        self.assertIs(
            keras.saving.deserialize_keras_object(
                keras.saving.serialize_keras_object(protocol_standardize)
            ),
            protocol_standardize,
        )

    def test_validated_vocabulary_snapshot_survives_same_length_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_public_artifacts(root)
            manifest = load_app_manifest(
                public_artifact_dir=root, protocol=public_protocol()
            )
            (root / "vocab.txt").write_bytes(b"\n[UNK]\nevil\nbad\nmovie\n")
            records = [
                {"text": "good movie", "label": 1},
                {"text": "evil movie", "label": 1},
                {"text": "bad movie", "label": 0},
                {"text": "movie", "label": 0},
            ]
            client_dir = write_client_artifacts(root / "client-0", manifest, records)

            train, validation = load_client_shard(
                client_dir, manifest, 0, validation_split=0.5
            )

        consumed = np.concatenate([train[0], validation[0]])
        self.assertTrue(any(np.array_equal(row[:2], [2, 4]) for row in consumed))
        self.assertTrue(any(np.array_equal(row[:2], [1, 4]) for row in consumed))

    def test_preparation_rejects_hostile_roots_before_loading_or_publication(
        self,
    ) -> None:
        for artifact_name in ("client", "public", "evaluation"):
            with (
                self.subTest(artifact_name=artifact_name),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                external = root / "external"
                external.mkdir()
                client = root / "clients"
                public = root / "public"
                evaluation = root / "evaluation"
                hostile_root = {
                    "client": client,
                    "public": public,
                    "evaluation": evaluation,
                }[artifact_name]
                hostile_root.symlink_to(external, target_is_directory=True)
                with (
                    patch("src.data_prep._validated_frameworks") as frameworks,
                    patch("src.data_prep.load_verified_imdb_dataset") as load_dataset,
                    patch("src.data_prep.publish_public_artifacts") as publish_public,
                    self.assertRaisesRegex(ValueError, "must not be a symlink"),
                ):
                    prepare_all(4, client, public, evaluation)

                frameworks.assert_not_called()
                load_dataset.assert_not_called()
                publish_public.assert_not_called()
                self.assertEqual(list(external.iterdir()), [])

        for artifact_name in ("client", "public", "evaluation"):
            with (
                self.subTest(path_type=artifact_name),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                root = Path(tmpdir)
                outputs = {
                    "client": root / "clients",
                    "public": root / "public",
                    "evaluation": root / "evaluation",
                }
                outputs[artifact_name].write_bytes(b"not a directory")
                with (
                    patch("src.data_prep.load_verified_imdb_dataset") as load_dataset,
                    self.assertRaisesRegex(ValueError, "regular directory"),
                ):
                    prepare_all(
                        4,
                        outputs["client"],
                        outputs["public"],
                        outputs["evaluation"],
                    )
                load_dataset.assert_not_called()

    def test_preflight_rejects_symlinked_ancestor_without_mutation(
        self,
    ) -> None:
        for relative in (False, True):
            for artifact_name, output_name in (
                ("client", "clients"),
                ("public", "public"),
                ("evaluation", "evaluation"),
            ):
                with (
                    self.subTest(artifact_name=artifact_name, relative=relative),
                    tempfile.TemporaryDirectory() as tmpdir,
                ):
                    root = Path(tmpdir)
                    external = root / "external"
                    external.mkdir()
                    marker = external / "must-survive"
                    marker.write_text("external", encoding="utf-8")
                    alias = root / "alias"
                    try:
                        alias.symlink_to(external, target_is_directory=True)
                    except OSError as error:
                        self.skipTest(f"directory symlinks are unavailable: {error}")
                    output = (
                        Path("alias") / output_name if relative else alias / output_name
                    )

                    with (
                        chdir(root),
                        self.assertRaisesRegex(ValueError, "path component"),
                    ):
                        _preflight_output_root(
                            output,
                            artifact_name,
                            reusable=True,
                            allow_prepared_alias=True,
                        )

                    self.assertEqual(marker.read_text(encoding="utf-8"), "external")
                    self.assertFalse((external / output_name).exists())

    def test_preparation_rejects_reserved_root_names_before_mutation(self) -> None:
        reserved_names = (
            ".prepared-current",
            ".prepared-generations",
            ".prepared-legacy",
            ".prepared-migration.json",
            ".fml-prepare.lock",
            ".prepared-attacker-residue",
            ".prepare-attacker.staging",
            "..prepared-current.11111111-1111-4111-8111-111111111111.tmp",
            "..prepared-current.rollback.tmp",
        )
        for relative in (False, True):
            for artifact_name in ("client", "public", "evaluation"):
                for reserved_name in reserved_names:
                    with (
                        self.subTest(
                            relative=relative,
                            artifact_name=artifact_name,
                            reserved_name=reserved_name,
                        ),
                        tempfile.TemporaryDirectory() as tmpdir,
                    ):
                        root = Path(tmpdir)
                        names = {
                            "client": "clients",
                            "public": "public",
                            "evaluation": "evaluation",
                        }
                        names[artifact_name] = reserved_name
                        outputs = {
                            name: Path(value) if relative else root / value
                            for name, value in names.items()
                        }
                        with (
                            chdir(root),
                            patch(
                                "src.data_prep._acquire_preparation_lock"
                            ) as acquire_lock,
                            patch("src.data_prep._validated_frameworks") as frameworks,
                            patch("src.data_prep.validate_protocol_runtime") as runtime,
                            patch(
                                "src.data_prep.load_verified_imdb_dataset"
                            ) as load_dataset,
                            self.assertRaisesRegex(
                                ValueError, "reserved preparation name"
                            ),
                        ):
                            prepare_all(
                                4,
                                outputs["client"],
                                outputs["public"],
                                outputs["evaluation"],
                            )

                        acquire_lock.assert_not_called()
                        frameworks.assert_not_called()
                        runtime.assert_not_called()
                        load_dataset.assert_not_called()
                        self.assertEqual(list(root.iterdir()), [])

    def test_preflight_accepts_safe_hidden_root_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in (
                ".clients",
                ".cache",
                ".preparedness",
                ".preparation-staging",
                "..prepared-current",
            ):
                with self.subTest(name=name):
                    self.assertEqual(
                        _preflight_output_root(
                            root / name,
                            "client",
                            reusable=True,
                            allow_prepared_alias=True,
                        ),
                        root / name,
                    )

            self.assertEqual(list(root.iterdir()), [])

    def test_preflight_preserves_relative_lexical_prepared_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            try:
                (root / "clients").symlink_to(
                    Path(".prepared-current") / "client",
                    target_is_directory=True,
                )
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            with chdir(root):
                output = _preflight_output_root(
                    "nested/../clients",
                    "client",
                    reusable=True,
                    allow_prepared_alias=True,
                )

            self.assertEqual(output, root / "clients")

    def test_preparation_rejects_relative_symlinked_ancestor_before_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            launch_dir = root / "launch"
            external = root / "external"
            generations = external / "nested" / ".prepared-generations"
            residue = generations / ".prepare-attacker.staging"
            launch_dir.mkdir()
            residue.mkdir(parents=True)
            (external / "marker.bin").write_bytes(b"external\x00content")
            (residue / "preserve.bin").write_bytes(b"prepared residue")
            try:
                (launch_dir / "alias").symlink_to(external, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")
            before = {
                path.relative_to(external): (
                    path.lstat().st_mode,
                    path.read_bytes() if path.is_file() else None,
                )
                for path in (external, *external.rglob("*"))
            }

            with (
                chdir(launch_dir),
                patch("src.data_prep._acquire_preparation_lock") as acquire_lock,
                self.assertRaisesRegex(ValueError, "path component"),
            ):
                prepare_all(
                    4,
                    "alias/nested/clients",
                    "alias/nested/public",
                    "alias/nested/evaluation",
                )

            acquire_lock.assert_not_called()
            after = {
                path.relative_to(external): (
                    path.lstat().st_mode,
                    path.read_bytes() if path.is_file() else None,
                )
                for path in (external, *external.rglob("*"))
            }
            self.assertEqual(after, before)

    def test_public_publisher_rejects_symlink_without_external_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            external = root / "external"
            external.mkdir()
            public = root / "public"
            public.symlink_to(external, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                publish_public_artifacts(object(), public, protocol={})

            self.assertEqual(list(external.iterdir()), [])

    def test_preparation_publishes_evaluation_only_after_retryable_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evaluation_dir = root / "evaluation"
            test_rows = [
                {"text": "untouched negative", "label": 0},
                {"text": "untouched positive", "label": 1},
            ]
            dataset = {
                "train": {
                    "text": ["negative", "positive", "bad", "good"],
                    "label": [0, 1, 0, 1],
                },
                "test": test_rows,
            }
            vocabulary = ["", "[UNK]", "negative", "positive", "good"]
            vocabulary_bytes = b"".join(
                term.encode("utf-8") + b"\n" for term in vocabulary
            )
            test_content = b"".join(
                canonical_source_row_bytes(row["text"], row["label"])
                for row in test_rows
            )
            train_content = b"".join(
                canonical_source_row_bytes(text, label)
                for text, label in zip(
                    dataset["train"]["text"],
                    dataset["train"]["label"],
                    strict=True,
                )
            )
            protocol = {
                "dataset": {
                    "id": "example/imdb",
                    "config": "plain_text",
                    "revision": "frozen",
                    "datasets_version": "1.0.0",
                    "splits": {
                        "train": {
                            "rows": 4,
                            "label_counts": [2, 2],
                            "raw_parquet_sha256": "1" * 64,
                            "content_sha256": hashlib.sha256(train_content).hexdigest(),
                        },
                        "test": {
                            "rows": 2,
                            "label_counts": [1, 1],
                            "raw_parquet_sha256": "3" * 64,
                            "content_sha256": hashlib.sha256(test_content).hexdigest(),
                        },
                    },
                },
                "preprocessing": {
                    "vocabulary_size": len(vocabulary),
                    "max_tokens": len(vocabulary),
                    "output_sequence_length": 4,
                    "vocabulary_sha256": hashlib.sha256(vocabulary_bytes).hexdigest(),
                },
                "model": {
                    "vocabulary_size": len(vocabulary),
                    "sequence_length": 4,
                    "embedding_dimension": 8,
                },
                "framework": {
                    "tensorflow_version": "2.20.0",
                    "keras_version": "3.14.0",
                    "numpy_version": "2.4.4",
                },
            }

            class Vectorizer:
                def get_vocabulary(self):
                    return vocabulary

            publication_attempts = 0

            def publish_once_then_succeed(roots, stages, request):
                nonlocal publication_attempts
                publication_attempts += 1
                if publication_attempts == 1:
                    original_replace = os.replace

                    def fail_generation_switch(source, destination):
                        if Path(destination).name == ".prepared-current":
                            raise RuntimeError("injected generation-switch failure")
                        return original_replace(source, destination)

                    with patch(
                        "src.data_prep.os.replace",
                        side_effect=fail_generation_switch,
                    ):
                        return _publish_prepared_roots(roots, stages, request)
                return _publish_prepared_roots(roots, stages, request)

            with (
                patch("src.data_prep.load_verified_imdb_dataset", return_value=dataset),
                patch("src.data_prep.build_vectorizer", return_value=Vectorizer()),
                patch("src.data_prep.load_scientific_protocol", return_value=protocol),
                patch(
                    "src.data_prep._publish_prepared_roots",
                    side_effect=publish_once_then_succeed,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "generation-switch"):
                    prepare_all(
                        2,
                        root / "clients",
                        root / "public",
                        evaluation_dir,
                    )
                self.assertFalse(evaluation_dir.exists())
                stale_generations = list(
                    (root / ".prepared-generations").glob("*-*-*-*-*")
                )
                self.assertEqual(len(stale_generations), 1)

                prepare_all(
                    2,
                    root / "clients",
                    root / "public",
                    evaluation_dir,
                )

                selected_before_retry = resolve_prepared_artifact_dir(
                    root / "public", "public"
                ).parent
                prepare_all(
                    2,
                    root / "clients",
                    root / "public",
                    evaluation_dir,
                )

            self.assertTrue(
                resolve_prepared_artifact_dir(evaluation_dir, "evaluation").is_dir()
            )
            self.assertEqual(publication_attempts, 3)
            generations = list((root / ".prepared-generations").glob("*-*-*-*-*"))
            self.assertEqual(len(generations), 3)
            selected = resolve_prepared_artifact_dir(root / "public", "public").parent
            self.assertIn(selected, generations)
            self.assertNotEqual(selected, stale_generations[0])
            self.assertNotEqual(selected, selected_before_retry)
            self.assertTrue((root / ".prepared-current").is_symlink())
            self.assertTrue((root / "public").is_symlink())
            self.assertTrue((root / "clients").is_symlink())
            self.assertTrue(evaluation_dir.is_symlink())

    def test_publication_migrates_legacy_roots_and_deployment_selects_new_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots = {
                "client": root / "clients",
                "public": root / "public",
                "evaluation": root / "evaluation",
            }
            for kind, logical_root in roots.items():
                logical_root.mkdir()
                (logical_root / "selected.txt").write_text(
                    f"legacy-{kind}", encoding="utf-8"
                )
            (roots["client"] / "client-0").mkdir()
            (roots["client"] / "client-0" / "shard.txt").write_text(
                "legacy-client-0", encoding="utf-8"
            )
            legacy_client_descriptor = os.open(
                roots["client"] / "client-0", os.O_RDONLY | os.O_DIRECTORY
            )
            legacy_public_descriptor = os.open(
                roots["public"], os.O_RDONLY | os.O_DIRECTORY
            )

            generations = root / ".prepared-generations"
            generation_stage = generations / ".prepare-integration.staging"
            stages = {kind: generation_stage / kind for kind in roots}
            protocol = write_complete_prepared_stage(generation_stage)
            try:
                with patch(
                    "src.data_prep.load_scientific_protocol",
                    return_value=protocol,
                ):
                    _publish_prepared_roots(roots, stages, preparation_request())

                selected_public = resolve_prepared_artifact_dir(
                    roots["public"], "public"
                )
                selected_client = resolve_prepared_artifact_dir(
                    roots["client"], "client"
                )
                self.assertEqual(
                    {path.name for path in selected_public.iterdir()},
                    {"manifest.json", "vocab.txt"},
                )
                self.assertEqual(
                    {path.name for path in selected_client.iterdir()}, {"client-0"}
                )

                compose = Path("compose.yaml").read_text(encoding="utf-8")
                public_source = next(
                    line.split("source: ", maxsplit=1)[1].strip()
                    for line in compose.splitlines()
                    if "source: ./artifacts/.prepared-current/public" in line
                )
                client_source = next(
                    line.split("source: ", maxsplit=1)[1].strip()
                    for line in compose.splitlines()
                    if "source: ./artifacts/.prepared-current/client/client-0" in line
                )
                deployment_public = root / Path(public_source).relative_to("artifacts")
                deployment_client = root / Path(client_source).relative_to("artifacts")
                self.assertEqual(deployment_public.resolve(), selected_public)
                self.assertEqual(
                    (deployment_client / "reviews.jsonl").read_bytes(),
                    (selected_client / "client-0" / "reviews.jsonl").read_bytes(),
                )

                archive = next((root / ".prepared-legacy").iterdir())
                self.assertEqual(
                    (archive / "public" / "selected.txt").read_text(encoding="utf-8"),
                    "legacy-public",
                )
                self.assertEqual(
                    (archive / "clients" / "client-0" / "shard.txt").read_text(
                        encoding="utf-8"
                    ),
                    "legacy-client-0",
                )
                public_file = os.open(
                    "selected.txt", os.O_RDONLY, dir_fd=legacy_public_descriptor
                )
                client_file = os.open(
                    "shard.txt", os.O_RDONLY, dir_fd=legacy_client_descriptor
                )
                try:
                    self.assertEqual(os.read(public_file, 100), b"legacy-public")
                    self.assertEqual(os.read(client_file, 100), b"legacy-client-0")
                finally:
                    os.close(public_file)
                    os.close(client_file)
            finally:
                os.close(legacy_public_descriptor)
                os.close(legacy_client_descriptor)

    def test_preparation_rejects_unsupported_platform_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch("src.artifact_compatibility.sys.platform", "win32"),
                patch("src.data_prep.load_verified_imdb_dataset") as load_dataset,
                self.assertRaisesRegex(RuntimeError, "require Linux"),
            ):
                prepare_all(
                    2,
                    root / "clients",
                    root / "public",
                    root / "evaluation",
                )

            load_dataset.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_legacy_migration_rolls_back_all_roots_when_activation_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots = {
                "client": root / "clients",
                "public": root / "public",
                "evaluation": root / "evaluation",
            }
            generations = root / ".prepared-generations"
            stage_root = generations / ".prepare-rollback.staging"
            stages = {kind: stage_root / kind for kind in roots}
            protocol = write_complete_prepared_stage(stage_root)
            for kind, logical_root in roots.items():
                logical_root.mkdir()
                (logical_root / "legacy.txt").write_text(kind, encoding="utf-8")

            original_replace = os.replace

            def reject_pointer(source: str | Path, destination: str | Path) -> None:
                if Path(destination).name == ".prepared-current":
                    raise OSError("injected pointer failure")
                original_replace(source, destination)

            with (
                patch(
                    "src.data_prep.load_scientific_protocol",
                    return_value=protocol,
                ),
                patch("src.data_prep.os.replace", side_effect=reject_pointer),
                self.assertRaisesRegex(OSError, "pointer failure"),
            ):
                _publish_prepared_roots(roots, stages, preparation_request())

            for kind, logical_root in roots.items():
                with self.subTest(kind=kind):
                    self.assertTrue(logical_root.is_dir())
                    self.assertFalse(logical_root.is_symlink())
                    self.assertEqual(
                        (logical_root / "legacy.txt").read_text(encoding="utf-8"),
                        kind,
                    )
            self.assertFalse((root / ".prepared-current").exists())
            self.assertEqual(list((root / ".prepared-legacy").iterdir()), [])

    def test_legacy_migration_recovers_after_process_death_at_every_boundary(
        self,
    ) -> None:
        phases = [
            "journal:published",
            "parent:journal-fsynced",
            "generation:renamed",
            "generations:fsynced",
            "legacy-root:created",
            "parent:legacy-root-fsynced",
            "legacy-archive:created",
            "legacy-root:archive-fsynced",
        ]
        for name in ("client", "evaluation", "public"):
            phases.extend(
                (
                    f"archive:{name}:renamed",
                    f"archive:{name}:fsynced",
                    f"parent:archive-{name}-fsynced",
                )
            )
        for name in ("client", "evaluation", "public"):
            phases.extend((f"alias:{name}:created", f"parent:alias-{name}-fsynced"))
        phases.extend(
            (
                "pointer-temporary:created",
                "parent:pointer-temporary-fsynced",
                "pointer:replaced",
                "parent:pointer-fsynced",
                "journal:removed",
                "parent:journal-removal-fsynced",
            )
        )
        child_code = textwrap.dedent(
            """
            import os
            import json
            import sys
            from pathlib import Path

            import src.data_prep as data_prep

            root = Path(sys.argv[1])
            phase = sys.argv[2]
            roots = {
                "client": root / "clients",
                "public": root / "public",
                "evaluation": root / "evaluation",
            }
            stage = root / ".prepared-generations" / ".prepare-crash.staging"
            stages = {name: stage / name for name in roots}
            protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
            data_prep.load_scientific_protocol = lambda: protocol

            def crash_at(value):
                if value == phase:
                    os._exit(91)

            data_prep._publication_checkpoint = crash_at
            data_prep._publish_prepared_roots(roots, stages, {"partitions": 1})
            """
        )

        for phase in phases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                roots = {
                    "client": root / "clients",
                    "public": root / "public",
                    "evaluation": root / "evaluation",
                }
                stage = root / ".prepared-generations" / ".prepare-crash.staging"
                protocol = write_complete_prepared_stage(stage)
                (root / "protocol.json").write_text(
                    json.dumps(protocol), encoding="utf-8"
                )
                for name, logical_root in roots.items():
                    logical_root.mkdir()
                    (logical_root / "legacy.txt").write_text(name, encoding="utf-8")

                completed = subprocess.run(
                    [sys.executable, "-c", child_code, str(root), phase],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )

                self.assertEqual(completed.returncode, 91, completed.stderr)
                with patch(
                    "src.data_prep.load_scientific_protocol",
                    return_value=protocol,
                ):
                    _recover_prepared_migration(roots, preparation_request())
                for name, logical_root in roots.items():
                    selected = resolve_prepared_artifact_dir(logical_root, name)
                    if name == "evaluation":
                        self.assertTrue((selected / "manifest.json").is_file())
                    elif name == "public":
                        self.assertEqual(
                            {path.name for path in selected.iterdir()},
                            {"manifest.json", "vocab.txt"},
                        )
                    else:
                        self.assertEqual(
                            {path.name for path in selected.iterdir()}, {"client-0"}
                        )
                archive = next((root / ".prepared-legacy").iterdir())
                for name, logical_root in roots.items():
                    self.assertEqual(
                        (archive / logical_root.name / "legacy.txt").read_text(
                            encoding="utf-8"
                        ),
                        name,
                    )

    def test_prepare_retry_recovers_before_dataset_or_framework_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots = {
                "client": root / "clients",
                "public": root / "public",
                "evaluation": root / "evaluation",
            }
            stage = root / ".prepared-generations" / ".prepare-retry.staging"
            protocol = write_complete_prepared_stage(stage)
            (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            for name, logical_root in roots.items():
                logical_root.mkdir()
                (logical_root / "legacy.txt").write_text(name, encoding="utf-8")
            child_code = textwrap.dedent(
                """
                import json
                import os
                import sys
                from pathlib import Path

                import src.data_prep as data_prep

                root = Path(sys.argv[1])
                roots = {
                    "client": root / "clients",
                    "public": root / "public",
                    "evaluation": root / "evaluation",
                }
                stage = root / ".prepared-generations" / ".prepare-retry.staging"
                stages = {name: stage / name for name in roots}
                protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
                data_prep.load_scientific_protocol = lambda: protocol

                def crash_at(value):
                    if value == "archive:client:renamed":
                        os._exit(92)

                data_prep._publication_checkpoint = crash_at
                data_prep._publish_prepared_roots(roots, stages, {"partitions": 1})
                """
            )
            completed = subprocess.run(
                [sys.executable, "-c", child_code, str(root)],
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            self.assertEqual(completed.returncode, 92, completed.stderr)

            with (
                patch(
                    "src.data_prep.load_scientific_protocol",
                    return_value=protocol,
                ),
                patch("src.data_prep._validated_frameworks") as frameworks,
                patch("src.data_prep.load_verified_imdb_dataset") as load_dataset,
            ):
                prepare_all(1, roots["client"], roots["public"], roots["evaluation"])

            frameworks.assert_not_called()
            load_dataset.assert_not_called()
            self.assertEqual(
                {
                    path.name
                    for path in resolve_prepared_artifact_dir(
                        roots["public"], "public"
                    ).iterdir()
                },
                {"manifest.json", "vocab.txt"},
            )

    def test_prepare_retry_discards_mismatched_request_and_regenerates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots, stale_candidate, protocol = write_pending_prepared_recovery(
                root, final=False
            )
            dataset = {
                "train": {
                    "text": ["bad", "good", "bad movie", "good movie"],
                    "label": [0, 1, 0, 1],
                },
                "test": [
                    {"text": "untouched negative", "label": 0},
                    {"text": "untouched positive", "label": 1},
                ],
            }

            class Vectorizer:
                def get_vocabulary(self):
                    return ["", "[UNK]", "good", "bad", "movie"]

            with (
                patch("src.data_prep._validated_frameworks") as frameworks,
                patch(
                    "src.data_prep.load_verified_imdb_dataset",
                    return_value=dataset,
                ) as load_dataset,
                patch("src.data_prep.build_vectorizer", return_value=Vectorizer()),
                patch("src.data_prep.load_scientific_protocol", return_value=protocol),
                patch(
                    "src.data_prep.dirichlet_split",
                    return_value={index: [index] for index in range(4)},
                ),
            ):
                prepare_all(4, roots["client"], roots["public"], roots["evaluation"])

            frameworks.assert_called_once_with()
            load_dataset.assert_called_once_with()
            selected_client = resolve_prepared_artifact_dir(roots["client"], "client")
            self.assertEqual(
                {path.name for path in selected_client.iterdir()},
                {f"client-{index}" for index in range(4)},
            )
            self.assertFalse(stale_candidate.exists())
            self.assertFalse((root / ".prepared-migration.json").exists())
            archive = next((root / ".prepared-legacy").iterdir())
            for name, logical_root in roots.items():
                self.assertEqual(
                    (archive / logical_root.name / "legacy.txt").read_text(
                        encoding="utf-8"
                    ),
                    name,
                )

    def test_schema_one_pending_preparation_is_preserved_unrecovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots, candidate, _ = write_pending_prepared_recovery(
                root, final=False, legacy_schema=True
            )

            with self.assertRaisesRegex(ValueError, "lacks durable stage identity"):
                _recover_prepared_migration(roots, preparation_request(4))

            self.assertTrue(candidate.exists())
            self.assertTrue((root / ".prepared-migration.json").exists())
            for name, logical_root in roots.items():
                self.assertTrue(logical_root.is_dir())
                self.assertFalse(logical_root.is_symlink())
                self.assertEqual(
                    (logical_root / "legacy.txt").read_text(encoding="utf-8"), name
                )

    def test_schema_two_pending_preparation_is_preserved_unrecovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots, candidate, _ = write_pending_prepared_recovery(root, final=False)
            index_path = candidate / "index.json"
            index = json.loads(index_path.read_bytes())
            index["schema_version"] = 2
            index.pop("inventory")
            index_path.write_bytes(canonical_json_bytes(index))
            journal_path = root / ".prepared-migration.json"
            journal = json.loads(journal_path.read_bytes())
            journal["schema_version"] = 2
            journal.pop("generation_index_checksum")
            for field in (
                "parent_device",
                "parent_inode",
                "generations_device",
                "generations_inode",
                "stage_device",
                "stage_inode",
            ):
                journal.pop(field)
            journal_path.write_bytes(canonical_json_bytes(journal))

            with self.assertRaisesRegex(ValueError, "lacks durable stage identity"):
                _recover_prepared_migration(roots, preparation_request())

            self.assertTrue(candidate.exists())
            self.assertTrue(journal_path.exists())

    def test_preparation_rejects_overlapping_artifact_boundaries_before_loading(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch("src.data_prep.load_verified_imdb_dataset") as load_dataset,
                self.assertRaisesRegex(ValueError, "artifact roots must be separate"),
            ):
                prepare_all(
                    4,
                    root / "clients",
                    root / "public",
                    root / "public" / "evaluation",
                )

            load_dataset.assert_not_called()

    def test_artifact_flow(self) -> None:
        checks = (
            check_public_manifest_loads_the_model_shape,
            check_raw_client_packaging_keeps_every_sample_private,
            check_client_tokenizes_its_raw_reviews_with_public_vocabulary,
            check_client_loader_preserves_unicode_line_separator_in_review,
            check_client_loader_preserves_control_character_in_vocabulary,
            check_cli_prepares_all_artifacts_with_one_dataset_load,
        )
        for check in checks:
            with (
                self.subTest(check=check.__name__),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                check(Path(tmpdir))

    def test_client_shard_loader_checks_each_schema_version_state(self) -> None:
        records = [
            {"text": "good movie", "label": 1},
            {"text": "bad movie", "label": 0},
            {"text": "good", "label": 1},
            {"text": "bad", "label": 0},
        ]
        for version, error in (
            (None, "no valid"),
            (1, "older"),
            (CLIENT_SHARD_SCHEMA_VERSION + 1, "newer"),
        ):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                public_dir = root / "public"
                public_dir.mkdir()
                manifest = write_public_artifacts(public_dir)
                path = root / "client-0"
                path.mkdir()
                metadata = {} if version is None else {"schema_version": version}
                (path / "client_metadata.json").write_bytes(
                    canonical_json_bytes(metadata)
                )
                (path / "reviews.jsonl").write_text(
                    "\n".join(json.dumps(record) for record in records),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, error):
                    load_client_shard(path, manifest, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            public_dir = root / "public"
            public_dir.mkdir()
            manifest = write_public_artifacts(public_dir)
            path = write_client_artifacts(root / "client-0", manifest, records)
            train, validation = load_client_shard(path, manifest, 0)
            self.assertEqual(len(train[1]) + len(validation[1]), len(records))

    def test_client_shard_metadata_binds_records_and_public_manifest(self) -> None:
        manifest_bytes = b"canonical public manifest\n"
        records_bytes = b"canonical records\n"
        metadata = client_shard_metadata(
            3,
            [0, 1],
            records_bytes=records_bytes,
            public_manifest_bytes=manifest_bytes,
            dataset={"split": "train"},
        )

        self.assertEqual(metadata["client_id"], 3)
        self.assertEqual(metadata["schema_version"], CLIENT_SHARD_SCHEMA_VERSION)
        self.assertEqual(
            metadata["records"]["checksum"],
            "sha256:" + hashlib.sha256(records_bytes).hexdigest(),
        )
        self.assertEqual(
            metadata["public_manifest"]["checksum"],
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
        )

    def test_client_shard_rejects_identity_content_and_binding_corruption(self) -> None:
        cases = (
            "wrong client",
            "stale dataset",
            "test identity",
            "duplicate identity",
            "non-binary label",
            "noncanonical record",
            "sample count",
            "label histogram",
            "record checksum",
            "public manifest",
        )
        records = [
            {"text": "good movie", "label": 1},
            {"text": "bad movie", "label": 0},
            {"text": "good", "label": 1},
            {"text": "bad", "label": 0},
        ]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                public_dir = root / "public"
                public_dir.mkdir()
                manifest = write_public_artifacts(public_dir)
                shard = write_client_artifacts(root / "client-0", manifest, records)
                metadata_path = shard / "client_metadata.json"
                records_path = shard / "reviews.jsonl"
                metadata = json.loads(metadata_path.read_bytes())
                decoded_records = [
                    json.loads(line) for line in records_path.read_bytes().splitlines()
                ]

                if case == "wrong client":
                    metadata["client_id"] = 9
                elif case == "stale dataset":
                    metadata["dataset"]["revision"] = "stale"
                elif case == "test identity":
                    decoded_records[0]["row_id"] = "test:0"
                elif case == "duplicate identity":
                    decoded_records[1]["row_id"] = "train:0"
                elif case == "non-binary label":
                    decoded_records[0]["label"] = 2
                elif case == "noncanonical record":
                    content = b"".join(
                        (json.dumps(record, ensure_ascii=False) + "\n").encode()
                        for record in decoded_records
                    )
                    records_path.write_bytes(content)
                    metadata["records"]["checksum"] = sha256_bytes(content)
                elif case == "sample count":
                    metadata["sample_count"] += 1
                elif case == "label histogram":
                    metadata["label_histogram"]["0"] += 1
                elif case == "record checksum":
                    metadata["records"]["checksum"] = "sha256:" + "0" * 64
                elif case == "public manifest":
                    metadata["public_manifest"]["checksum"] = "sha256:" + "0" * 64

                if case in {"test identity", "duplicate identity", "non-binary label"}:
                    content = b"".join(
                        canonical_client_row_bytes(
                            str(record["row_id"]),
                            str(record["text"]),
                            int(record["label"]),
                        )
                        for record in decoded_records
                    )
                    records_path.write_bytes(content)
                    metadata["records"]["checksum"] = sha256_bytes(content)
                metadata_path.write_bytes(canonical_json_bytes(metadata))

                with self.assertRaises(ValueError):
                    load_client_shard_snapshot(shard, manifest, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            public_dir = root / "public"
            public_dir.mkdir()
            manifest = write_public_artifacts(public_dir)
            shard = write_client_artifacts(root / "client-0", manifest, records)
            with self.assertRaisesRegex(ValueError, "expected client ID"):
                load_client_shard_snapshot(shard, manifest, 1)

    def test_publishers_reject_unsafe_existing_child_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            external = root / "external.txt"
            external.write_text("outside", encoding="utf-8")
            public = root / "public"
            public.mkdir()
            (public / "vocab.txt").symlink_to(external)
            protocol = public_protocol()

            class Vectorizer:
                def get_vocabulary(self):
                    return ["", "[UNK]", "good", "bad", "movie"]

            with self.assertRaisesRegex(ValueError, "unsafe"):
                publish_public_artifacts(Vectorizer(), public, protocol=protocol)
            self.assertEqual(external.read_text(encoding="utf-8"), "outside")

        if os.name != "nt":
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                external = root / "external.txt"
                external.write_text("outside", encoding="utf-8")
                public = root / "public"
                public.mkdir()
                os.link(external, public / "vocab.txt")

                class Vectorizer:
                    def get_vocabulary(self):
                        return ["", "[UNK]", "good", "bad", "movie"]

                with self.assertRaisesRegex(ValueError, "unsafe"):
                    publish_public_artifacts(
                        Vectorizer(), public, protocol=public_protocol()
                    )
                self.assertEqual(external.read_text(encoding="utf-8"), "outside")

    def test_public_manifest_rejects_unexpected_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            public = Path(tmpdir)
            write_public_artifacts(public)
            (public / "legacy-test-export.txt").write_text(
                "external test sentinel", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "unexpected files"):
                load_app_manifest(
                    public_artifact_dir=public, protocol=public_protocol()
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            public = Path(tmpdir)
            (public / "legacy-test-export.txt").write_text(
                "external test sentinel", encoding="utf-8"
            )

            class Vectorizer:
                def get_vocabulary(self):
                    return ["", "[UNK]", "good", "bad", "movie"]

            with self.assertRaisesRegex(ValueError, "unexpected files"):
                publish_public_artifacts(
                    Vectorizer(), public, protocol=public_protocol()
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clients = root / "clients"
            client = clients / "client-0"
            client.mkdir(parents=True)
            external = root / "outside.jsonl"
            external.write_text("outside", encoding="utf-8")
            (client / "reviews.jsonl").symlink_to(external)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                package_raw_client_shards(
                    ["good", "bad"],
                    [1, 0],
                    clients,
                    1,
                    manifest={"dataset": {"split": "train"}},
                    dataset={"split": "train"},
                )
            self.assertEqual(external.read_text(encoding="utf-8"), "outside")

    def test_client_packaging_removes_stale_partition_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            public_dir = root / "public"
            public_dir.mkdir()
            manifest = write_public_artifacts(public_dir)
            clients = root / "clients"
            texts = ["bad", "good", "negative", "positive"]
            labels = [0, 1, 0, 1]
            package_raw_client_shards(
                texts,
                labels,
                clients,
                4,
                manifest=manifest.payload,
                dataset=manifest.payload["dataset"],
            )
            package_raw_client_shards(
                texts,
                labels,
                clients,
                2,
                manifest=manifest.payload,
                dataset=manifest.payload["dataset"],
            )

            self.assertEqual(
                {path.name for path in clients.glob("client-*")},
                {"client-0", "client-1"},
            )

    def test_preparation_lock_rejects_concurrent_writer_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            roots = {
                "client": root / "clients",
                "public": root / "public",
                "evaluation": root / "evaluation",
            }
            lock = _acquire_preparation_lock(roots)
            try:
                alternate_roots = {
                    "client": root / "other-clients",
                    "public": root / "other-public",
                    "evaluation": root / "other-evaluation",
                }
                with self.assertRaisesRegex(RuntimeError, "already in progress"):
                    _acquire_preparation_lock(alternate_roots)
                with (
                    patch("src.data_prep.load_verified_imdb_dataset") as load_dataset,
                    self.assertRaisesRegex(RuntimeError, "already in progress"),
                ):
                    prepare_all(
                        2,
                        roots["client"],
                        roots["public"],
                        roots["evaluation"],
                    )
                load_dataset.assert_not_called()

                independent_root = root / "independent"
                independent_roots = {
                    "client": independent_root / "clients",
                    "public": independent_root / "public",
                    "evaluation": independent_root / "evaluation",
                }
                independent_root.mkdir()
                independent_lock = _acquire_preparation_lock(independent_roots)
                try:
                    self.assertNotEqual(lock.path, independent_lock.path)
                finally:
                    independent_lock.release()
            finally:
                lock.release()


if __name__ == "__main__":
    unittest.main()
