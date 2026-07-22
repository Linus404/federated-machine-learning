import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.app_manifest import AppManifest, load_app_manifest
from src.artifact_compatibility import (
    CLIENT_SHARD_SCHEMA_VERSION,
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_bytes,
)
from src.contracts import canonical_client_row_bytes, client_shard_metadata
from src.data_prep import (
    _acquire_preparation_lock,
    _publish_prepared_roots,
    _recover_prepared_migration,
    build_vectorizer,
    main,
    package_raw_client_shards,
    prepare_all,
    publish_public_artifacts,
)
from src.evaluation_artifact import canonical_source_row_bytes
from src.local_training import (
    build_model_from_manifest,
    load_client_shard,
    load_client_shard_snapshot,
)
from src.paths import resolve_prepared_artifact_dir
from src.text_preprocessing import create_text_vectorizer, protocol_standardize


def write_public_artifacts(path: Path, sequence_length: int = 4) -> AppManifest:
    vocabulary = b"\n[UNK]\ngood\nbad\nmovie\n"
    vocabulary_sha256 = hashlib.sha256(vocabulary).hexdigest()
    (path / "vocab.txt").write_bytes(vocabulary)
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
            "content_sha256": "2" * 64,
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


def public_protocol(sequence_length: int = 4) -> dict[str, object]:
    """Return the small frozen public-artifact protocol used by flow tests.

    Returns
    -------
    dict of str to object
        Dataset and vocabulary identity matching ``write_public_artifacts``.
    """
    vocabulary = b"\n[UNK]\ngood\nbad\nmovie\n"
    return {
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


def write_client_artifacts(
    path: Path,
    manifest: AppManifest,
    records: list[dict[str, object]],
    *,
    client_id: int = 0,
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

    Returns
    -------
    pathlib.Path
        Written client shard directory.
    """
    path.mkdir()
    records_bytes = b"".join(
        canonical_client_row_bytes(
            f"train:{index}", str(record["text"]), int(record["label"])
        )
        for index, record in enumerate(records)
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
    (public_dir / "keep.txt").write_text("keep", encoding="utf-8")
    (client_dir / "keep.txt").write_text("keep", encoding="utf-8")
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
    protocol = {
        "dataset": {
            "id": "stanfordnlp/imdb",
            "config": "plain_text",
            "revision": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
            "datasets_version": "4.8.5",
            "splits": {
                "train": {
                    "rows": 12,
                    "raw_parquet_sha256": "1" * 64,
                    "content_sha256": "2" * 64,
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
    assert (public_generation / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (client_generation / "keep.txt").read_text(encoding="utf-8") == "keep"


class ArtifactFlowTests(unittest.TestCase):
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

            def publish_once_then_succeed(roots, stages):
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
                        return _publish_prepared_roots(roots, stages)
                return _publish_prepared_roots(roots, stages)

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
            for kind, stage in stages.items():
                stage.mkdir(parents=True, exist_ok=True)
                (stage / "selected.txt").write_text(
                    f"generation-{kind}", encoding="utf-8"
                )
            (stages["client"] / "client-0").mkdir()
            (stages["client"] / "client-0" / "shard.txt").write_text(
                "generation-client-0", encoding="utf-8"
            )

            try:
                _publish_prepared_roots(roots, stages)

                selected_public = resolve_prepared_artifact_dir(
                    roots["public"], "public"
                )
                selected_client = resolve_prepared_artifact_dir(
                    roots["client"], "client"
                )
                self.assertEqual(
                    (selected_public / "selected.txt").read_text(encoding="utf-8"),
                    "generation-public",
                )
                self.assertEqual(
                    (selected_client / "client-0" / "shard.txt").read_text(
                        encoding="utf-8"
                    ),
                    "generation-client-0",
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
                self.assertEqual(
                    (deployment_public / "selected.txt").read_text(encoding="utf-8"),
                    "generation-public",
                )
                self.assertEqual(
                    (deployment_client / "shard.txt").read_text(encoding="utf-8"),
                    "generation-client-0",
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
            for kind, logical_root in roots.items():
                logical_root.mkdir()
                (logical_root / "legacy.txt").write_text(kind, encoding="utf-8")
                stages[kind].mkdir(parents=True, exist_ok=True)
                (stages[kind] / "new.txt").write_text(kind, encoding="utf-8")

            original_replace = os.replace

            def reject_pointer(source: str | Path, destination: str | Path) -> None:
                if Path(destination).name == ".prepared-current":
                    raise OSError("injected pointer failure")
                original_replace(source, destination)

            with (
                patch("src.data_prep.os.replace", side_effect=reject_pointer),
                self.assertRaisesRegex(OSError, "pointer failure"),
            ):
                _publish_prepared_roots(roots, stages)

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

            def crash_at(value):
                if value == phase:
                    os._exit(91)

            data_prep._publication_checkpoint = crash_at
            data_prep._publish_prepared_roots(roots, stages)
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
                for name, logical_root in roots.items():
                    logical_root.mkdir()
                    (logical_root / "legacy.txt").write_text(name, encoding="utf-8")
                    (stage / name).mkdir(parents=True)
                    (stage / name / "new.txt").write_text(name, encoding="utf-8")

                completed = subprocess.run(
                    [sys.executable, "-c", child_code, str(root), phase],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy(),
                )

                self.assertEqual(completed.returncode, 91, completed.stderr)
                _recover_prepared_migration(roots)
                for name, logical_root in roots.items():
                    selected = resolve_prepared_artifact_dir(logical_root, name)
                    self.assertEqual(
                        (selected / "new.txt").read_text(encoding="utf-8"), name
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
            for name, logical_root in roots.items():
                logical_root.mkdir()
                (logical_root / "legacy.txt").write_text(name, encoding="utf-8")
                (stage / name).mkdir(parents=True)
                (stage / name / "new.txt").write_text(name, encoding="utf-8")
            child_code = textwrap.dedent(
                """
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

                def crash_at(value):
                    if value == "archive:client:renamed":
                        os._exit(92)

                data_prep._publication_checkpoint = crash_at
                data_prep._publish_prepared_roots(roots, stages)
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
                patch("src.data_prep._validated_frameworks") as frameworks,
                patch("src.data_prep.load_verified_imdb_dataset") as load_dataset,
            ):
                prepare_all(4, roots["client"], roots["public"], roots["evaluation"])

            frameworks.assert_not_called()
            load_dataset.assert_not_called()
            self.assertEqual(
                (
                    resolve_prepared_artifact_dir(roots["public"], "public") / "new.txt"
                ).read_text(encoding="utf-8"),
                "public",
            )

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
            finally:
                lock.release()


if __name__ == "__main__":
    unittest.main()
