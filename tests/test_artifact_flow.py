import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.app_manifest import AppManifest, load_app_manifest
from src.data_prep import main, package_raw_client_shards
from src.local_training import (
    build_model_from_manifest,
    load_client_shard,
)


def write_public_artifacts(path: Path, sequence_length: int = 4) -> AppManifest:
    vocabulary = "\n[UNK]\ngood\nbad\nmovie"
    (path / "vocab.txt").write_text(vocabulary, encoding="utf-8")
    payload = {
        "embedding_dim": 100,
        "sequence_length": sequence_length,
        "vocabulary_size": 5,
        "vocabulary": {"filename": "vocab.txt"},
    }
    manifest_path = path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return AppManifest(payload, path / "vocab.txt")


def check_public_manifest_loads_the_model_shape(tmp_path: Path) -> None:
    write_public_artifacts(tmp_path, sequence_length=500)

    manifest = load_app_manifest(public_artifact_dir=tmp_path)

    assert manifest.vocabulary_path == tmp_path.resolve() / "vocab.txt"
    assert manifest.payload["embedding_dim"] == 100


def check_raw_client_packaging_keeps_every_sample_private(tmp_path: Path) -> None:
    texts = np.asarray([f"review {index}\nline" for index in range(12)])
    labels = np.asarray([0, 1] * 6)

    shards = package_raw_client_shards(texts, labels, tmp_path, num_clients=4)

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
    manifest = write_public_artifacts(tmp_path)
    records = [
        {"text": "good movie", "label": 1},
        {"text": "bad movie", "label": 0},
        {"text": "unknown movie", "label": 0},
        {"text": "good", "label": 1},
    ]
    (tmp_path / "reviews.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    (train_x, train_y), (val_x, val_y) = load_client_shard(
        tmp_path, manifest, validation_split=0.25
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
    manifest = write_public_artifacts(tmp_path)
    records = [
        {"text": "good\u2028movie", "label": 1},
        {"text": "bad movie", "label": 0},
    ]
    (tmp_path / "reviews.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    (train_x, train_y), (val_x, val_y) = load_client_shard(
        tmp_path, manifest, validation_split=0.5
    )

    assert train_x.shape == (1, 4)
    assert val_x.shape == (1, 4)
    np.testing.assert_array_equal(np.sort(np.concatenate([train_y, val_y])), [0, 1])


def check_client_loader_preserves_control_character_in_vocabulary(
    tmp_path: Path,
) -> None:
    manifest = write_public_artifacts(tmp_path)
    (tmp_path / "vocab.txt").write_text(
        "\n[UNK]\ngood\nbad\nmovie\n\x85\nrare", encoding="utf-8"
    )
    records = [
        {"text": "good movie", "label": 1},
        {"text": "bad movie", "label": 0},
    ]
    (tmp_path / "reviews.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    (train_x, train_y), (val_x, val_y) = load_client_shard(
        tmp_path, manifest, validation_split=0.5
    )

    assert train_x.shape == (1, 4)
    assert val_x.shape == (1, 4)
    np.testing.assert_array_equal(np.sort(np.concatenate([train_y, val_y])), [0, 1])


def check_cli_prepares_all_artifacts_with_one_dataset_load(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    client_dir = tmp_path / "clients"
    public_dir.mkdir()
    client_dir.mkdir()
    (public_dir / "keep.txt").write_text("keep", encoding="utf-8")
    (client_dir / "keep.txt").write_text("keep", encoding="utf-8")
    texts = np.asarray([f"Review {index} good movie" for index in range(12)])
    labels = np.asarray([index % 2 for index in range(12)], dtype="int32")

    with patch(
        "src.data_prep.load_imdb_training_data", return_value=(texts, labels)
    ) as load_dataset:
        main(
            [
                "--partitions",
                "4",
                "--client-shard-dir",
                str(client_dir),
                "--public-artifact-dir",
                str(public_dir),
            ]
        )

    load_dataset.assert_called_once_with()
    assert not list(tmp_path.rglob("*.npy"))
    manifest = json.loads((public_dir / "manifest.json").read_text(encoding="utf-8"))
    vocabulary = (public_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    assert manifest["vocabulary_size"] == len(vocabulary)
    assert "glove" not in json.dumps(manifest).lower()
    assert "embedding_matrix" not in manifest
    assert not list(client_dir.glob("client-*.tar.gz"))
    assert all(
        (client_dir / f"client-{index}" / "reviews.jsonl").exists()
        for index in range(4)
    )
    assert (public_dir / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (client_dir / "keep.txt").read_text(encoding="utf-8") == "keep"


class ArtifactFlowTests(unittest.TestCase):
    def run_with_temp_dir(self, check) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            check(Path(tmpdir))

    def test_public_manifest_loads_the_model_shape(self) -> None:
        self.run_with_temp_dir(check_public_manifest_loads_the_model_shape)

    def test_raw_client_packaging_keeps_every_sample_private(self) -> None:
        self.run_with_temp_dir(check_raw_client_packaging_keeps_every_sample_private)

    def test_client_tokenizes_its_raw_reviews_with_public_vocabulary(self) -> None:
        self.run_with_temp_dir(
            check_client_tokenizes_its_raw_reviews_with_public_vocabulary
        )

    def test_client_loader_preserves_unicode_line_separator_in_review(self) -> None:
        self.run_with_temp_dir(
            check_client_loader_preserves_unicode_line_separator_in_review
        )

    def test_client_loader_preserves_control_character_in_vocabulary(self) -> None:
        self.run_with_temp_dir(
            check_client_loader_preserves_control_character_in_vocabulary
        )

    def test_cli_prepares_all_artifacts_with_one_dataset_load(self) -> None:
        self.run_with_temp_dir(check_cli_prepares_all_artifacts_with_one_dataset_load)


if __name__ == "__main__":
    unittest.main()
