import copy
import hashlib
import json
import tomllib
import unittest
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_PATH = Path("docs/scientific-protocol-v1.toml")


def classification_metrics(
    config: dict[str, Any], labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Execute the registered binary-classification metric contract.

    Parameters
    ----------
    config : dict[str, Any]
        Parsed ``metrics.classification`` protocol table.
    labels : numpy.ndarray
        Rank-one binary labels with exact ``int64`` dtype.
    probabilities : numpy.ndarray
        Rank-one sigmoid probabilities with exact ``float32`` dtype.

    Returns
    -------
    dict[str, Any]
        Predictions, confusion matrix, scalar metrics, and ROC construction.

    Raises
    ------
    ValueError
        If either input violates the registered evaluation boundary.
    """
    if (
        not isinstance(labels, np.ndarray)
        or not isinstance(probabilities, np.ndarray)
        or labels.ndim != 1
        or probabilities.ndim != 1
        or labels.size == 0
        or labels.shape != probabilities.shape
    ):
        raise ValueError("classification inputs must be same-length nonempty vectors")
    if labels.dtype != np.dtype(config["label_dtype"]):
        raise ValueError("classification labels must use exactly int64")
    if probabilities.dtype != np.dtype(config["prediction_dtype"]):
        raise ValueError("classification probabilities must use exactly float32")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("classification labels must be binary")
    if not np.all(np.isfinite(probabilities)) or not np.all(
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ):
        raise ValueError("classification probabilities must be finite probabilities")

    positives = int(np.count_nonzero(labels == config["positive_label"]))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("classification ROC requires both classes")

    predicted_positive = probabilities >= np.float32(config["decision_threshold"])
    true_negative = int(np.count_nonzero((labels == 0) & ~predicted_positive))
    false_positive = int(np.count_nonzero((labels == 0) & predicted_positive))
    false_negative = int(np.count_nonzero((labels == 1) & ~predicted_positive))
    true_positive = int(np.count_nonzero((labels == 1) & predicted_positive))
    total = true_negative + false_positive + false_negative + true_positive
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative

    thresholds = np.concatenate(([np.inf], np.unique(probabilities)[::-1]))
    fpr = np.empty(thresholds.size, dtype=np.float64)
    tpr = np.empty(thresholds.size, dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        threshold_predictions = probabilities >= threshold
        tpr[index] = np.count_nonzero(threshold_predictions & (labels == 1)) / positives
        fpr[index] = np.count_nonzero(threshold_predictions & (labels == 0)) / negatives

    return {
        "predicted_positive": predicted_positive,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "accuracy": (true_negative + true_positive) / total,
        "precision": (
            true_positive / precision_denominator
            if precision_denominator
            else float(config["zero_division"])
        ),
        "recall": (
            true_positive / recall_denominator
            if recall_denominator
            else float(config["zero_division"])
        ),
        "f1": (
            2 * true_positive / f1_denominator
            if f1_denominator
            else float(config["zero_division"])
        ),
        "roc_thresholds": thresholds,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": float(np.trapezoid(tpr, fpr)),
    }


def canonical_json_bytes(config: dict[str, Any], value: Any) -> bytes:
    """Serialize a provenance value under the registered canonical contract.

    Parameters
    ----------
    config : dict[str, Any]
        Parsed ``provenance.canonical_serialization`` table.
    value : Any
        JSON-compatible value to serialize.

    Returns
    -------
    bytes
        Canonical UTF-8 JSON bytes with the registered trailing newline.
    """
    separators = tuple(config["separators"])
    serialized = json.dumps(
        value,
        ensure_ascii=config["ensure_ascii"],
        sort_keys=config["sort_keys"],
        separators=separators,
        allow_nan=config["allow_nan"],
    ).encode(config["encoding"])
    return serialized + (b"\n" if config["trailing_newline"] else b"")


def _require_exact_fields(
    value: dict[str, Any], expected: list[str], path: str
) -> None:
    """Require one provenance object to have its exact versioned field set.

    Parameters
    ----------
    value : dict[str, Any]
        Provenance object under validation.
    expected : list[str]
        Exact registered field names.
    path : str
        Human-readable object path for failures.

    Returns
    -------
    None
        Validation succeeds without mutation.

    Raises
    ------
    ValueError
        If a field is missing or additional.
    """
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{path} does not match its registered field set")


def validate_provenance_manifest(
    protocol: dict[str, Any], manifest: dict[str, Any]
) -> bytes:
    """Validate the golden future-runner manifest and return canonical bytes.

    Parameters
    ----------
    protocol : dict[str, Any]
        Parsed scientific protocol.
    manifest : dict[str, Any]
        Candidate experiment provenance manifest.

    Returns
    -------
    bytes
        Canonical serialized manifest.

    Raises
    ------
    ValueError
        If the schema, identities, ordering, or retry relationships are invalid.
    """
    provenance = protocol["provenance"]
    required = provenance["required_fields"]
    _require_exact_fields(manifest, provenance["required_top_level_fields"], "manifest")
    for field in (
        "schema",
        "cell",
        "config",
        "code",
        "environment",
        "hardware",
        "data",
        "seeds",
        "execution",
        "results",
    ):
        _require_exact_fields(manifest[field], required[field], field)
    _require_exact_fields(manifest["data"]["dataset"], required["dataset"], "dataset")
    _require_exact_fields(
        manifest["data"]["vocabulary"], required["vocabulary"], "vocabulary"
    )
    _require_exact_fields(
        manifest["data"]["partition"], required["partition"], "partition"
    )
    for split in manifest["data"]["dataset"]["splits"]:
        _require_exact_fields(split, required["dataset_split"], "dataset split")
    for untracked_file in manifest["code"]["untracked_files"]:
        _require_exact_fields(
            untracked_file, required["untracked_file"], "untracked file"
        )
    for accelerator in manifest["hardware"]["accelerators"]:
        _require_exact_fields(accelerator, required["accelerator"], "accelerator")
    for derived_seed in manifest["seeds"]["derived"]:
        _require_exact_fields(derived_seed, required["derived_seed"], "derived seed")
    for artifact in manifest["artifacts"]:
        _require_exact_fields(artifact, required["artifact"], "artifact")

    if manifest["schema"] != {
        "name": provenance["schema_name"],
        "version": provenance["schema_version"],
    }:
        raise ValueError("provenance schema identity is invalid")
    if manifest["config"]["protocol_version"] != protocol["protocol_version"]:
        raise ValueError("provenance protocol version is invalid")
    cell_fields = {
        field: manifest["cell"][field]
        for field in protocol["seeding"]["cell_id"]["fields"]
    }
    cell_id = canonical_json_bytes(
        {**provenance["canonical_serialization"], "trailing_newline": False},
        cell_fields,
    ).decode("ascii")
    if manifest["cell"]["cell_id"] != cell_id:
        raise ValueError("provenance cell_id is not canonical")

    canonical_config = provenance["canonical_serialization"]
    effective_hash = hashlib.sha256(
        canonical_json_bytes(canonical_config, manifest["config"]["effective"])
    ).hexdigest()
    if manifest["config"]["effective_sha256"] != effective_hash:
        raise ValueError("effective configuration hash is invalid")
    if manifest["code"]["dirty"] is False and (
        manifest["code"]["tracked_diff_sha256"] != hashlib.sha256(b"").hexdigest()
        or manifest["code"]["untracked_files"]
    ):
        raise ValueError("clean code identity is inconsistent")

    attempts = manifest["execution"]["attempts"]
    for index, attempt in enumerate(attempts):
        _require_exact_fields(attempt, required["attempt"], "attempt")
        if attempt["failure"] is not None:
            _require_exact_fields(attempt["failure"], required["failure"], "failure")
        expected_retry = None if index == 0 else index - 1
        if attempt["attempt"] != index or attempt["retry_of"] != expected_retry:
            raise ValueError("attempt indices or retry relationships are invalid")
        if (attempt["status"] == "succeeded") != (attempt["failure"] is None):
            raise ValueError("attempt status and failure are inconsistent")

    artifact_paths = [artifact["path"] for artifact in manifest["artifacts"]]
    if artifact_paths != sorted(set(artifact_paths)):
        raise ValueError("artifact paths must be unique and sorted")
    seed_keys = [
        (seed["namespace"].encode("utf-8"), seed["attempt"])
        for seed in manifest["seeds"]["derived"]
    ]
    if seed_keys != sorted(set(seed_keys)):
        raise ValueError("derived seeds must be unique and sorted")
    return canonical_json_bytes(canonical_config, manifest)


class ClassificationEvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Load the frozen protocol once for executable contract tests."""
        cls.protocol = tomllib.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.config = cls.protocol["metrics"]["classification"]

    def test_inference_and_reporting_configuration_is_exact(self) -> None:
        self.assertEqual(self.config["prediction_dtype"], "float32")
        self.assertEqual(self.config["label_dtype"], "int64")
        self.assertEqual(self.config["inference_batch_size"], 256)
        self.assertFalse(self.config["inference_drop_remainder"])
        self.assertEqual(self.config["decision_threshold"], 0.5)
        self.assertEqual(
            self.config["threshold_comparison"],
            "probability_greater_than_or_equal",
        )
        self.assertEqual(
            self.config["threshold_applies_to"],
            [
                "every_server_validation_union_evaluation",
                "every_final_untouched_test_evaluation",
                "every_centralized_validation_evaluation",
                "every_local_only_validation_evaluation",
            ],
        )
        self.assertTrue(self.config["threshold_applies_to_every_registered_model"])
        self.assertFalse(self.config["threshold_tuning"])
        self.assertEqual(
            self.config["reported"],
            [
                "accuracy",
                "confusion_matrix",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ],
        )

    def test_classification_golden_vectors_execute_exactly(self) -> None:
        for vector in self.config["golden_vectors"]:
            with self.subTest(name=vector["name"]):
                actual = classification_metrics(
                    self.config,
                    np.asarray(vector["labels"], dtype=np.int64),
                    np.asarray(vector["probabilities"], dtype=np.float32),
                )
                np.testing.assert_array_equal(
                    actual["predicted_positive"], vector["predicted_positive"]
                )
                self.assertEqual(actual["confusion_matrix"], vector["confusion_matrix"])
                for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                    self.assertEqual(actual[metric], vector[metric])
                expected_thresholds = np.asarray(
                    [
                        np.inf if threshold == "inf" else np.float32(threshold)
                        for threshold in vector["roc_thresholds"]
                    ],
                    dtype=np.float64,
                )
                np.testing.assert_array_equal(
                    actual["roc_thresholds"], expected_thresholds
                )
                np.testing.assert_array_equal(actual["fpr"], vector["fpr"])
                np.testing.assert_array_equal(actual["tpr"], vector["tpr"])

    def test_classification_rejects_hostile_inputs_without_repair(self) -> None:
        labels = np.array([0, 1], dtype=np.int64)
        probabilities = np.array([0.25, 0.75], dtype=np.float32)
        hostile = [
            (labels.astype(np.int32), probabilities),
            (labels, probabilities.astype(np.float64)),
            (labels[:1], probabilities),
            (labels.reshape(1, 2), probabilities),
            (np.array([0, 2], dtype=np.int64), probabilities),
            (np.array([1, 1], dtype=np.int64), probabilities),
            (labels, np.array([np.nan, 0.5], dtype=np.float32)),
            (labels, np.array([np.inf, 0.5], dtype=np.float32)),
            (labels, np.array([-0.1, 0.5], dtype=np.float32)),
            (labels, np.array([0.5, 1.1], dtype=np.float32)),
        ]
        for hostile_labels, hostile_probabilities in hostile:
            with self.subTest(
                labels=hostile_labels, probabilities=hostile_probabilities
            ):
                with self.assertRaises(ValueError):
                    classification_metrics(
                        self.config, hostile_labels, hostile_probabilities
                    )


class ProvenanceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Load the frozen protocol and canonical provenance probe."""
        cls.protocol = tomllib.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.provenance = cls.protocol["provenance"]
        cls.manifest = json.loads(cls.provenance["golden_probe"]["canonical_json"])

    def test_schema_required_fields_and_canonical_hash_are_exact(self) -> None:
        canonical = validate_provenance_manifest(self.protocol, self.manifest)
        probe = self.provenance["golden_probe"]

        self.assertEqual(
            canonical,
            probe["canonical_json"].encode("utf-8") + b"\n",
        )
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(), probe["manifest_sha256"]
        )
        self.assertEqual(
            self.manifest["config"]["effective_sha256"],
            probe["effective_config_sha256"],
        )
        sidecar = (
            f"{probe['manifest_sha256']}  {self.provenance['manifest_filename']}\n"
        )
        self.assertEqual(
            sidecar,
            "cb92a4f27ad815de9d6cb18ef0b22b898023cf27df19ded0e83250969f39afba"
            "  provenance.json\n",
        )

    def test_schema_rejects_missing_extra_noncanonical_and_nonfinite_data(
        self,
    ) -> None:
        hostile_manifests = []
        missing = copy.deepcopy(self.manifest)
        del missing["hardware"]["memory_bytes"]
        hostile_manifests.append(missing)
        extra = copy.deepcopy(self.manifest)
        extra["results"]["unregistered"] = True
        hostile_manifests.append(extra)
        wrong_cell_id = copy.deepcopy(self.manifest)
        wrong_cell_id["cell"]["cell_id"] = "not-canonical"
        hostile_manifests.append(wrong_cell_id)
        wrong_retry = copy.deepcopy(self.manifest)
        wrong_retry["execution"]["attempts"][1]["retry_of"] = None
        hostile_manifests.append(wrong_retry)
        unsorted_artifacts = copy.deepcopy(self.manifest)
        unsorted_artifacts["artifacts"].reverse()
        hostile_manifests.append(unsorted_artifacts)

        for manifest in hostile_manifests:
            with self.subTest(manifest=manifest):
                with self.assertRaises(ValueError):
                    validate_provenance_manifest(self.protocol, manifest)

        nonfinite = copy.deepcopy(self.manifest)
        nonfinite["results"]["metrics"]["accuracy"] = np.nan
        with self.assertRaises(ValueError):
            validate_provenance_manifest(self.protocol, nonfinite)


if __name__ == "__main__":
    unittest.main()
