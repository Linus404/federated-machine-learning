import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np


PROTOCOL_PATH = Path("docs/scientific-protocol-v1.toml")


def derive_seed(
    protocol: dict[str, Any], master_seed: int, namespace: str, attempt: int = 0
) -> int:
    """Derive one protocol RNG seed.

    Args:
        protocol: Parsed scientific protocol.
        master_seed: Registered experiment seed.
        namespace: Fully expanded ASCII RNG namespace.
        attempt: Zero-based retry attempt.

    Returns:
        Unsigned 64-bit integer encoded by the first eight SHA-256 bytes.
    """
    self_contained_material = f"{master_seed}|{namespace}|{attempt}".encode("ascii")
    return int.from_bytes(
        hashlib.sha256(self_contained_material).digest()[:8],
        protocol["seeding"]["uint64_byte_order"],
        signed=False,
    )


def generator(
    protocol: dict[str, Any], master_seed: int, namespace: str, attempt: int = 0
) -> np.random.Generator:
    """Instantiate a fresh Generator for one derived seed.

    Args:
        protocol: Parsed scientific protocol.
        master_seed: Registered experiment seed.
        namespace: Fully expanded ASCII RNG namespace.
        attempt: Zero-based retry attempt.

    Returns:
        A new NumPy Generator backed by PCG64.
    """
    return np.random.Generator(
        np.random.PCG64(derive_seed(protocol, master_seed, namespace, attempt))
    )


def official_labels(protocol: dict[str, Any]) -> np.ndarray:
    """Build the offline official-train label vector from frozen index ranges.

    Args:
        protocol: Parsed scientific protocol.

    Returns:
        Labels in zero-based official row-index order.
    """
    split = protocol["dataset"]["splits"]["train"]
    labels = np.empty(split["rows"], dtype=np.int64)
    for label, (first, last) in zip(
        protocol["partitioning"]["labels"], split["label_index_ranges"], strict=True
    ):
        labels[first : last + 1] = label
    return labels


def iid_partition(
    protocol: dict[str, Any], labels: np.ndarray, seed: int, client_scale: int
) -> tuple[int, list[np.ndarray]]:
    """Execute the registered IID partition algorithm offline.

    Args:
        protocol: Parsed scientific protocol.
        labels: Official train labels by row index.
        seed: Registered master seed.
        client_scale: Number of clients.

    Returns:
        Attempt zero and sorted row-index shards in client order.
    """
    allocations: list[list[np.ndarray]] = [[] for _ in range(client_scale)]
    template = protocol["seeding"]["namespaces"]["partition_iid"]
    for label in protocol["partitioning"]["labels"]:
        rows = np.flatnonzero(labels == label)
        namespace = template.format(client_scale=client_scale, label=label)
        shuffled = generator(protocol, seed, namespace).permutation(rows)
        for client_id, part in enumerate(np.array_split(shuffled, client_scale)):
            allocations[client_id].append(part)
    return 0, [np.sort(np.concatenate(parts)) for parts in allocations]


def dirichlet_partition(
    protocol: dict[str, Any],
    labels: np.ndarray,
    seed: int,
    partition_name: str,
    alpha: float,
    client_scale: int,
) -> tuple[int, list[np.ndarray]]:
    """Execute the registered Dirichlet partition and retry algorithm offline.

    Args:
        protocol: Parsed scientific protocol.
        labels: Official train labels by row index.
        seed: Registered master seed.
        partition_name: Registered Dirichlet partition name.
        alpha: Registered Dirichlet concentration.
        client_scale: Number of clients.

    Returns:
        Accepted zero-based attempt and sorted row-index shards.

    Raises:
        ValueError: If no attempt satisfies the minimum client size.
    """
    partitioning = protocol["partitioning"]
    template = protocol["seeding"]["namespaces"]["partition_dirichlet"]
    for attempt in range(
        partitioning["first_attempt"], partitioning["last_attempt"] + 1
    ):
        label_allocations: list[tuple[np.ndarray, np.ndarray]] = []
        client_sizes = np.zeros(client_scale, dtype=np.int64)
        for label in partitioning["labels"]:
            rows = np.flatnonzero(labels == label)
            namespace = template.format(
                partition_name=partition_name,
                client_scale=client_scale,
                label=label,
            )
            rng = generator(protocol, seed, namespace, attempt)
            shuffled = rng.permutation(rows)
            probabilities = rng.dirichlet(np.full(client_scale, alpha))
            counts = rng.multinomial(rows.size, probabilities)
            label_allocations.append((shuffled, counts))
            client_sizes += counts
        if client_sizes.min() >= partitioning["minimum_samples_per_client"]:
            allocations: list[list[np.ndarray]] = [[] for _ in range(client_scale)]
            for shuffled, counts in label_allocations:
                boundaries = np.concatenate(([0], np.cumsum(counts)))
                for client_id in range(client_scale):
                    allocations[client_id].append(
                        shuffled[boundaries[client_id] : boundaries[client_id + 1]]
                    )
            shards = [np.sort(np.concatenate(parts)) for parts in allocations]
            return attempt, shards
    raise ValueError(
        f"no accepted {partition_name}/{client_scale} partition for seed {seed}"
    )


def validation_split(
    protocol: dict[str, Any],
    labels: np.ndarray,
    seed: int,
    partition_name: str,
    client_scale: int,
    shards: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Execute deterministic client-local stratified validation splitting.

    Args:
        protocol: Parsed scientific protocol.
        labels: Official train labels by row index.
        seed: Registered master seed.
        partition_name: Registered partition name.
        client_scale: Number of clients.
        shards: Accepted client shards in numeric client order.

    Returns:
        Sorted fitted and validation row-index arrays by client.
    """
    template = protocol["seeding"]["namespaces"]["validation"]
    fraction = protocol["training"]["validation_fraction"]
    fitted_by_client: list[np.ndarray] = []
    validation_by_client: list[np.ndarray] = []
    for client_id, shard in enumerate(shards):
        fitted_parts: list[np.ndarray] = []
        validation_parts: list[np.ndarray] = []
        for label in protocol["partitioning"]["labels"]:
            rows = np.sort(shard[labels[shard] == label])
            namespace = template.format(
                partition_name=partition_name,
                client_scale=client_scale,
                client_id=client_id,
                label=label,
            )
            shuffled = generator(protocol, seed, namespace).permutation(rows)
            validation_size = math.floor(rows.size * fraction)
            validation_parts.append(shuffled[:validation_size])
            fitted_parts.append(shuffled[validation_size:])
        fitted_by_client.append(np.sort(np.concatenate(fitted_parts)))
        validation_by_client.append(np.sort(np.concatenate(validation_parts)))
    return fitted_by_client, validation_by_client


def select_update_client(
    protocol: dict[str, Any], labels: np.ndarray, fitted: list[np.ndarray]
) -> int:
    """Select the lowest feasible update-leakage client.

    Args:
        protocol: Parsed scientific protocol.
        labels: Official train labels by row index.
        fitted: Fitted row-index arrays in numeric client order.

    Returns:
        The lowest qualifying zero-based client ID.

    Raises:
        ValueError: If no client has the fixed count for every label.
    """
    count = protocol["privacy"]["update_leakage"]["records_per_class_per_target_label"]
    for client_id, rows in enumerate(fitted):
        if all(
            np.count_nonzero(labels[rows] == label) >= count
            for label in protocol["partitioning"]["labels"]
        ):
            return client_id
    raise ValueError("no update-leakage client has enough fitted rows per label")


def malicious_clients(
    protocol: dict[str, Any], seed: int, malicious_count: int
) -> list[int]:
    """Execute the registered nested malicious-client selection.

    Args:
        protocol: Parsed scientific protocol.
        seed: Registered master seed.
        malicious_count: Number of malicious clients to select.

    Returns:
        Sorted zero-based malicious client IDs.
    """
    client_scale = protocol["threats"]["client_scale"]
    namespace = protocol["seeding"]["namespaces"]["malicious_order"]
    order = generator(protocol, seed, namespace).permutation(client_scale)
    return sorted(int(client_id) for client_id in order[:malicious_count])


def privacy_sample(
    protocol: dict[str, Any],
    labels: np.ndarray,
    seed: int,
    namespace: str,
    eligible: np.ndarray,
    count_per_label: int,
) -> np.ndarray:
    """Select one registered balanced privacy candidate pool.

    Args:
        protocol: Parsed scientific protocol.
        labels: Labels for the candidate split by row index.
        seed: Registered master seed.
        namespace: Namespace template containing ``{label}``.
        eligible: Eligible row indices in the candidate split.
        count_per_label: Fixed number of records selected per label.

    Returns:
        Selected indices concatenated in ascending label order.

    Raises:
        ValueError: If any label pool has too few eligible rows.
    """
    selected: list[np.ndarray] = []
    for label in protocol["partitioning"]["labels"]:
        rows = np.sort(eligible[labels[eligible] == label])
        if rows.size < count_per_label:
            raise ValueError(f"label {label} has {rows.size} eligible rows")
        label_namespace = namespace.replace("{label}", str(label))
        selected.append(
            generator(protocol, seed, label_namespace).permutation(rows)[
                :count_per_label
            ]
        )
    return np.concatenate(selected)


def privacy_binary_crossentropy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    clip_minimum: float,
    clip_maximum: float,
) -> np.ndarray:
    """Compute the registered unreduced privacy binary cross-entropy.

    Args:
        labels: Binary membership labels.
        probabilities: Predicted positive-class probabilities.
        clip_minimum: Inclusive lower probability bound.
        clip_maximum: Inclusive upper probability bound.

    Returns:
        Float64 loss for each input pair.
    """
    y = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probability, clip_minimum, clip_maximum)
    return -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))


def privacy_roc(
    labels: np.ndarray, scores: np.ndarray, target_fpr: float
) -> dict[str, Any]:
    """Compute the registered tie-aware privacy ROC summaries.

    Args:
        labels: Binary membership labels with members encoded as one.
        scores: Finite attack scores where larger means more likely member.
        target_fpr: False-positive rate used for interpolated TPR reporting.

    Returns:
        Thresholds, ROC points, AUC, maximum advantage, and target-FPR TPR.

    Raises:
        ValueError: If shapes differ, values are invalid, or a class is absent.
    """
    y = np.asarray(labels, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    if y.shape != score.shape or y.ndim != 1:
        raise ValueError("privacy labels and scores must be same-length vectors")
    if not np.all(np.isfinite(score)) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("privacy labels must be binary and scores must be finite")
    positives = int(np.count_nonzero(y == 1))
    negatives = int(np.count_nonzero(y == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("privacy ROC requires both membership classes")

    thresholds = np.concatenate(([np.inf], np.unique(score)[::-1]))
    fpr = np.empty(thresholds.size, dtype=np.float64)
    tpr = np.empty(thresholds.size, dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        predicted_positive = score >= threshold
        tpr[index] = np.count_nonzero(predicted_positive & (y == 1)) / positives
        fpr[index] = np.count_nonzero(predicted_positive & (y == 0)) / negatives

    advantage = tpr - fpr
    best_index = int(np.argmax(advantage))
    distinct_fpr = np.unique(fpr)
    maximum_tpr = np.asarray([tpr[fpr == value].max() for value in distinct_fpr])
    return {
        "thresholds": thresholds,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": float(np.trapezoid(tpr, fpr)),
        "max_tpr_minus_fpr": float(advantage[best_index]),
        "max_threshold": float(thresholds[best_index]),
        "tpr_at_target_fpr": float(np.interp(target_fpr, distinct_fpr, maximum_tpr)),
    }


def canonical_cell_id(cell: dict[str, Any]) -> str:
    """Serialize one matrix cell as its canonical identifier.

    Args:
        cell: Complete canonical cell field mapping.

    Returns:
        Compact sorted ASCII JSON.
    """
    return json.dumps(
        cell,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def cell(
    matrix_kind: str,
    strategy: str,
    partition: str | None,
    client_scale: int | None,
    seed: int,
    threat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the complete canonical field mapping for one cell.

    Args:
        matrix_kind: Matrix subsection name.
        strategy: Registered strategy or privacy source model.
        partition: Registered partition or ``None``.
        client_scale: Registered client count or ``None``.
        seed: Registered master seed.
        threat: Exact robustness threat mapping or ``None``.

    Returns:
        Canonical cell fields with derived alpha.
    """
    alpha = None if partition in (None, "iid_stratified") else float(partition[10:])
    return {
        "matrix_kind": matrix_kind,
        "strategy": strategy,
        "partition": partition,
        "alpha": alpha,
        "client_scale": client_scale,
        "seed": seed,
        "threat": threat,
    }


def registered_cells(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand every registered matrix cell.

    Args:
        protocol: Parsed scientific protocol.

    Returns:
        Complete canonical field mappings for all registered cells.
    """
    matrix = protocol["matrix"]
    seeds = protocol["seeding"]["seeds"]
    cells: list[dict[str, Any]] = []
    for matrix_kind in ("primary_federated", "local_only"):
        entry = matrix[matrix_kind]
        cells.extend(
            cell(matrix_kind, strategy, partition, entry["client_scale"], seed)
            for strategy, partition, seed in itertools.product(
                entry["strategies"], entry["partitions"], seeds
            )
        )
    centralized = matrix["centralized"]
    cells.extend(
        cell("centralized", strategy, None, None, seed)
        for strategy, seed in itertools.product(centralized["strategies"], seeds)
    )
    scale = matrix["scale"]
    cells.extend(
        cell("scale", strategy, partition, client_scale, seed)
        for client_scale in scale["client_scales"]
        for strategy, partition, seed in itertools.product(
            scale["strategies"],
            scale["partitions_by_client_scale"][str(client_scale)],
            seeds,
        )
    )
    robustness = matrix["robustness"]
    cells.extend(
        cell(
            "robustness",
            strategy,
            robustness["partition"],
            robustness["client_scale"],
            seed,
            {"attack": attack, "malicious_fraction": fraction},
        )
        for strategy, attack, fraction, seed in itertools.product(
            robustness["strategies"],
            robustness["attacks"],
            robustness["malicious_fractions"],
            seeds,
        )
    )
    membership = matrix["membership_inference"]
    cells.extend(
        cell(
            "membership_inference",
            model,
            None if model == "centralized" else membership["fedavg_partition"],
            None if model == "centralized" else membership["fedavg_client_scale"],
            seed,
        )
        for model, seed in itertools.product(membership["models"], seeds)
    )
    leakage = matrix["update_leakage"]
    cells.extend(
        cell(
            "update_leakage",
            model,
            leakage["partition"],
            leakage["client_scale"],
            seed,
        )
        for model, seed in itertools.product(leakage["models"], seeds)
    )
    return cells


def paired_contrast_pairs(
    protocol: dict[str, Any], contrast: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Expand one registered paired contrast into candidate-baseline cells.

    Args:
        protocol: Parsed scientific protocol.
        contrast: One statistics.paired_contrasts entry.

    Returns:
        Candidate and mapped baseline cells in canonical cell order.
    """
    all_cells = registered_cells(protocol)
    registered_ids = {
        canonical_cell_id(registered_cell) for registered_cell in all_cells
    }
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for candidate in all_cells:
        if candidate["matrix_kind"] != contrast["candidate_matrix"]:
            continue
        if any(
            candidate[field] in excluded_values
            for field, excluded_values in contrast["candidate_exclude"].items()
        ):
            continue
        baseline = {
            "matrix_kind": contrast["baseline_matrix"],
            **{field: candidate[field] for field in contrast["preserve_fields"]},
            **contrast["set_fields"],
        }
        for field in contrast["clear_fields"]:
            baseline[field] = None
        if set(baseline) != set(candidate):
            raise ValueError(f"incomplete baseline mapping for {contrast['name']}")
        if canonical_cell_id(baseline) not in registered_ids:
            raise ValueError(f"unregistered baseline for {contrast['name']}")
        pairs.append((candidate, baseline))
    return pairs


def iterative_huber_aggregate(
    client_vectors: np.ndarray,
    sample_weights: np.ndarray,
    threshold: float,
    epsilon: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute the registered iterative Huber aggregation.

    Args:
        client_vectors: Finite flattened post-fit model parameter vectors.
        sample_weights: Positive fitted-row counts in client order.
        threshold: Registered Huber residual threshold.
        epsilon: Registered zero-residual denominator floor.
        iterations: Exact number of iteratively reweighted updates.

    Returns:
        Initial sample-weighted mean and final aggregate.
    """
    vectors = np.asarray(client_vectors, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float64)
    weights /= weights.sum()
    estimate = np.average(vectors, axis=0, weights=weights)
    initial = estimate.copy()
    for _ in range(iterations):
        residuals = np.linalg.norm(estimate - vectors, axis=1)
        effective = weights * np.where(
            residuals <= threshold,
            1.0,
            threshold / (residuals + epsilon),
        )
        effective /= effective.sum()
        estimate = np.average(vectors, axis=0, weights=effective)
    return initial, estimate


class ScientificProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
        cls.protocol = tomllib.loads(cls.raw_protocol)
        cls.labels = official_labels(cls.protocol)

    @classmethod
    def partition_results(
        cls,
    ) -> dict[tuple[int, str, int], tuple[int, list[np.ndarray]] | None]:
        """Execute each distinct partition used by registered cells once.

        Returns:
            Partition results keyed by seed, partition, and client scale. A
            value of ``None`` records the protocol's required retry exhaustion.
        """
        if hasattr(cls, "_partition_results"):
            return cls._partition_results
        keys = {
            (
                int(registered_cell["seed"]),
                str(registered_cell["partition"]),
                int(registered_cell["client_scale"]),
            )
            for registered_cell in registered_cells(cls.protocol)
            if registered_cell["partition"] is not None
        }
        keys.update(
            (seed, "iid_stratified", 4) for seed in cls.protocol["seeding"]["seeds"]
        )
        results: dict[tuple[int, str, int], tuple[int, list[np.ndarray]] | None] = {}
        for seed, partition_name, client_scale in sorted(keys):
            if partition_name == "iid_stratified":
                result = iid_partition(cls.protocol, cls.labels, seed, client_scale)
            else:
                alpha = float(partition_name[10:])
                try:
                    result = dirichlet_partition(
                        cls.protocol,
                        cls.labels,
                        seed,
                        partition_name,
                        alpha,
                        client_scale,
                    )
                except ValueError:
                    result = None
            results[(seed, partition_name, client_scale)] = result
        cls._partition_results = results
        return results

    def test_protocol_is_frozen_parseable_and_complete(self) -> None:
        self.assertEqual(self.protocol["protocol_version"], 1)
        self.assertEqual(self.protocol["status"], "frozen")
        self.assertIsNone(
            re.search(r"\b(?:TODO|TBD)\b", self.raw_protocol, re.IGNORECASE)
        )

    def test_dataset_identity_and_checksums_are_exact(self) -> None:
        dataset = self.protocol["dataset"]
        self.assertEqual(dataset["id"], "stanfordnlp/imdb")
        self.assertEqual(dataset["config"], "plain_text")
        self.assertEqual(
            dataset["revision"], "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
        )
        self.assertEqual(len(dataset["revision"]), 40)
        self.assertEqual(dataset["datasets_version"], "4.8.5")

        expected = {
            "train": (
                25000,
                "db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c",
                "4639bf1063fddebc3d7a446004a1032832227b0d3d28b01f58f99a4f94c0ab8a",
            ),
            "test": (
                25000,
                "b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da",
                "48168f7787435514498e777646a1312897c1207d239c61acdde5e049bc3f1a3f",
            ),
            "unsupervised": (
                50000,
                "74d14fbfcbb39fb7d299c38ca9f0ae6d231bf97108da85d620027ba437b6d52e",
                "d56c24962d82526173fc9fab14b820873ea281d64ab11915ec12eaf73f65deeb",
            ),
        }
        for split, (rows, raw_hash, content_hash) in expected.items():
            actual = dataset["splits"][split]
            self.assertEqual(actual["rows"], rows)
            self.assertEqual(actual["raw_parquet_sha256"], raw_hash)
            self.assertEqual(actual["content_sha256"], content_hash)
            self.assertRegex(raw_hash, r"^[0-9a-f]{64}$")
            self.assertRegex(content_hash, r"^[0-9a-f]{64}$")
        for split in ("train", "test"):
            split_spec = dataset["splits"][split]
            self.assertEqual(split_spec["label_counts"], [12500, 12500])
            self.assertEqual(
                split_spec["label_index_ranges"], [[0, 12499], [12500, 24999]]
            )
        self.assertTrue(
            np.array_equal(
                np.bincount(self.labels),
                self.protocol["dataset"]["splits"]["train"]["label_counts"],
            )
        )

    def test_content_hash_recipe_is_canonical_and_preserves_unicode(self) -> None:
        dataset = self.protocol["dataset"]
        self.assertEqual(
            dataset["content_hash_python"],
            'json.dumps({"label": int(row["label"]), "text": row["text"]}, '
            'ensure_ascii=False, sort_keys=True, separators=(",", ":"), '
            'allow_nan=False).encode("utf-8") + b"\\n"',
        )
        self.assertFalse(dataset["content_hash_json_ensure_ascii"])
        self.assertTrue(dataset["content_hash_json_sort_keys"])
        self.assertFalse(dataset["content_hash_json_allow_nan"])
        self.assertEqual(dataset["content_hash_json_separators"], [",", ":"])
        self.assertEqual(
            dataset["content_hash_unicode_normalization"],
            "none_preserve_dataset_code_points_exactly",
        )

        def encode(text: str) -> bytes:
            return (
                json.dumps(
                    {"label": 1, "text": text},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )

        row = {"text": "caf\N{LATIN SMALL LETTER E WITH ACUTE}", "label": 1}
        self.assertEqual(
            eval(dataset["content_hash_python"], {"json": json}, {"row": row}),
            encode(row["text"]),
        )
        self.assertEqual(
            encode("caf\N{LATIN SMALL LETTER E WITH ACUTE}"),
            b'{"label":1,"text":"caf\xc3\xa9"}\n',
        )
        self.assertNotEqual(
            encode("caf\N{LATIN SMALL LETTER E WITH ACUTE}"),
            encode("cafe\N{COMBINING ACUTE ACCENT}"),
        )

    def test_global_test_split_is_untouched(self) -> None:
        dataset = self.protocol["dataset"]
        untouched = self.protocol["untouched_test"]

        self.assertEqual(
            dataset["splits"]["test"]["use"], "sole_untouched_global_test_set"
        )
        self.assertEqual(dataset["splits"]["unsupervised"]["use"], "excluded")
        self.assertEqual(untouched["split"], "test")
        self.assertTrue(untouched["sole_global_test_set"])
        for prohibited_use in (
            "available_to_client_training",
            "available_to_validation",
            "available_to_tuning",
            "available_to_model_selection",
            "available_to_partitioning",
        ):
            self.assertFalse(untouched[prohibited_use])
        self.assertEqual(untouched["decision_threshold"], 0.5)
        self.assertIn("Never tune", untouched["selection_rule"])
        self.assertEqual(
            untouched["access"],
            "registered_final_model_evaluation_and_registered_privacy_analysis_only",
        )

    def test_registered_training_and_partition_constants_are_exact(self) -> None:
        self.assertEqual(self.protocol["seeding"]["seeds"], [67, 101, 211, 307, 401])

        partitioning = self.protocol["partitioning"]
        self.assertEqual(
            partitioning["registered"],
            ["iid_stratified", "dirichlet_1.0", "dirichlet_0.5", "dirichlet_0.1"],
        )
        self.assertEqual(partitioning["dirichlet_alphas"], [1.0, 0.5, 0.1])
        self.assertEqual(partitioning["minimum_samples_per_client"], 32)
        self.assertEqual(
            (partitioning["first_attempt"], partitioning["last_attempt"]), (0, 9999)
        )
        self.assertEqual(partitioning["client_scales"], [4, 16, 64])

        model = self.protocol["model"]
        self.assertEqual(
            (
                model["vocabulary_size"],
                model["sequence_length"],
                model["embedding_dimension"],
                model["convolution_filters"],
                model["convolution_kernel_size"],
                model["hidden_units"],
                model["dropout_rate"],
            ),
            (20000, 500, 100, 64, 3, 32, 0.3),
        )
        self.assertTrue(model["embedding_trainable"])
        self.assertEqual(model["padding_token_id"], 0)
        self.assertFalse(model["convolution_use_bias"])
        self.assertEqual(model["loss"], "keras.losses.BinaryCrossentropy")
        self.assertEqual(
            (
                model["loss_from_logits"],
                model["loss_label_smoothing"],
                model["loss_axis"],
                model["loss_reduction"],
                model["loss_probability_clip"],
            ),
            (False, 0.0, -1, "sum_over_batch_size", 1e-7),
        )

        training = self.protocol["training"]
        self.assertEqual(
            (
                training["rounds"],
                training["local_epochs"],
                training["batch_size"],
                training["validation_fraction"],
            ),
            (20, 1, 64, 0.2),
        )
        self.assertEqual(
            (
                training["learning_rate"],
                training["beta_1"],
                training["beta_2"],
                training["epsilon"],
            ),
            (0.001, 0.9, 0.999, 1e-7),
        )
        self.assertFalse(training["early_stopping"])
        self.assertFalse(training["update_noise"])

    def test_preprocessing_and_framework_reproducibility_are_executable(self) -> None:
        preprocessing = self.protocol["preprocessing"]
        framework = self.protocol["framework"]
        self.assertEqual(preprocessing["reserved_tokens"], ["", "[UNK]"])
        self.assertEqual(
            (
                preprocessing["padding_token_id"],
                preprocessing["oov_token_id"],
                preprocessing["oov_buckets"],
                preprocessing["learned_token_limit"],
            ),
            (0, 1, 1, 19998),
        )
        self.assertEqual(
            (
                preprocessing["max_tokens"],
                preprocessing["output_sequence_length"],
                preprocessing["vocabulary_size"],
            ),
            (20000, 500, 20000),
        )
        self.assertEqual(
            preprocessing["punctuation_characters"],
            r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""",
        )
        self.assertEqual(
            preprocessing["vocabulary_sha256"],
            "021e9f054e1b1ad4622fdb87d6d98f17337d202cc35faefdcfd8ae15a3887b9b",
        )
        self.assertRegex(preprocessing["vocabulary_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn(
            "official_train_entire_25000_rows", preprocessing["adaptation_split"]
        )
        self.assertIn("Never call adapt again", preprocessing["adaptation_lifecycle"])
        self.assertIn("first_500", preprocessing["truncation"])
        self.assertIn("token_id_0", preprocessing["padding"])
        self.assertEqual(preprocessing["layer_dtype"], "string")
        self.assertEqual(preprocessing["output_dtype"], "int64")
        self.assertIn(
            "ASCII U+0041 through U+005A", preprocessing["lowercase_semantics"]
        )
        self.assertIn("U+0009 through U+000D", preprocessing["tokenization"])
        self.assertIn("Non-ASCII whitespace", preprocessing["tokenization"])

        probe = r"""
import hashlib
import json
import re
import string
import sys

import keras
import numpy as np
import tensorflow as tf

config = json.loads(sys.argv[1])
model_config = json.loads(sys.argv[2])
tf.config.experimental.enable_op_determinism()
namespace = {"keras": keras, "re": re, "string": string, "tf": tf}
exec(config["standardize_python"], namespace)
vectorizer = eval(config["vectorizer_python"], namespace)
texts = ["<b>Alpha</b> beta!", "beta, GAMMA", "can't stop", "café alpha"]
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(2))
vocabulary = vectorizer.get_vocabulary()
token_ids = vectorizer(tf.constant(["<i>ALPHA</i> unknown!!!"]))[0, :5]
standardized_cases = [
    namespace["protocol_standardize"](tf.constant(case["input"])).numpy().decode("utf-8")
    for case in config["golden_cases"]
]
tokenized_cases = [
    [token.decode("utf-8") for token in tf.strings.split([text]).flat_values.numpy()]
    for text in standardized_cases
]

golden = model_config["loss_golden_probe"]
logits = tf.constant(golden["logits"], dtype=tf.float32)[:, None]
labels = tf.constant(golden["labels"], dtype=tf.float32)[:, None]
probabilities = keras.layers.Activation("sigmoid")(logits)
bce_arguments = {
    "from_logits": model_config["loss_from_logits"],
    "label_smoothing": model_config["loss_label_smoothing"],
    "axis": model_config["loss_axis"],
}
bce_elementwise = keras.losses.BinaryCrossentropy(
    **bce_arguments, reduction="none"
)(labels, probabilities)
bce_reduced = keras.losses.BinaryCrossentropy(
    **bce_arguments, reduction=model_config["loss_reduction"]
)(labels, probabilities)

def dropout_sequence():
    layer = keras.layers.Dropout(0.3, seed=54321)
    inputs = tf.ones((4, 8), dtype=tf.float32)
    return [layer(inputs, training=True).numpy(), layer(inputs, training=True).numpy()]

def training_hash():
    keras.backend.clear_session()
    keras.utils.set_random_seed(12345)
    inputs = keras.Input(shape=(4,), dtype="float32")
    hidden = keras.layers.Dense(5, activation="relu")(inputs)
    hidden = keras.layers.Dropout(0.3, seed=54321)(hidden)
    outputs = keras.layers.Dense(1, activation="sigmoid")(hidden)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction="sum_over_batch_size",
        ),
    )
    x = np.arange(32, dtype=np.float32).reshape(8, 4) / 31.0
    y = np.asarray([[0], [1], [0], [1], [1], [0], [1], [0]], dtype=np.float32)
    for _ in range(2):
        model.train_on_batch(x, y)
    digest = hashlib.sha256()
    for weight in model.weights:
        digest.update(np.asarray(weight).tobytes(order="C"))
    return digest.hexdigest()

first_masks = dropout_sequence()
second_masks = dropout_sequence()
print(json.dumps({
    "tensorflow_version": tf.__version__,
    "keras_version": keras.__version__,
    "numpy_version": np.__version__,
    "vocabulary": vocabulary,
    "token_ids": token_ids.numpy().tolist(),
    "vectorizer_layer_dtype": vectorizer.dtype,
    "vectorizer_output_dtype": vectorizer(tf.constant(["alpha"])).dtype.name,
    "standardized_cases": standardized_cases,
    "tokenized_cases": tokenized_cases,
    "bce_probabilities": probabilities.numpy().ravel().tolist(),
    "bce_elementwise": bce_elementwise.numpy().tolist(),
    "bce_reduced": float(bce_reduced.numpy()),
    "bce_has_attached_logits": hasattr(probabilities, "_keras_logits"),
    "dropout_rebuild_equal": all(
        np.array_equal(left, right)
        for left, right in zip(first_masks, second_masks, strict=True)
    ),
    "dropout_calls_differ": not np.array_equal(first_masks[0], first_masks[1]),
    "training_hashes": [training_hash(), training_hash()],
}))
"""
        environment = os.environ.copy()
        environment.update(framework["execution_environment_before_import"])
        environment["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                probe,
                json.dumps(preprocessing),
                json.dumps(self.protocol["model"]),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["tensorflow_version"], framework["tensorflow_version"])
        self.assertEqual(result["keras_version"], framework["keras_version"])
        self.assertEqual(result["numpy_version"], framework["numpy_version"])
        self.assertEqual(
            result["vocabulary"],
            ["", "[UNK]", "beta", "alpha", "stop", "gamma", "cant", "café"],
        )
        self.assertEqual(result["token_ids"], [3, 1, 0, 0, 0])
        self.assertEqual(result["vectorizer_layer_dtype"], preprocessing["layer_dtype"])
        self.assertEqual(
            result["vectorizer_output_dtype"], preprocessing["output_dtype"]
        )
        self.assertEqual(
            result["standardized_cases"],
            [case["standardized"] for case in preprocessing["golden_cases"]],
        )
        self.assertEqual(
            result["tokenized_cases"],
            [case["tokens"] for case in preprocessing["golden_cases"]],
        )
        golden = self.protocol["model"]["loss_golden_probe"]
        self.assertTrue(self.protocol["model"]["loss_logit_recovery_required"])
        self.assertTrue(result["bce_has_attached_logits"])
        self.assertEqual(result["bce_probabilities"], golden["sigmoid_probabilities"])
        np.testing.assert_allclose(
            result["bce_elementwise"], golden["elementwise_losses"], rtol=0, atol=0
        )
        self.assertEqual(result["bce_reduced"], golden["sum_over_batch_size_loss"])
        self.assertTrue(result["dropout_rebuild_equal"])
        self.assertTrue(result["dropout_calls_differ"])
        self.assertEqual(result["training_hashes"][0], result["training_hashes"][1])
        self.assertEqual(
            framework["deterministic_ops_api"],
            "tf.config.experimental.enable_op_determinism()",
        )
        self.assertEqual(
            framework["seed_api"], "keras.utils.set_random_seed(tensorflow_seed)"
        )
        self.assertIn("Dropout(seed=seed)", framework["dropout"]["seed_argument"])
        self.assertIn("never reset", self.protocol["training"]["dropout_rng"])

    def test_seed_derivation_and_rng_namespaces_are_executable_and_exact(self) -> None:
        seeding = self.protocol["seeding"]
        self.assertEqual(seeding["hash_algorithm"], "sha256")
        self.assertEqual(seeding["uint64_byte_slice"], "first_8_digest_bytes")
        self.assertEqual(seeding["uint64_byte_order"], "big")
        self.assertEqual(
            seeding["python_derivation"],
            'int.from_bytes(hashlib.sha256(f"{master_seed}|{namespace}|{attempt}"'
            '.encode("ascii")).digest()[:8], "big", signed=False)',
        )
        self.assertEqual(seeding["default_attempt"], 0)
        self.assertTrue(seeding["namespace_reuse_forbidden"])
        tensorflow_seed_namespace = {"derived_uint64": 2147483647}
        exec(seeding["tensorflow_seed_python"], {}, tensorflow_seed_namespace)
        self.assertEqual(tensorflow_seed_namespace["seed"], 1)
        tensorflow_seed_namespace = {"derived_uint64": 2147483648}
        exec(seeding["tensorflow_seed_python"], {}, tensorflow_seed_namespace)
        self.assertEqual(tensorflow_seed_namespace["seed"], 1)
        self.assertEqual(
            set(seeding["application"]),
            {
                "partitioning",
                "validation_split",
                "model_initialization",
                "training_order",
                "dropout",
                "threat_construction",
                "privacy_sampling",
            },
        )

        self.assertEqual(
            derive_seed(
                self.protocol,
                67,
                "partition/iid/4/round--1/epoch--1/client--1/label-0",
            ),
            7719479962854520267,
        )
        self.assertEqual(
            derive_seed(
                self.protocol,
                67,
                "partition/dirichlet_0.5/4/round--1/epoch--1/client--1/label-1",
                3,
            ),
            12278398448260194388,
        )
        self.assertEqual(
            derive_seed(
                self.protocol,
                401,
                "privacy/membership/CELL/member/round--1/epoch--1/client--1/label-1",
            ),
            10707713437964235329,
        )
        namespace = "partition/iid/4/round--1/epoch--1/client--1/label-0"
        master_seed = 67
        attempt = 0
        self.assertEqual(
            eval(
                seeding["python_derivation"],
                {"hashlib": hashlib},
                {
                    "master_seed": master_seed,
                    "namespace": namespace,
                    "attempt": attempt,
                },
            ),
            derive_seed(self.protocol, master_seed, namespace, attempt),
        )

        namespaces = seeding["namespaces"]
        self.assertEqual(
            namespaces,
            {
                "partition_iid": "partition/iid/{client_scale}/round--1/epoch--1/client--1/label-{label}",
                "partition_dirichlet": "partition/{partition_name}/{client_scale}/round--1/epoch--1/client--1/label-{label}",
                "validation": "validation/{partition_name}/{client_scale}/round--1/epoch--1/client-{client_id}/label-{label}",
                "model_initialization": "model/{cell_id}/round--1/epoch--1/client-{client_id}",
                "training_order": "training/{cell_id}/round-{round_index}/epoch-{epoch_index}/client-{client_id}",
                "dropout": "dropout/{cell_id}/round-{round_index}/epoch--1/client-{client_id}",
                "malicious_order": "malicious-order/dirichlet_0.5/4/round--1/epoch--1/client--1",
                "membership_member": "privacy/membership/{cell_id}/member/round--1/epoch--1/client--1/label-{label}",
                "membership_nonmember": "privacy/membership/{cell_id}/nonmember/round--1/epoch--1/client--1/label-{label}",
                "update_member": "privacy/update-leakage/{cell_id}/member/round-{round_index}/epoch--1/client-{client_id}/label-{label}",
                "update_nonmember": "privacy/update-leakage/{cell_id}/nonmember/round-{round_index}/epoch--1/client-{client_id}/label-{label}",
            },
        )
        first_generator = generator(self.protocol, 67, namespace)
        second_generator = generator(self.protocol, 67, namespace)
        self.assertIsNot(first_generator, second_generator)
        self.assertTrue(
            np.array_equal(
                first_generator.integers(0, 2**32, 8),
                second_generator.integers(0, 2**32, 8),
            )
        )
        label_one_namespace = namespace.replace("label-0", "label-1")
        self.assertNotEqual(
            derive_seed(self.protocol, 67, namespace),
            derive_seed(self.protocol, 67, label_one_namespace),
        )
        self.assertIn("Never share or reuse", seeding["generator_lifecycle"])
        for key in (
            "partition_iid",
            "partition_dirichlet",
            "validation",
            "membership_member",
            "membership_nonmember",
            "update_member",
            "update_nonmember",
        ):
            self.assertIn("label-{label}", namespaces[key])

    def test_cell_ids_and_zero_based_seed_namespaces_are_canonical(self) -> None:
        cell_id_spec = self.protocol["seeding"]["cell_id"]
        expected_fields = [
            "matrix_kind",
            "strategy",
            "partition",
            "alpha",
            "client_scale",
            "seed",
            "threat",
        ]
        self.assertEqual(cell_id_spec["fields"], expected_fields)
        self.assertEqual(cell_id_spec["serialization"], "compact_sorted_json")
        self.assertTrue(cell_id_spec["serialization_is_identifier"])

        cells = registered_cells(self.protocol)
        identifiers = [canonical_cell_id(registered_cell) for registered_cell in cells]
        self.assertEqual(len(cells), self.protocol["matrix"]["totals"]["maximum_cells"])
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for registered_cell, identifier in zip(cells, identifiers, strict=True):
            self.assertEqual(list(json.loads(identifier)), sorted(expected_fields))
            self.assertNotIn(" ", identifier)
            self.assertEqual(json.loads(identifier), registered_cell)
            self.assertEqual(
                eval(
                    cell_id_spec["python_serialization"],
                    {"json": json},
                    registered_cell,
                ),
                identifier,
            )

        example = cell(
            "robustness",
            "fedavg",
            "dirichlet_0.5",
            4,
            67,
            {"attack": "outlier", "malicious_fraction": 0.25},
        )
        self.assertEqual(canonical_cell_id(example), cell_id_spec["canonical_example"])

        indexing = self.protocol["indexing"]
        self.assertTrue(indexing["all_indices_zero_based"])
        self.assertEqual(indexing["non_applicable_seed_namespace_index"], -1)
        self.assertEqual(
            indexing["federated_round_indices"], "integers from 0 through 19"
        )
        self.assertEqual(indexing["federated_epoch_indices_per_round"], [0])
        self.assertEqual(
            indexing["centralized_epoch_indices"], "integers from 0 through 19"
        )
        self.assertEqual(
            indexing["local_only_epoch_indices"], "integers from 0 through 19"
        )
        self.assertEqual(indexing["label_indices"], [0, 1])
        self.assertEqual(
            indexing["namespace_contexts"],
            {
                "federated_training": {
                    "round_index": "0_through_19",
                    "epoch_index": 0,
                    "client_id": "0_through_client_scale_minus_1",
                },
                "centralized_training": {
                    "round_index": -1,
                    "epoch_index": "0_through_19",
                    "client_id": -1,
                },
                "local_only_training": {
                    "round_index": -1,
                    "epoch_index": "0_through_19",
                    "client_id": "0_through_3",
                },
                "global_model_initialization": {
                    "round_index": -1,
                    "epoch_index": -1,
                    "client_id": -1,
                },
                "local_only_model_initialization": {
                    "round_index": -1,
                    "epoch_index": -1,
                    "client_id": "0_through_3",
                },
            },
        )
        namespaces = self.protocol["seeding"]["namespaces"]
        for namespace in namespaces.values():
            self.assertIn("round-", namespace)
            self.assertIn("epoch-", namespace)
            self.assertIn("client-", namespace)
        for key in (
            "partition_iid",
            "partition_dirichlet",
            "validation",
            "malicious_order",
            "membership_member",
            "membership_nonmember",
        ):
            self.assertIn("--1", namespaces[key])
        identifier = canonical_cell_id(
            cell("local_only", "local_only", "iid_stratified", 4, 67)
        )
        self.assertEqual(
            namespaces["model_initialization"].format(cell_id=identifier, client_id=2),
            f"model/{identifier}/round--1/epoch--1/client-2",
        )

    def test_offline_partitions_retries_validation_and_clients_cover_cells(
        self,
    ) -> None:
        results = self.partition_results()
        seeds = self.protocol["seeding"]["seeds"]
        expected_attempts = {
            ("iid_stratified", 4): [0, 0, 0, 0, 0],
            ("iid_stratified", 16): [0, 0, 0, 0, 0],
            ("iid_stratified", 64): [0, 0, 0, 0, 0],
            ("dirichlet_0.5", 4): [0, 0, 0, 0, 0],
            ("dirichlet_0.1", 4): [0, 1, 4, 3, 1],
            ("dirichlet_1.0", 16): [0, 0, 0, 0, 0],
            ("dirichlet_0.5", 16): [3, 0, 0, 2, 0],
            ("dirichlet_0.1", 16): [4664, 167, 1148, 41, 547],
            ("dirichlet_1.0", 64): [1, 0, 1, 0, 0],
            ("dirichlet_0.5", 64): [66, 209, 172, 113, 316],
        }
        actual_attempts = {
            (partition_name, client_scale): [
                None
                if results[(seed, partition_name, client_scale)] is None
                else results[(seed, partition_name, client_scale)][0]
                for seed in seeds
            ]
            for partition_name, client_scale in expected_attempts
        }
        self.assertEqual(actual_attempts, expected_attempts)

        registered = registered_cells(self.protocol)
        for registered_cell in registered:
            client_scale = registered_cell["client_scale"]
            self.assertEqual(
                list(range(client_scale or 0)),
                sorted(range(client_scale or 0)),
            )
            partition_name = registered_cell["partition"]
            if partition_name is None:
                continue
            result = results[
                (
                    registered_cell["seed"],
                    partition_name,
                    client_scale,
                )
            ]
            self.assertIsNotNone(result)
            attempt, shards = result
            self.assertGreaterEqual(
                attempt, self.protocol["partitioning"]["first_attempt"]
            )
            self.assertLessEqual(attempt, self.protocol["partitioning"]["last_attempt"])
            self.assertEqual(len(shards), client_scale)
            combined = np.concatenate(shards)
            self.assertEqual(combined.size, self.labels.size)
            self.assertTrue(
                np.array_equal(np.sort(combined), np.arange(self.labels.size))
            )
            self.assertTrue(all(np.all(shard[:-1] <= shard[1:]) for shard in shards))

        for (seed, partition_name, client_scale), result in results.items():
            if result is None:
                continue
            _, shards = result
            fitted, validation = validation_split(
                self.protocol,
                self.labels,
                seed,
                partition_name,
                client_scale,
                shards,
            )
            for shard, fitted_rows, validation_rows in zip(
                shards, fitted, validation, strict=True
            ):
                self.assertEqual(
                    np.intersect1d(fitted_rows, validation_rows).size,
                    0,
                )
                self.assertTrue(
                    np.array_equal(
                        np.sort(np.concatenate((fitted_rows, validation_rows))),
                        shard,
                    )
                )
                for label in self.protocol["partitioning"]["labels"]:
                    shard_count = np.count_nonzero(self.labels[shard] == label)
                    validation_count = np.count_nonzero(
                        self.labels[validation_rows] == label
                    )
                    self.assertEqual(
                        validation_count,
                        math.floor(
                            shard_count
                            * self.protocol["training"]["validation_fraction"]
                        ),
                    )

        iid_result = results[(67, "iid_stratified", 64)]
        self.assertIsNotNone(iid_result)
        for label in self.protocol["partitioning"]["labels"]:
            label_counts = [
                np.count_nonzero(self.labels[shard] == label) for shard in iid_result[1]
            ]
            self.assertLessEqual(max(label_counts) - min(label_counts), 1)

    def test_offline_update_malicious_and_privacy_selection_cover_cells(
        self,
    ) -> None:
        results = self.partition_results()
        seeds = self.protocol["seeding"]["seeds"]
        update = self.protocol["privacy"]["update_leakage"]
        expected_selected_clients = [0, 0, 1, 0, 3]
        selected_clients: dict[int, int] = {}
        fitted_by_seed: dict[int, list[np.ndarray]] = {}
        for seed, expected_client in zip(seeds, expected_selected_clients, strict=True):
            result = results[(seed, "dirichlet_0.5", 4)]
            self.assertIsNotNone(result)
            _, shards = result
            fitted, _ = validation_split(
                self.protocol,
                self.labels,
                seed,
                "dirichlet_0.5",
                4,
                shards,
            )
            selected_client = select_update_client(self.protocol, self.labels, fitted)
            selected_clients[seed] = selected_client
            fitted_by_seed[seed] = fitted
            self.assertEqual(selected_client, expected_client)
            required = update["records_per_class_per_target_label"]
            for label in self.protocol["partitioning"]["labels"]:
                self.assertGreaterEqual(
                    np.count_nonzero(self.labels[fitted[selected_client]] == label),
                    required,
                )
            for earlier_client in range(selected_client):
                self.assertTrue(
                    any(
                        np.count_nonzero(self.labels[fitted[earlier_client]] == label)
                        < required
                        for label in self.protocol["partitioning"]["labels"]
                    )
                )
        with self.assertRaisesRegex(ValueError, "no update-leakage client"):
            select_update_client(
                self.protocol,
                self.labels,
                [np.arange(499), np.arange(12500, 12999)],
            )

        robustness_cells = [
            registered_cell
            for registered_cell in registered_cells(self.protocol)
            if registered_cell["matrix_kind"] == "robustness"
        ]
        for seed in seeds:
            one = malicious_clients(self.protocol, seed, 1)
            two = malicious_clients(self.protocol, seed, 2)
            self.assertEqual(len(one), 1)
            self.assertEqual(len(two), 2)
            self.assertLessEqual(set(one), set(two))
            for registered_cell in robustness_cells:
                if registered_cell["seed"] != seed:
                    continue
                count = round(
                    registered_cell["threat"]["malicious_fraction"]
                    * registered_cell["client_scale"]
                )
                self.assertEqual(
                    malicious_clients(self.protocol, seed, count),
                    one if count == 1 else two,
                )

        privacy_cells = [
            registered_cell
            for registered_cell in registered_cells(self.protocol)
            if registered_cell["matrix_kind"]
            in {"membership_inference", "update_leakage"}
        ]
        namespaces = self.protocol["seeding"]["namespaces"]
        all_rows = np.arange(self.labels.size)
        for registered_cell in privacy_cells:
            seed = registered_cell["seed"]
            identifier = canonical_cell_id(registered_cell)
            if registered_cell["matrix_kind"] == "membership_inference":
                partition_name = registered_cell["partition"] or "iid_stratified"
                client_scale = registered_cell["client_scale"] or 4
                result = results[(seed, partition_name, client_scale)]
                self.assertIsNotNone(result)
                _, shards = result
                fitted, _ = validation_split(
                    self.protocol,
                    self.labels,
                    seed,
                    partition_name,
                    client_scale,
                    shards,
                )
                member_rows = np.sort(np.concatenate(fitted))
                count = self.protocol["privacy"]["membership_inference"][
                    "records_per_class_per_membership"
                ]
                member_namespace = namespaces["membership_member"].replace(
                    "{cell_id}", identifier
                )
                nonmember_namespace = namespaces["membership_nonmember"].replace(
                    "{cell_id}", identifier
                )
                member_sample = privacy_sample(
                    self.protocol,
                    self.labels,
                    seed,
                    member_namespace,
                    member_rows,
                    count,
                )
                nonmember_sample = privacy_sample(
                    self.protocol,
                    self.labels,
                    seed,
                    nonmember_namespace,
                    all_rows,
                    count,
                )
            else:
                fitted = fitted_by_seed[seed]
                selected_client = selected_clients[seed]
                count = update["records_per_class_per_target_label"]
                replacements = {
                    "{cell_id}": identifier,
                    "{round_index}": str(update["target_round_index"]),
                    "{client_id}": str(selected_client),
                }
                member_namespace = namespaces["update_member"]
                nonmember_namespace = namespaces["update_nonmember"]
                for token, value in replacements.items():
                    member_namespace = member_namespace.replace(token, value)
                    nonmember_namespace = nonmember_namespace.replace(token, value)
                member_sample = privacy_sample(
                    self.protocol,
                    self.labels,
                    seed,
                    member_namespace,
                    fitted[selected_client],
                    count,
                )
                nonmember_sample = privacy_sample(
                    self.protocol,
                    self.labels,
                    seed,
                    nonmember_namespace,
                    np.sort(
                        np.concatenate(
                            [
                                rows
                                for client_id, rows in enumerate(fitted)
                                if client_id != selected_client
                            ]
                        )
                    ),
                    count,
                )
            for sample in (member_sample, nonmember_sample):
                self.assertEqual(sample.size, 2 * count)
                self.assertEqual(np.unique(sample).size, sample.size)
                self.assertTrue(np.all(self.labels[sample[:count]] == 0))
                self.assertTrue(np.all(self.labels[sample[count:]] == 1))

        with self.assertRaisesRegex(ValueError, "499 eligible rows"):
            privacy_sample(
                self.protocol,
                self.labels,
                67,
                "privacy/test/label-{label}",
                np.arange(499),
                500,
            )

    def test_excluded_dirichlet_point_one_64_exhausts_every_seed(self) -> None:
        scale = self.protocol["matrix"]["scale"]
        partitioning = self.protocol["partitioning"]
        self.assertEqual(scale["excluded_partition"], "dirichlet_0.1")
        self.assertEqual(scale["excluded_client_scale"], 64)
        self.assertEqual(scale["excluded_seeds"], self.protocol["seeding"]["seeds"])
        self.assertEqual(
            scale["exclusion_attempt_range"],
            [partitioning["first_attempt"], partitioning["last_attempt"]],
        )
        protocol_generator = generator
        for seed in scale["excluded_seeds"]:
            attempts: list[int] = []

            def tracked_generator(
                protocol: dict[str, Any],
                master_seed: int,
                namespace: str,
                attempt: int = 0,
            ) -> np.random.Generator:
                """Record and execute one partition generator construction.

                Args:
                    protocol: Parsed scientific protocol.
                    master_seed: Registered experiment seed.
                    namespace: Fully expanded RNG namespace.
                    attempt: Zero-based retry attempt.

                Returns:
                    A fresh NumPy generator for the registered seed material.
                """
                attempts.append(attempt)
                return protocol_generator(protocol, master_seed, namespace, attempt)

            with (
                self.subTest(seed=seed),
                patch(f"{__name__}.generator", side_effect=tracked_generator),
                self.assertRaisesRegex(
                    ValueError,
                    rf"no accepted dirichlet_0\.1/64 partition for seed {seed}",
                ),
            ):
                dirichlet_partition(
                    self.protocol,
                    self.labels,
                    seed,
                    scale["excluded_partition"],
                    0.1,
                    scale["excluded_client_scale"],
                )
            np.testing.assert_array_equal(
                np.bincount(attempts, minlength=10000), np.full(10000, 2)
            )

    def test_partition_and_client_selection_algorithms_are_closed(self) -> None:
        partitioning = self.protocol["partitioning"]
        self.assertEqual(
            partitioning["row_identity"], "zero_based_official_split_row_index"
        )
        self.assertEqual(partitioning["labels"], [0, 1])
        self.assertIn("numpy.array_split", partitioning["iid"]["algorithm"])
        self.assertIn("ascending client ID", partitioning["iid"]["algorithm"])
        self.assertEqual(partitioning["iid"]["attempt"], 0)
        self.assertIn("generator.dirichlet", partitioning["dirichlet"]["algorithm"])
        self.assertIn("generator.multinomial", partitioning["dirichlet"]["algorithm"])
        self.assertEqual(
            partitioning["dirichlet"]["retry_scope"],
            "discard_every_client_allocation_and_restart_with_next_attempt",
        )
        self.assertIn(
            "floor(label_count_times_0.2)", partitioning["validation"]["algorithm"]
        )
        self.assertEqual(
            partitioning["validation"]["fitted_rows_use"], "optimizer_input_only"
        )

        clients = self.protocol["selection"]["clients"]
        self.assertFalse(clients["random_sampling"])
        self.assertEqual(
            clients["fit_participation"], "all_registered_clients_every_round"
        )
        self.assertEqual(clients["evaluation_participation"], "none")
        self.assertIn("server evaluates", clients["evaluation_execution"])
        self.assertIn("no evaluate request", clients["evaluation_execution"])
        self.assertEqual(clients["order"], "ascending_numeric_client_id")
        malicious = self.protocol["selection"]["malicious"]
        self.assertIn("first k IDs", malicious["algorithm"])
        self.assertIn("one-client set is a subset", malicious["pairing"])

    def test_baseline_exposure_and_local_reporting_are_exact(self) -> None:
        training = self.protocol["training"]
        centralized = self.protocol["strategies"]["centralized"]
        local = self.protocol["strategies"]["local_only"]
        self.assertEqual(
            (
                training["rounds"] * training["local_epochs"],
                training["federated_fitted_row_exposures"],
                centralized["training_epochs"],
                centralized["fitted_row_exposures"],
                local["training_epochs_per_model"],
                local["fitted_row_exposures"],
            ),
            (20, 20, 20, 20, 20, 20),
        )
        self.assertEqual(local["models_per_cell"], 4)
        self.assertEqual(local["client_scale"], 4)
        self.assertEqual(local["model_aggregation"], "none")
        self.assertEqual(local["prediction_aggregation"], "none")
        self.assertIn("unweighted arithmetic mean", local["reporting"])
        self.assertIn("exactly four models", local["reporting"])
        self.assertIn("iid_stratified_at_4_clients", centralized["split_reference"])
        self.assertIn("epoch indices 0 through 19", centralized["validation_reporting"])
        self.assertIn("epoch indices 0 through 19", local["validation_reporting"])
        self.assertEqual(
            centralized["convergence_round_reporting"],
            "not_applicable_and_must_not_be_reported",
        )
        self.assertEqual(
            local["convergence_round_reporting"],
            "not_applicable_and_must_not_be_reported",
        )
        self.assertEqual(
            local["models_per_cell"],
            self.protocol["matrix"]["local_only"]["training_invocations_per_cell"],
        )

    def test_strategies_and_threats_are_fully_registered(self) -> None:
        strategies = self.protocol["strategies"]
        self.assertEqual(
            set(strategies),
            {
                "centralized",
                "local_only",
                "fedavg",
                "fedprox",
                "fedprox_huber",
                "fedmedian",
                "fedtrimmedavg",
            },
        )
        fedprox = strategies["fedprox"]
        self.assertEqual(fedprox["mu"], 0.1)
        self.assertIn("keras.ops.convert_to_tensor", fedprox["reference"])
        self.assertIn("weight.numpy().copy()", fedprox["reference"])
        self.assertIn("sum_over_batch_size", fedprox["data_loss"])
        self.assertIn("keras.ops.sum", fedprox["proximal_term"])
        self.assertIn("left-fold", fedprox["proximal_term"])
        self.assertIn(
            "not divided or multiplied by batch size", fedprox["local_objective"]
        )
        self.assertIn("fitted training rows", fedprox["aggregation_sample_weight"])
        self.assertIn("ascending client_id order", fedprox["aggregation"])
        fedprox_golden = fedprox["golden_probe"]
        data_loss = float(np.mean(fedprox_golden["batch_example_losses"]))
        squared_distance = sum(
            np.sum(
                (
                    np.asarray(current, dtype=np.float32)
                    - np.asarray(reference, dtype=np.float32)
                )
                ** 2
            )
            for current, reference in zip(
                fedprox_golden["current_trainable_variables"],
                fedprox_golden["reference_trainable_variables"],
                strict=True,
            )
        )
        proximal_term = fedprox["mu"] / 2.0 * squared_distance
        self.assertEqual(data_loss, fedprox_golden["data_loss"])
        self.assertEqual(proximal_term, fedprox_golden["proximal_term"])
        self.assertEqual(data_loss + proximal_term, fedprox_golden["local_objective"])

        huber = strategies["fedprox_huber"]
        self.assertEqual(
            (
                huber["threshold"],
                huber["iterations"],
                huber["epsilon"],
            ),
            (10.0, 10, 1e-8),
        )
        self.assertIn("model.weights order", huber["client_vector"])
        self.assertIn("float32", huber["client_vector"])
        self.assertIn("positive integer", huber["sample_weight"])
        self.assertIn("divide", huber["sample_weight"])
        self.assertIn("numpy.linalg.norm", huber["residual_norm"])
        self.assertIn("residual_i+epsilon", huber["epsilon_rule"])
        self.assertIn("residual_i <= threshold", huber["epsilon_rule"])
        self.assertIn("exactly 10 updates", huber["update"])
        huber_golden = huber["golden_probe"]
        initial, aggregate = iterative_huber_aggregate(
            np.asarray(huber_golden["client_vectors"]),
            np.asarray(huber_golden["sample_weights"]),
            huber["threshold"],
            huber["epsilon"],
            huber["iterations"],
        )
        np.testing.assert_allclose(
            initial, huber_golden["initial_estimate"], rtol=0, atol=0
        )
        np.testing.assert_allclose(
            aggregate,
            huber_golden["estimate_after_10_updates"],
            rtol=0,
            atol=1e-15,
        )

        beta = strategies["fedtrimmedavg"]["beta"]
        self.assertEqual(beta, 0.25)
        for clients in self.protocol["partitioning"]["client_scales"]:
            trimmed_per_side = math.floor(beta * clients)
            self.assertGreater(clients - 2 * trimmed_per_side, 0)

        threats = self.protocol["threats"]
        self.assertEqual(threats["malicious_fractions"], [0.25, 0.5])
        self.assertEqual(threats["malicious_client_counts"], [1, 2])
        self.assertEqual(threats["outlier"]["scale"], 10.0)
        self.assertEqual(threats["sign_flip"]["scale"], -10.0)

    def test_metrics_and_statistics_are_exact(self) -> None:
        metrics = self.protocol["metrics"]
        self.assertEqual(
            metrics["classification"]["confusion_matrix_order"],
            [["TN", "FP"], ["FN", "TP"]],
        )
        self.assertEqual(
            metrics["classification"]["reported"],
            ["accuracy", "precision", "recall", "f1", "roc_auc"],
        )
        self.assertEqual(metrics["classification"]["zero_division"], 0)
        self.assertTrue(metrics["classification"]["raw_predictions"])
        self.assertEqual(
            metrics["system"]["communication_volume"],
            "exact_serialized_application_bytes",
        )

        statistics = self.protocol["statistics"]
        self.assertEqual(statistics["replicates"], 5)
        self.assertTrue(statistics["sample_variance"])
        self.assertTrue(statistics["sample_standard_deviation"])
        self.assertEqual(statistics["variance_ddof"], 1)
        self.assertEqual(statistics["confidence_interval"], 0.95)
        self.assertEqual(statistics["t_critical"], 2.776)
        self.assertTrue(statistics["paired_deltas"])
        self.assertEqual(statistics["paired_delta_dtype"], "float64")
        self.assertEqual(
            statistics["paired_delta_operation"], "candidate_minus_baseline"
        )
        self.assertIn("candidate_value", statistics["paired_delta_formula"])
        self.assertIn("baseline_value", statistics["paired_delta_formula"])
        self.assertIn("never reverse", statistics["paired_delta_formula"])

    def test_paired_baseline_pair_sets_are_exact_and_registered(self) -> None:
        matrix = self.protocol["matrix"]
        contrasts = {
            contrast["name"]: contrast
            for contrast in self.protocol["statistics"]["paired_contrasts"]
        }
        expected_by_matrix = {
            "primary_federated": [
                "primary_strategy_vs_fedavg",
                "primary_partition_vs_iid",
            ],
            "centralized": [],
            "local_only": ["local_partition_vs_iid"],
            "scale": ["scale_vs_4_clients"],
            "robustness": ["robustness_vs_no_attack"],
            "membership_inference": ["membership_fedavg_vs_centralized"],
            "update_leakage": [],
        }
        for matrix_kind, names in expected_by_matrix.items():
            self.assertEqual(matrix[matrix_kind]["paired_contrasts"], names)
            if not names:
                self.assertEqual(
                    matrix[matrix_kind]["paired_delta_policy"],
                    "none_because_no_registered_factor_has_multiple_levels",
                )
        self.assertEqual(
            set(contrasts), set(itertools.chain(*expected_by_matrix.values()))
        )

        expected_pair_counts = {
            "primary_strategy_vs_fedavg": 80,
            "primary_partition_vs_iid": 75,
            "local_partition_vs_iid": 15,
            "scale_vs_4_clients": 70,
            "robustness_vs_no_attack": 80,
            "membership_fedavg_vs_centralized": 5,
        }
        for name, expected_count in expected_pair_counts.items():
            contrast = contrasts[name]
            pairs = paired_contrast_pairs(self.protocol, contrast)
            self.assertEqual(len(pairs), expected_count)
            self.assertTrue(contrast["metrics"])
            for candidate, baseline in pairs:
                self.assertEqual(candidate["seed"], baseline["seed"])
                candidate_value = float(candidate["seed"] + 1)
                baseline_value = float(baseline["seed"] - 1)
                self.assertEqual(candidate_value - baseline_value, 2.0)

        strategy_pair = paired_contrast_pairs(
            self.protocol, contrasts["primary_strategy_vs_fedavg"]
        )[0]
        self.assertEqual(strategy_pair[1]["strategy"], "fedavg")
        partition_pair = paired_contrast_pairs(
            self.protocol, contrasts["primary_partition_vs_iid"]
        )[0]
        self.assertEqual(partition_pair[1]["partition"], "iid_stratified")
        self.assertIsNone(partition_pair[1]["alpha"])
        scale_pair = paired_contrast_pairs(
            self.protocol, contrasts["scale_vs_4_clients"]
        )[0]
        self.assertEqual(
            (scale_pair[1]["matrix_kind"], scale_pair[1]["client_scale"]),
            ("primary_federated", 4),
        )
        robustness_pair = paired_contrast_pairs(
            self.protocol, contrasts["robustness_vs_no_attack"]
        )[0]
        self.assertEqual(robustness_pair[1]["matrix_kind"], "primary_federated")
        self.assertIsNone(robustness_pair[1]["threat"])
        membership_pair = paired_contrast_pairs(
            self.protocol, contrasts["membership_fedavg_vs_centralized"]
        )[0]
        self.assertEqual(membership_pair[1]["strategy"], "centralized")
        self.assertIsNone(membership_pair[1]["partition"])

    def test_privacy_analyses_have_exact_candidates_labels_and_scores(self) -> None:
        privacy = self.protocol["privacy"]
        membership = privacy["membership_inference"]
        self.assertEqual(
            membership["member_source"],
            "fitted_training_rows_actually_supplied_to_the_model_optimizer",
        )
        self.assertEqual(membership["nonmember_source"], "official_test_split")
        self.assertEqual(
            (
                membership["member_records"],
                membership["nonmember_records"],
                membership["records_per_class_per_membership"],
            ),
            (1000, 1000, 500),
        )
        self.assertFalse(membership["sampling_replacement"])
        self.assertEqual(
            (
                membership["membership_positive_label"],
                membership["membership_negative_label"],
            ),
            (1, 0),
        )
        self.assertEqual(
            (
                membership["prediction_clip_minimum"],
                membership["prediction_clip_maximum"],
            ),
            (1e-7, 0.9999999),
        )
        self.assertIn(
            "negative_per_example_binary_crossentropy", membership["attack_score"]
        )
        self.assertIn("first 500", membership["sampling_algorithm"])

        leakage = privacy["update_leakage"]
        self.assertEqual(leakage["target_round_index"], 0)
        self.assertIn("lowest numeric zero-based client_id", leakage["selected_client"])
        self.assertIn("hard-fail", leakage["selected_client"])
        self.assertIn(
            "every registered master seed", leakage["selected_client_feasibility"]
        )
        self.assertEqual(
            leakage["member_source"], "selected_client fitted training rows"
        )
        self.assertEqual(
            leakage["nonmember_source"],
            "union of all non-selected clients fitted training rows from the same accepted partition",
        )
        self.assertEqual(
            (
                leakage["member_records"],
                leakage["nonmember_records"],
                leakage["records_per_class_per_target_label"],
            ),
            (1000, 1000, 500),
        )
        self.assertFalse(leakage["sampling_replacement"])
        self.assertEqual(
            (
                leakage["target_membership_positive_label"],
                leakage["target_membership_negative_label"],
            ),
            (1, 0),
        )
        self.assertEqual(leakage["gradient_model"], "pre_update_model")
        self.assertIn("binary_crossentropy", leakage["gradient_loss"])
        self.assertIn("negative_gradient", leakage["gradient_vector"])
        self.assertIn("product_of_l2_norms", leakage["score"])
        self.assertEqual(leakage["zero_norm_score"], 0.0)
        self.assertIn("larger_means_more_likely", leakage["score_direction"])

    def test_privacy_bce_and_roc_golden_vectors_are_exact(self) -> None:
        bce = self.protocol["privacy"]["binary_crossentropy"]
        self.assertEqual(bce["dtype"], "float64")
        self.assertEqual(bce["reduction"], "none")
        for vector in bce["golden_vectors"]:
            actual = privacy_binary_crossentropy(
                np.asarray(vector["labels"]),
                np.asarray(vector["probabilities"]),
                bce["clip_minimum"],
                bce["clip_maximum"],
            )
            np.testing.assert_allclose(
                actual,
                vector["losses"],
                rtol=0.0,
                atol=1e-15,
            )

        metric = self.protocol["privacy"]["metrics"]
        self.assertEqual(metric["target_fpr"], 0.01)
        self.assertIn("score >= t", metric["thresholds"])
        self.assertIn("all equal-score records atomically", metric["thresholds"])
        self.assertIn("numpy.trapezoid", metric["roc_auc"])
        self.assertIn("numpy.interp", metric["tpr_at_1_percent_fpr"])
        for vector in metric["golden_vectors"]:
            actual = privacy_roc(
                np.asarray(vector["labels"]),
                np.asarray(vector["scores"]),
                metric["target_fpr"],
            )
            expected_thresholds = np.asarray(
                [
                    np.inf if threshold == "inf" else float(threshold)
                    for threshold in vector["thresholds"]
                ]
            )
            np.testing.assert_array_equal(actual["thresholds"], expected_thresholds)
            np.testing.assert_allclose(actual["fpr"], vector["fpr"], rtol=0, atol=0)
            np.testing.assert_allclose(actual["tpr"], vector["tpr"], rtol=0, atol=0)
            self.assertAlmostEqual(actual["roc_auc"], vector["roc_auc"], places=15)
            self.assertAlmostEqual(
                actual["max_tpr_minus_fpr"],
                vector["max_tpr_minus_fpr"],
                places=15,
            )
            expected_max_threshold = (
                np.inf if vector["max_threshold"] == "inf" else vector["max_threshold"]
            )
            self.assertEqual(actual["max_threshold"], expected_max_threshold)
            self.assertAlmostEqual(
                actual["tpr_at_target_fpr"],
                vector["tpr_at_1_percent_fpr"],
                places=15,
            )

        with self.assertRaisesRegex(ValueError, "both membership classes"):
            privacy_roc(np.asarray([1, 1]), np.asarray([0.2, 0.1]), 0.01)

    def test_system_measurement_boundaries_and_retry_rules_are_exact(self) -> None:
        system = self.protocol["metrics"]["system"]
        self.assertIn("registered validation accuracy", system["convergence"])
        self.assertIn("round indices 0 through 19", system["convergence"])
        self.assertFalse(system["pre_training_state_included_for_convergence"])
        self.assertEqual(
            system["convergence_scope"],
            "federated matrices only: primary_federated, scale, and robustness",
        )
        self.assertEqual(
            system["reported_federated"],
            ["convergence_round", "communication_bytes", "client_training_time_ns"],
        )
        self.assertIn(
            "never report convergence_round", system["centralized_local_only_curves"]
        )
        self.assertEqual(
            system["reported_centralized_local_only"],
            ["epoch_validation_curve", "training_time_ns"],
        )
        self.assertEqual(
            system["convergence_evaluation_data"],
            "union_of_registered_train_split_validation_rows",
        )
        self.assertIn("final_trained_model_only", system["test_evaluation_schedule"])
        self.assertIn("never_for_convergence", system["test_evaluation_schedule"])
        self.assertEqual(system["client_fit_timer"], "time.perf_counter_ns")
        self.assertIn("immediately_before", system["client_fit_timer_start"])
        self.assertIn("immediately_after", system["client_fit_timer_stop"])
        self.assertIn("every attempt duration", system["client_fit_timer_retry_rule"])
        self.assertEqual(system["client_fit_timer_report_unit"], "nanoseconds")
        self.assertIn(
            "sum client fit attempt durations", system["federated_round_reduction"]
        )
        self.assertIn(
            "Sum the 20 federated round totals", system["federated_run_reduction"]
        )
        self.assertIn(
            "sum of the four model totals", system["local_only_model_reduction"]
        )
        self.assertIn("client_fit_timer_start", system["local_only_timer_boundaries"])
        self.assertIn("single model.fit call", system["centralized_timer_start"])
        self.assertIn(
            "every initial and retried", system["centralized_retry_reduction"]
        )
        self.assertIn("Never sum timing across seeds", system["statistics_reduction"])
        self.assertEqual(system["communication_flower_version"], "1.32.1")
        self.assertEqual(
            system["communication_directions"],
            ["server_to_client", "client_to_server"],
        )
        self.assertEqual(
            system["communication_counting_point"], "sender_once_per_message"
        )
        self.assertIn(
            "immediately before transport framing",
            system["communication_serialization_boundary"],
        )
        self.assertIn(
            "every initial attempt and retry", system["communication_retry_rule"]
        )
        self.assertIn("never double-count", system["communication_retry_rule"])
        self.assertEqual(
            system["communication_included_messages"],
            ["fit_request", "fit_response"],
        )
        self.assertIn("none", system["communication_evaluation_messages"])
        self.assertIn("transport_framing", system["communication_excluded_bytes"])
        self.assertEqual(
            system["communication_reporting"],
            "report server_to_client_bytes client_to_server_bytes and their exact sum per run",
        )

    def test_matrix_arithmetic_is_exact(self) -> None:
        matrix = self.protocol["matrix"]
        primary = matrix["primary_federated"]
        centralized = matrix["centralized"]
        local = matrix["local_only"]
        scale = matrix["scale"]
        robustness = matrix["robustness"]
        membership = matrix["membership_inference"]
        leakage = matrix["update_leakage"]

        self.assertEqual(
            primary["cells"],
            len(primary["strategies"]) * len(primary["partitions"]) * primary["seeds"],
        )
        self.assertEqual(
            centralized["cells"], len(centralized["strategies"]) * centralized["seeds"]
        )
        self.assertEqual(
            local["cells"],
            len(local["strategies"]) * len(local["partitions"]) * local["seeds"],
        )
        self.assertEqual(
            local["training_invocations"],
            local["cells"] * local["training_invocations_per_cell"],
        )
        self.assertEqual(
            scale["cells"],
            len(scale["strategies"])
            * scale["seeds"]
            * sum(
                len(scale["partitions_by_client_scale"][str(client_scale)])
                for client_scale in scale["client_scales"]
            ),
        )
        self.assertEqual(
            scale["partitions_by_client_scale"],
            {
                "16": [
                    "iid_stratified",
                    "dirichlet_1.0",
                    "dirichlet_0.5",
                    "dirichlet_0.1",
                ],
                "64": ["iid_stratified", "dirichlet_1.0", "dirichlet_0.5"],
            },
        )
        self.assertIn("not registered", scale["excluded_cells"])
        self.assertNotIn("dirichlet_0.1", scale["partitions_by_client_scale"]["64"])
        self.assertEqual(
            robustness["cells"],
            len(robustness["strategies"])
            * len(robustness["attacks"])
            * len(robustness["malicious_fractions"])
            * robustness["seeds"],
        )
        self.assertEqual(
            membership["cells"], len(membership["models"]) * membership["seeds"]
        )
        self.assertEqual(leakage["cells"], len(leakage["models"]) * leakage["seeds"])

        cells = sum(
            section["cells"] for name, section in matrix.items() if name != "totals"
        )
        self.assertEqual(cells, matrix["totals"]["maximum_cells"])
        analysis_only_cells = membership["cells"] + leakage["cells"]
        invocations = (
            cells - analysis_only_cells - local["cells"] + local["training_invocations"]
        )
        totals = matrix["totals"]
        self.assertEqual(invocations, totals["planned_training_invocations"])
        self.assertEqual(analysis_only_cells, totals["analysis_only_cells"])
        self.assertEqual(
            totals["training_invocation_retry_reserve"],
            totals["maximum_training_invocations"] - invocations,
        )
        self.assertEqual(
            (
                cells,
                invocations,
                totals["maximum_training_invocations"],
                totals["training_invocation_retry_reserve"],
            ),
            (290, 335, 350, 15),
        )
        self.assertTrue(matrix["totals"]["privacy_analyses_reuse_trained_models"])
        self.assertIn(
            "matrix.centralized_same_seed",
            membership["centralized_model_source"],
        )
        self.assertIn("same_seed_final_model", membership["fedavg_model_source"])
        self.assertIn(
            "round_index_0_selected_client_update", leakage["fedavg_model_source"]
        )

    def test_compute_and_abort_budget_is_exact(self) -> None:
        budget = self.protocol["budget"]
        self.assertEqual(
            (
                budget["maximum_cells"],
                budget["planned_training_invocations"],
                budget["maximum_training_invocations"],
                budget["training_invocation_retry_reserve"],
            ),
            (290, 335, 350, 15),
        )
        self.assertEqual(
            (budget["maximum_gpu_hours"], budget["maximum_cpu_hours"]), (500, 5000)
        )
        self.assertEqual(
            (
                budget["maximum_memory_gib_per_job"],
                budget["maximum_vram_gib_per_job"],
                budget["maximum_storage_gib"],
            ),
            (32, 12, 50),
        )
        self.assertEqual(
            (
                budget["timeout_hours_4_clients"],
                budget["timeout_hours_16_clients"],
                budget["timeout_hours_64_clients"],
            ),
            (2, 3, 6),
        )
        self.assertEqual(budget["campaign_days"], 14)
        self.assertEqual(budget["retry_limit"], 1)
        self.assertTrue(budget["retry_counts_toward_budget"])
        self.assertEqual(budget["abort_when_failed_cell_fraction_exceeds"], 0.05)
        self.assertFalse(budget["retuning_after_start"])
        self.assertFalse(budget["incomplete_matrix_may_support_readme_evidence"])
        for field in (
            "maximum_cells",
            "planned_training_invocations",
            "maximum_training_invocations",
            "training_invocation_retry_reserve",
        ):
            self.assertEqual(budget[field], self.protocol["matrix"]["totals"][field])


if __name__ == "__main__":
    unittest.main()
