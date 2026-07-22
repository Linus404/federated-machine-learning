import math
import re
import tomllib
import unittest
from pathlib import Path


PROTOCOL_PATH = Path("docs/scientific-protocol-v1.toml")


class ScientificProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
        cls.protocol = tomllib.loads(cls.raw_protocol)

    def test_protocol_is_frozen_parseable_and_complete(self) -> None:
        self.assertEqual(self.protocol["protocol_version"], 1)
        self.assertEqual(self.protocol["status"], "frozen")
        self.assertIsNone(
            re.search(r"\b(?:TODO|TBD|null)\b", self.raw_protocol, re.IGNORECASE)
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
        self.assertEqual(strategies["fedprox"]["mu"], 0.1)
        self.assertEqual(
            (
                strategies["fedprox_huber"]["threshold"],
                strategies["fedprox_huber"]["iterations"],
                strategies["fedprox_huber"]["epsilon"],
            ),
            (10.0, 10, 1e-8),
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
            * len(scale["partitions"])
            * scale["seeds"]
            * len(scale["client_scales"]),
        )
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
        invocations = cells - local["cells"] + local["training_invocations"]
        self.assertEqual(invocations, matrix["totals"]["maximum_training_invocations"])
        self.assertEqual((cells, invocations), (300, 360))
        self.assertTrue(matrix["totals"]["privacy_analyses_reuse_trained_models"])

    def test_compute_and_abort_budget_is_exact(self) -> None:
        budget = self.protocol["budget"]
        self.assertEqual(
            (budget["maximum_cells"], budget["maximum_training_invocations"]),
            (300, 360),
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


if __name__ == "__main__":
    unittest.main()
