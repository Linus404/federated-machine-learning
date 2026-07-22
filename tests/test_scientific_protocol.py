import hashlib
import json
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

        def derive(master_seed: int, namespace: str, attempt: int) -> int:
            material = f"{master_seed}|{namespace}|{attempt}".encode("ascii")
            return int.from_bytes(
                hashlib.sha256(material).digest()[:8], "big", signed=False
            )

        self.assertEqual(derive(67, "partition/iid/4/label-0", 0), 6926234993375836234)
        self.assertEqual(
            derive(67, "partition/dirichlet_0.5/4", 3), 13382714623973522527
        )
        self.assertEqual(
            derive(401, "privacy/membership/fedavg/member", 0),
            12749602675305737329,
        )
        namespace = "partition/iid/4/label-0"
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
            derive(master_seed, namespace, attempt),
        )

        namespaces = seeding["namespaces"]
        self.assertEqual(
            set(namespaces),
            {
                "partition_iid",
                "partition_dirichlet",
                "validation",
                "model_initialization",
                "training_order",
                "dropout",
                "malicious_order",
                "membership_member",
                "membership_nonmember",
                "update_member",
                "update_nonmember",
            },
        )
        self.assertEqual(
            namespaces["malicious_order"], "malicious-order/dirichlet_0.5/4"
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
        self.assertEqual(
            (leakage["target_round"], leakage["target_client"]), (1, "client-0")
        )
        self.assertEqual(leakage["member_source"], "client-0_fitted_training_rows")
        self.assertEqual(
            leakage["nonmember_source"],
            "union_of_client-1_client-2_client-3_fitted_training_rows",
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

    def test_system_measurement_boundaries_and_retry_rules_are_exact(self) -> None:
        system = self.protocol["metrics"]["system"]
        self.assertIn("registered_validation_accuracy", system["convergence"])
        self.assertIn("rounds_1_through_20", system["convergence"])
        self.assertFalse(system["convergence_round_zero_included"])
        self.assertEqual(
            system["convergence_evaluation_data"],
            "union_of_registered_train_split_validation_rows",
        )
        self.assertIn("final_trained_model_only", system["test_evaluation_schedule"])
        self.assertIn("never_for_convergence", system["test_evaluation_schedule"])
        self.assertEqual(system["client_fit_timer"], "time.perf_counter_ns")
        self.assertIn("immediately_before", system["client_fit_timer_start"])
        self.assertIn("immediately_after", system["client_fit_timer_stop"])
        self.assertIn(
            "sum_all_attempt_durations", system["client_fit_timer_retry_rule"]
        )
        self.assertEqual(system["client_fit_timer_report_unit"], "nanoseconds")
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
            (300, 345, 360, 15),
        )
        self.assertTrue(matrix["totals"]["privacy_analyses_reuse_trained_models"])
        self.assertIn(
            "matrix.centralized_same_seed",
            membership["centralized_model_source"],
        )
        self.assertIn("same_seed_final_model", membership["fedavg_model_source"])
        self.assertIn("round_1_client-0_update", leakage["fedavg_model_source"])

    def test_compute_and_abort_budget_is_exact(self) -> None:
        budget = self.protocol["budget"]
        self.assertEqual(
            (
                budget["maximum_cells"],
                budget["planned_training_invocations"],
                budget["maximum_training_invocations"],
                budget["training_invocation_retry_reserve"],
            ),
            (300, 345, 360, 15),
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
