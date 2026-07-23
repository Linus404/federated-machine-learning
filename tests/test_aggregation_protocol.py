from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy.aggregate import (
    aggregate_inplace,
    aggregate_median,
    aggregate_trimmed_avg,
)


PROTOCOL_PATH = Path("docs/scientific-protocol-v1.toml")


def _split_vector(vector: np.ndarray, shapes: list[list[int]]) -> list[np.ndarray]:
    """Split one protocol vector into float32 model-weight arrays.

    Args:
        vector: Flat model vector.
        shapes: Model-weight shapes in registered order.

    Returns:
        Float32 arrays reconstructed in C order.
    """
    arrays: list[np.ndarray] = []
    offset = 0
    for registered_shape in shapes:
        shape = tuple(registered_shape)
        size = int(np.prod(shape))
        arrays.append(vector[offset : offset + size].reshape(shape, order="C"))
        offset += size
    if offset != vector.size:
        raise ValueError("golden vector does not match its registered tensor shapes")
    return arrays


class AggregationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = tomllib.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.contract = cls.protocol["aggregation"]
        cls.probe = cls.contract["golden_probe"]

        round_start = np.asarray(cls.probe["round_start_vector"], dtype=np.float32)
        honest = np.asarray(cls.probe["honest_post_fit_vectors"], dtype=np.float32)
        submitted = honest.copy()
        scale = np.float32(cls.probe["threat_scale"])
        for client_id in cls.probe["malicious_client_ids"]:
            delta = np.subtract(honest[client_id], round_start)
            threatened_delta = np.multiply(delta, scale)
            submitted[client_id] = np.add(round_start, threatened_delta)

        cls.submitted = submitted
        cls.weight_sets = [
            _split_vector(vector, cls.probe["tensor_shapes"]) for vector in submitted
        ]

    def test_contract_freezes_complete_validated_aggregation_inputs(self) -> None:
        self.assertEqual(self.contract["flower_version"], "1.32.1")
        self.assertEqual(self.contract["numpy_version"], "2.4.4")
        self.assertIn("exactly one successful FitRes", self.contract["result_set"])
        self.assertIn(
            "partial-round aggregation is forbidden", self.contract["result_set"]
        )
        self.assertIn("type int but not bool", self.contract["client_validation"])
        self.assertIn("equal range(client_scale)", self.contract["client_validation"])
        self.assertIn("dtype exactly numpy.float32", self.contract["model_validation"])
        self.assertIn("complete post-fit", self.contract["submitted_value"])
        self.assertIn(
            "never a pre-aggregated update or delta", self.contract["submitted_value"]
        )
        self.assertIn("flatten(order='C')", self.contract["coordinate_layout"])
        self.assertIn(
            "reshape(part, original_shape, order='C')",
            self.contract["coordinate_layout"],
        )
        self.assertIn(
            "No invalid registered input", self.contract["common_invalid_inputs"]
        )

    def test_protocol_scope_does_not_claim_current_matrix_execution(self) -> None:
        scope = self.protocol["scope"]

        self.assertIn("src/strategy_runner.py", scope["implementation_status"])
        self.assertIn("does not orchestrate", scope["implementation_status"])
        self.assertIn(
            "not the registered experiment runner", scope["reviewed_server_boundary"]
        )
        self.assertIn("FedMedian", scope["reviewed_server_boundary"])
        self.assertIn("FedTrimmedAvg", scope["reviewed_server_boundary"])
        self.assertIn("threat transforms", scope["reviewed_server_boundary"])
        self.assertIn("client-local evaluation", scope["reviewed_server_boundary"])
        self.assertIn(
            "does not independently derive", scope["reviewed_server_boundary"]
        )
        self.assertIn("no client evaluation", scope["experiment_runner_requirement"])

    def test_golden_threat_is_applied_to_delta_before_full_weight_aggregation(
        self,
    ) -> None:
        expected = np.asarray(
            self.probe["submitted_post_fit_vectors"], dtype=np.float32
        )

        np.testing.assert_array_equal(self.submitted, expected)
        np.testing.assert_array_equal(
            self.submitted[0],
            np.asarray(self.probe["honest_post_fit_vectors"][0], dtype=np.float32),
        )
        self.assertIn("numpy.subtract(H,G)", self.contract["threat_transformation"])
        self.assertIn("numpy.add(G,T)", self.contract["threat_transformation"])

    def test_registered_flower_aggregators_match_independent_golden_outputs(
        self,
    ) -> None:
        sample_counts = self.probe["sample_counts"]
        fit_results = [
            (
                None,
                SimpleNamespace(
                    parameters=ndarrays_to_parameters(weights),
                    num_examples=sample_count,
                ),
            )
            for weights, sample_count in zip(
                self.weight_sets, sample_counts, strict=True
            )
        ]
        aggregations = {
            "fedavg": aggregate_inplace(fit_results),
            "fedmedian": aggregate_median(
                list(zip(self.weight_sets, sample_counts, strict=True))
            ),
            "fedtrimmedavg": aggregate_trimmed_avg(
                list(zip(self.weight_sets, sample_counts, strict=True)),
                self.protocol["strategies"]["fedtrimmedavg"]["beta"],
            ),
        }

        for strategy_name, aggregate_weights in aggregations.items():
            with self.subTest(strategy=strategy_name):
                aggregate = np.concatenate(aggregate_weights)
                golden = self.protocol["strategies"][strategy_name]["golden_probe"]
                expected = np.asarray(golden["aggregate_vector"], dtype=np.float32)

                self.assertEqual(aggregate.dtype, np.dtype(np.float32))
                np.testing.assert_array_equal(aggregate, expected)
                np.testing.assert_array_equal(
                    aggregate.view(np.uint32),
                    np.asarray(golden["aggregate_float32_uint32"], dtype=np.uint32),
                )
                self.assertEqual(
                    [list(array.shape) for array in aggregate_weights],
                    self.probe["tensor_shapes"],
                )

    def test_robust_aggregators_ignore_sample_counts_exactly(self) -> None:
        original_counts = self.probe["sample_counts"]
        replacement_counts = [101, 7, 23, 2]

        for aggregate in (
            lambda counts: aggregate_median(
                list(zip(self.weight_sets, counts, strict=True))
            ),
            lambda counts: aggregate_trimmed_avg(
                list(zip(self.weight_sets, counts, strict=True)), 0.25
            ),
        ):
            with self.subTest(aggregate=aggregate):
                original = np.concatenate(aggregate(original_counts))
                replaced = np.concatenate(aggregate(replacement_counts))
                np.testing.assert_array_equal(original, replaced)

    def test_strategy_contracts_freeze_reduction_and_edge_rules(self) -> None:
        strategies = self.protocol["strategies"]
        fedavg = strategies["fedavg"]
        median = strategies["fedmedian"]
        trimmed = strategies["fedtrimmedavg"]

        self.assertIn("aggregate_inplace", fedavg["executable"])
        self.assertIn("strict float32 left fold", fedavg["reduction"])
        self.assertIn("Do not aggregate deltas", fedavg["result"])
        self.assertEqual(median["sample_weighting"].split(".", 1)[0], "None")
        self.assertIn("two central order statistics", median["even_client_rule"])
        self.assertIn("float32 mean implementation", median["reduction"])
        self.assertEqual(trimmed["beta"], 0.25)
        self.assertIn("numpy.partition", trimmed["executable"])
        self.assertIn("float32 sum accumulator", trimmed["reduction"])
        self.assertIn("trim 1, 4, and 16 values per side", trimmed["trim_count_rule"])
        self.assertIn("0.0 <= beta < 0.5", trimmed["invalid_inputs"])


if __name__ == "__main__":
    unittest.main()
