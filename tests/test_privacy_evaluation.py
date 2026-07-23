"""Tests for frozen empirical privacy-attack evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from src.privacy_evaluation import (
    membership_inference_evaluation,
    privacy_attack_metrics,
    update_leakage_evaluation,
)


PROTOCOL = {
    "privacy": {
        "metrics": {"target_fpr": 0.01},
        "binary_crossentropy": {
            "clip_minimum": 1e-7,
            "clip_maximum": 0.9999999,
        },
    }
}


class PrivacyEvaluationTests(unittest.TestCase):
    """Exercise the registered golden metrics and attack score definitions."""

    def test_registered_attack_metrics_golden_vector(self) -> None:
        """Preserve tie handling, maximum advantage, and FPR interpolation."""
        result = privacy_attack_metrics(
            np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int64),
            np.asarray([0.9, 0.8, 0.8, 0.4, 0.2, 0.2], dtype=np.float64),
            protocol=PROTOCOL,
        )
        self.assertEqual(result["thresholds"], ["inf", 0.9, 0.8, 0.4, 0.2])
        self.assertEqual(result["fpr"], [0.0, 0.0, 1 / 3, 2 / 3, 1.0])
        self.assertEqual(result["tpr"], [0.0, 1 / 3, 2 / 3, 2 / 3, 1.0])
        self.assertAlmostEqual(result["roc_auc"], 2 / 3)
        self.assertAlmostEqual(result["max_tpr_minus_fpr"], 1 / 3)
        self.assertEqual(result["max_threshold"], 0.9)
        self.assertAlmostEqual(result["tpr_at_1_percent_fpr"], 0.3433333333333333)

    def test_membership_and_update_scores_follow_registered_direction(self) -> None:
        """Make easier members rank above nonmembers and honor zero-norm scores."""
        membership = membership_inference_evaluation(
            np.asarray([1, 0, 1, 0], dtype=np.int64),
            np.asarray([0.9, 0.1, 0.4, 0.6], dtype=np.float32),
            np.asarray([1, 1, 0, 0], dtype=np.int64),
            protocol=PROTOCOL,
        )
        self.assertEqual(membership["records"], 4)
        self.assertEqual(membership["metrics"]["roc_auc"], 1.0)

        update = update_leakage_evaluation(
            np.asarray([[1, 0], [2, 0], [-1, 0], [0, 0]], dtype=np.float32),
            np.asarray([1, 0], dtype=np.float32),
            np.asarray([1, 1, 0, 0], dtype=np.int64),
            protocol=PROTOCOL,
        )
        self.assertEqual(update["zero_norm_scores"], 1)
        self.assertEqual(update["metrics"]["roc_auc"], 1.0)

    def test_rejects_wrong_dtypes_and_single_membership_class(self) -> None:
        """Fail rather than silently cast privacy evidence at the trust boundary."""
        with self.assertRaisesRegex(ValueError, "float64"):
            privacy_attack_metrics(
                np.asarray([1, 0], dtype=np.int64),
                np.asarray([1.0, 0.0], dtype=np.float32),
                protocol=PROTOCOL,
            )
        with self.assertRaisesRegex(ValueError, "member and nonmember"):
            privacy_attack_metrics(
                np.asarray([1, 1], dtype=np.int64),
                np.asarray([1.0, 0.0], dtype=np.float64),
                protocol=PROTOCOL,
            )


if __name__ == "__main__":
    unittest.main()
