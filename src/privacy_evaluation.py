"""Reproducible empirical privacy-attack evaluation for experiment artifacts."""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.artifact_compatibility import read_regular_file, write_json_atomically
from src.evaluation_artifact import load_scientific_protocol


def _validate_labels(labels: np.ndarray, expected_size: int) -> None:
    """Validate exact binary privacy labels.

    Parameters
    ----------
    labels : numpy.ndarray
        Candidate binary labels.
    expected_size : int
        Required number of labels.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If labels are not a same-size, rank-one, binary ``int64`` array.
    """
    if (
        not isinstance(labels, np.ndarray)
        or labels.ndim != 1
        or labels.size != expected_size
        or labels.dtype != np.dtype(np.int64)
        or not np.all(np.isin(labels, [0, 1]))
    ):
        raise ValueError("privacy labels must be a same-size binary int64 vector")


def privacy_attack_metrics(
    membership_labels: np.ndarray,
    scores: np.ndarray,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the frozen ROC summaries for empirical attack scores.

    Parameters
    ----------
    membership_labels : numpy.ndarray
        Rank-one binary ``int64`` labels where one means member.
    scores : numpy.ndarray
        Same-size rank-one finite ``float64`` scores; larger means more likely member.
    protocol : mapping of str to Any or None, optional
        Frozen protocol override used by deterministic tests.

    Returns
    -------
    dict of str to Any
        ROC curve, area, maximum TPR-FPR, its threshold, and TPR at one-percent FPR.

    Raises
    ------
    ValueError
        If inputs violate the frozen privacy metric boundary.
    """
    if (
        not isinstance(scores, np.ndarray)
        or scores.ndim != 1
        or scores.size == 0
        or scores.dtype != np.dtype(np.float64)
        or not np.all(np.isfinite(scores))
    ):
        raise ValueError("privacy scores must be a nonempty finite float64 vector")
    _validate_labels(membership_labels, scores.size)
    positives = int(np.count_nonzero(membership_labels == 1))
    negatives = int(scores.size - positives)
    if not positives or not negatives:
        raise ValueError("privacy metrics require member and nonmember records")

    thresholds = np.concatenate(([np.inf], np.unique(scores)[::-1]))
    fpr = np.empty(thresholds.size, dtype=np.float64)
    tpr = np.empty(thresholds.size, dtype=np.float64)
    for index, threshold in enumerate(thresholds):
        predicted_member = scores >= threshold
        tpr[index] = (
            np.count_nonzero(predicted_member & (membership_labels == 1)) / positives
        )
        fpr[index] = (
            np.count_nonzero(predicted_member & (membership_labels == 0)) / negatives
        )

    advantage = tpr - fpr
    maximum_index = int(np.argmax(advantage))
    distinct_fpr = np.unique(fpr)
    retained_tpr = np.asarray(
        [np.max(tpr[fpr == value]) for value in distinct_fpr], dtype=np.float64
    )
    frozen = protocol or load_scientific_protocol()
    target_fpr = float(frozen["privacy"]["metrics"]["target_fpr"])
    maximum_threshold = thresholds[maximum_index]
    return {
        "thresholds": [
            "inf" if np.isposinf(value) else float(value) for value in thresholds
        ],
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": float(np.trapezoid(tpr, fpr)),
        "max_tpr_minus_fpr": float(advantage[maximum_index]),
        "max_threshold": (
            "inf" if np.isposinf(maximum_threshold) else float(maximum_threshold)
        ),
        "tpr_at_1_percent_fpr": float(
            np.interp(target_fpr, distinct_fpr, retained_tpr)
        ),
    }


def membership_inference_evaluation(
    class_labels: np.ndarray,
    probabilities: np.ndarray,
    membership_labels: np.ndarray,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen loss-based black-box membership attack.

    Parameters
    ----------
    class_labels : numpy.ndarray
        Candidate records' binary class labels with exact ``int64`` dtype.
    probabilities : numpy.ndarray
        Matching model sigmoid probabilities with exact ``float32`` dtype.
    membership_labels : numpy.ndarray
        Matching binary ``int64`` membership labels.
    protocol : mapping of str to Any or None, optional
        Frozen protocol override used by deterministic tests.

    Returns
    -------
    dict of str to Any
        Attack identity, record count, score direction, and registered metrics.

    Raises
    ------
    ValueError
        If candidate labels or probabilities violate the registered boundary.
    """
    if (
        not isinstance(probabilities, np.ndarray)
        or probabilities.ndim != 1
        or probabilities.size == 0
        or probabilities.dtype != np.dtype(np.float32)
        or not np.all(np.isfinite(probabilities))
        or not np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    ):
        raise ValueError(
            "membership probabilities must be a nonempty float32 probability vector"
        )
    _validate_labels(class_labels, probabilities.size)
    _validate_labels(membership_labels, probabilities.size)
    frozen = protocol or load_scientific_protocol()
    config = frozen["privacy"]["binary_crossentropy"]
    labels = class_labels.astype(np.float64)
    clipped = np.clip(
        probabilities.astype(np.float64),
        float(config["clip_minimum"]),
        float(config["clip_maximum"]),
    )
    losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    return {
        "attack": "negative_per_example_binary_crossentropy_loss",
        "records": int(probabilities.size),
        "score_direction": "larger_means_more_likely_member",
        "metrics": privacy_attack_metrics(membership_labels, -losses, protocol=frozen),
    }


def update_leakage_evaluation(
    negative_loss_gradients: np.ndarray,
    client_update: np.ndarray,
    membership_labels: np.ndarray,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate cosine leakage from exact per-record gradients and one client update.

    Parameters
    ----------
    negative_loss_gradients : numpy.ndarray
        Rank-two finite ``float32`` negative per-example loss gradients, one row per
        candidate and flattened in trainable-variable order.
    client_update : numpy.ndarray
        Rank-one finite ``float32`` post-fit-minus-pre-fit target client update in the
        same flattened order.
    membership_labels : numpy.ndarray
        Candidate binary ``int64`` target-client membership labels.
    protocol : mapping of str to Any or None, optional
        Frozen protocol override used by deterministic tests.

    Returns
    -------
    dict of str to Any
        Attack identity, zero-norm count, score direction, and registered metrics.

    Raises
    ------
    ValueError
        If gradients, update, or labels are malformed or non-finite.
    """
    if (
        not isinstance(negative_loss_gradients, np.ndarray)
        or negative_loss_gradients.ndim != 2
        or negative_loss_gradients.shape[0] == 0
        or negative_loss_gradients.shape[1] == 0
        or negative_loss_gradients.dtype != np.dtype(np.float32)
        or not np.all(np.isfinite(negative_loss_gradients))
    ):
        raise ValueError("negative loss gradients must be a finite float32 matrix")
    if (
        not isinstance(client_update, np.ndarray)
        or client_update.ndim != 1
        or client_update.shape[0] != negative_loss_gradients.shape[1]
        or client_update.dtype != np.dtype(np.float32)
        or not np.all(np.isfinite(client_update))
    ):
        raise ValueError("client update must be a matching finite float32 vector")
    _validate_labels(membership_labels, negative_loss_gradients.shape[0])

    gradients = negative_loss_gradients.astype(np.float64)
    update = client_update.astype(np.float64)
    denominators = np.linalg.norm(gradients, axis=1) * np.linalg.norm(update)
    scores = np.zeros(gradients.shape[0], dtype=np.float64)
    nonzero = denominators != 0.0
    scores[nonzero] = (gradients[nonzero] @ update) / denominators[nonzero]
    return {
        "attack": "negative_loss_gradient_client_update_cosine_similarity",
        "records": int(gradients.shape[0]),
        "score_direction": "larger_means_more_likely_target_client_member",
        "zero_norm_scores": int(np.count_nonzero(~nonzero)),
        "metrics": privacy_attack_metrics(membership_labels, scores, protocol=protocol),
    }


def _load_npy(path: Path) -> np.ndarray:
    """Load one contained regular NumPy artifact without pickle support.

    Parameters
    ----------
    path : pathlib.Path
        Input ``.npy`` artifact.

    Returns
    -------
    numpy.ndarray
        Exact array stored in the artifact.
    """
    return np.load(
        io.BytesIO(read_regular_file(path, parent=path.parent)), allow_pickle=False
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the empirical privacy-evaluation command.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    argparse.Namespace
        Parsed input and output artifact paths.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate registered empirical privacy attack scores."
    )
    subparsers = parser.add_subparsers(dest="attack", required=True)
    membership = subparsers.add_parser("membership")
    membership.add_argument("--class-labels", type=Path, required=True)
    membership.add_argument("--probabilities", type=Path, required=True)
    membership.add_argument("--membership-labels", type=Path, required=True)
    update = subparsers.add_parser("update-leakage")
    update.add_argument("--negative-loss-gradients", type=Path, required=True)
    update.add_argument("--client-update", type=Path, required=True)
    update.add_argument("--membership-labels", type=Path, required=True)
    for command in (membership, update):
        command.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one attack from immutable NumPy inputs and persist JSON metrics.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed privacy-evaluation arguments.

    Returns
    -------
    dict of str to Any
        Persisted empirical attack result.
    """
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = (
        membership_inference_evaluation(
            _load_npy(args.class_labels),
            _load_npy(args.probabilities),
            _load_npy(args.membership_labels),
        )
        if args.attack == "membership"
        else update_leakage_evaluation(
            _load_npy(args.negative_loss_gradients),
            _load_npy(args.client_update),
            _load_npy(args.membership_labels),
        )
    )
    write_json_atomically(args.output, result, overwrite=False)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the empirical privacy evaluator.

    Parameters
    ----------
    argv : list of str or None, optional
        Explicit arguments, or process arguments when omitted.

    Returns
    -------
    None
    """
    run(parse_args(argv))


if __name__ == "__main__":
    main()
