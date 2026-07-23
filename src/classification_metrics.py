"""Canonical binary-classification evaluation for the frozen protocol."""

from __future__ import annotations

from typing import Any

import numpy as np

INFERENCE_BATCH_SIZE = 256
LOCAL_ONLY_VALIDATION_SCOPE = "local_only_validation_only"


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    evaluation_scope: str = "standard",
) -> dict[str, Any]:
    """Compute the frozen binary-classification metrics and ROC construction.

    Parameters
    ----------
    labels : numpy.ndarray
        Nonempty rank-one binary labels with exact ``int64`` dtype.
    probabilities : numpy.ndarray
        Same-length sigmoid probabilities with exact ``float32`` dtype.
    evaluation_scope : str, optional
        Scope used only to permit undefined ROC values for local-only validation.

    Returns
    -------
    dict of str to Any
        Threshold predictions, confusion matrix, scalar metrics, and ROC arrays.

    Raises
    ------
    ValueError
        If inputs violate the frozen evaluation contract.
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
    if labels.dtype != np.dtype("int64"):
        raise ValueError("classification labels must use exactly int64")
    if probabilities.dtype != np.dtype("float32"):
        raise ValueError("classification probabilities must use exactly float32")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("classification labels must be binary")
    if not np.all(np.isfinite(probabilities)) or not np.all(
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ):
        raise ValueError("classification probabilities must be finite probabilities")

    positives = int(np.count_nonzero(labels == 1))
    negatives = int(labels.size - positives)
    single_class = positives == 0 or negatives == 0
    if single_class and evaluation_scope != LOCAL_ONLY_VALIDATION_SCOPE:
        raise ValueError("classification ROC requires both classes")

    predicted_positive = probabilities >= np.float32(0.5)
    true_negative = int(np.count_nonzero((labels == 0) & ~predicted_positive))
    false_positive = int(np.count_nonzero((labels == 0) & predicted_positive))
    false_negative = int(np.count_nonzero((labels == 1) & ~predicted_positive))
    true_positive = int(np.count_nonzero((labels == 1) & predicted_positive))
    total = true_negative + false_positive + false_negative + true_positive
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative

    thresholds = None
    fpr = None
    tpr = None
    roc_auc = None
    if not single_class:
        thresholds = np.concatenate(([np.inf], np.unique(probabilities)[::-1]))
        fpr = np.empty(thresholds.size, dtype=np.float64)
        tpr = np.empty(thresholds.size, dtype=np.float64)
        for index, threshold in enumerate(thresholds):
            threshold_predictions = probabilities >= threshold
            tpr[index] = (
                np.count_nonzero(threshold_predictions & (labels == 1)) / positives
            )
            fpr[index] = (
                np.count_nonzero(threshold_predictions & (labels == 0)) / negatives
            )
        roc_auc = float(np.trapezoid(tpr, fpr))

    return {
        "predicted_positive": predicted_positive,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "accuracy": (true_negative + true_positive) / total,
        "precision": true_positive / precision_denominator
        if precision_denominator
        else 0.0,
        "recall": true_positive / recall_denominator if recall_denominator else 0.0,
        "f1": 2 * true_positive / f1_denominator if f1_denominator else 0.0,
        "roc_thresholds": thresholds,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "roc_auc_status": "undefined_single_class" if single_class else "defined",
    }


def predict_probabilities(model: Any, token_ids: np.ndarray) -> np.ndarray:
    """Run frozen-order batched inference and validate sigmoid output tensors.

    Parameters
    ----------
    model : Any
        Trained model callable accepting ``training=False``.
    token_ids : numpy.ndarray
        Nonempty rank-two token IDs in official evaluation order.

    Returns
    -------
    numpy.ndarray
        Contiguous rank-one probabilities with exact ``float32`` dtype.

    Raises
    ------
    ValueError
        If inputs or any model output violate the frozen shape or dtype contract.
    """
    if (
        not isinstance(token_ids, np.ndarray)
        or token_ids.ndim != 2
        or len(token_ids) == 0
    ):
        raise ValueError("evaluation token IDs must be a nonempty rank-two array")
    batches = []
    for start in range(0, len(token_ids), INFERENCE_BATCH_SIZE):
        values = np.asarray(
            model(token_ids[start : start + INFERENCE_BATCH_SIZE], training=False)
        )
        expected_shape = (min(INFERENCE_BATCH_SIZE, len(token_ids) - start), 1)
        if values.shape != expected_shape or values.dtype != np.dtype("float32"):
            raise ValueError(
                "model probabilities must have shape (batch_rows, 1) and float32 dtype"
            )
        batches.append(values)
    return np.ascontiguousarray(np.concatenate(batches, axis=0)[:, 0])


def evaluate_classifier(
    model: Any,
    token_ids: np.ndarray,
    labels: np.ndarray,
    *,
    evaluation_scope: str = "standard",
) -> dict[str, Any]:
    """Infer probabilities and compute the canonical classification result.

    Parameters
    ----------
    model : Any
        Trained classifier.
    token_ids : numpy.ndarray
        Token IDs in official row order.
    labels : numpy.ndarray
        Corresponding exact ``int64`` labels.
    evaluation_scope : str, optional
        Scope used only for local-only single-class validation.

    Returns
    -------
    dict of str to Any
        Complete canonical prediction and metric result.
    """
    probabilities = predict_probabilities(model, token_ids)
    result = classification_metrics(
        labels, probabilities, evaluation_scope=evaluation_scope
    )
    result["probabilities"] = probabilities
    return result
