from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np

from src.local_training import load_partition
from src.paths import data_dir_path, default_data_dir

NUM_PARTITIONS = 4


def compute_confidence(model, x: np.ndarray) -> np.ndarray:
    """Distance from the decision boundary, scaled to [0, 1].

    A sigmoid output near 0 or 1 means the model is very confident;
    near 0.5 means it's unsure. Overfit/memorized samples tend to get
    pushed to extreme confidence.
    """
    preds = model.predict(x, verbose=0).flatten()
    return np.abs(preds - 0.5) * 2.0


def run_membership_inference(model_path: str, data_dir: str | Path | None = None) -> None:
    resolved_data_dir = data_dir_path(data_dir)
    model = keras.models.load_model(model_path)

    member_confidences = []   # samples the model WAS trained on
    nonmember_confidences = []  # samples the model was NOT trained on

    for partition in range(NUM_PARTITIONS):
        train_data, val_data = load_partition(resolved_data_dir, partition, validation_split=0.2)
        x_train, _ = train_data
        x_val, _ = val_data

        member_confidences.append(compute_confidence(model, x_train))
        nonmember_confidences.append(compute_confidence(model, x_val))

    member_confidences = np.concatenate(member_confidences)
    nonmember_confidences = np.concatenate(nonmember_confidences)

    mean_member = member_confidences.mean()
    mean_nonmember = nonmember_confidences.mean()
    gap = mean_member - mean_nonmember

    # Simple threshold attack: guess "member" if confidence > median of everything.
    combined = np.concatenate([member_confidences, nonmember_confidences])
    threshold = np.median(combined)
    true_labels = np.concatenate([
        np.ones_like(member_confidences),
        np.zeros_like(nonmember_confidences),
    ])
    predicted_member = (combined > threshold).astype(float)
    attack_accuracy = (predicted_member == true_labels).mean()

    print(f"Model: {model_path}")
    print(f"Mean confidence on TRAINING samples (members):     {mean_member:.4f}")
    print(f"Mean confidence on HELD-OUT samples (non-members): {mean_nonmember:.4f}")
    print(f"Confidence gap (higher = more leakage):            {gap:.4f}")
    print(f"Membership-inference attack accuracy:               {attack_accuracy:.4f}")
    print(f"(0.50 = random guessing / perfect privacy, 1.00 = attacker always right)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple membership inference attack.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_membership_inference(args.model_path, args.data_dir)
