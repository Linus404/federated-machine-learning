import json
import subprocess
import sys
import unittest

from src.reproducibility import derive_seed, effective_master_seed


_PROCESS_SCRIPT = r"""
import hashlib
import json
import sys

import numpy as np

from src.local_training import build_model, seed_model_training

master_seed, client_id, server_round = map(int, sys.argv[1:])
model = build_model(
    vocab_size=17,
    sequence_length=11,
    embedding_dim=5,
    master_seed=master_seed,
    seed_namespace=("client", client_id),
)

def digest_weights():
    digest = hashlib.sha256()
    for weight in model.get_weights():
        digest.update(weight.dtype.str.encode())
        digest.update(str(weight.shape).encode())
        digest.update(weight.tobytes())
    return digest.hexdigest()

initial = digest_weights()
x = np.asarray(
    [[2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0],
     [4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0],
     [2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
     [4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype="int32",
)
y = np.asarray([1, 0, 1, 0], dtype="float32")
seed_model_training(
    model,
    master_seed,
    "client",
    client_id,
    "round",
    server_round,
)
model.fit(x, y, epochs=2, batch_size=2, shuffle=True, verbose=0)
print(json.dumps({"initial": initial, "trained": digest_weights()}))
"""


class ReproducibilityTests(unittest.TestCase):
    def run_training_process(
        self, master_seed: int, client_id: int, server_round: int
    ) -> dict[str, str]:
        """Run one deterministic training process and return its model digests.

        Parameters
        ----------
        master_seed : int
            Effective master seed.
        client_id : int
            Client namespace.
        server_round : int
            Round namespace.

        Returns
        -------
        dict of str to str
            Initial and trained model SHA-256 values.
        """
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _PROCESS_SCRIPT,
                str(master_seed),
                str(client_id),
                str(server_round),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_independent_processes_repeat_models_and_training_exactly(self) -> None:
        first = self.run_training_process(67, 2, 4)
        second = self.run_training_process(67, 2, 4)

        self.assertEqual(first, second)

    def test_master_client_and_round_namespaces_separate_streams(self) -> None:
        baseline = self.run_training_process(67, 2, 4)
        next_master = self.run_training_process(68, 2, 4)
        next_client = self.run_training_process(67, 3, 4)
        next_round = self.run_training_process(67, 2, 5)

        self.assertNotEqual(baseline["initial"], next_master["initial"])
        self.assertNotEqual(baseline["initial"], next_client["initial"])
        self.assertEqual(baseline["initial"], next_round["initial"])
        self.assertNotEqual(baseline["trained"], next_round["trained"])

    def test_seed_derivation_rejects_hostile_inputs_and_separates_namespaces(
        self,
    ) -> None:
        seeds = {
            derive_seed(67, "model", 0),
            derive_seed(67, "dropout", 0),
            derive_seed(67, "client", 0, "round", 1, "update-noise"),
            derive_seed(67, "client", 1, "round", 1, "update-noise"),
            derive_seed(67, "client", 0, "round", 2, "update-noise"),
        }
        self.assertEqual(len(seeds), 5)
        for invalid in (-1, 2**32, True, "67"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                effective_master_seed({"master-seed": invalid})
        for namespace in ((), ("",), (True,), (-1,), (1.5,)):
            with self.subTest(namespace=namespace), self.assertRaises(ValueError):
                derive_seed(67, *namespace)


if __name__ == "__main__":
    unittest.main()
