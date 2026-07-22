import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STARTUP_ENVIRONMENT = {
    "KERAS_BACKEND": "tensorflow",
    "PYTHONHASHSEED": "0",
    "TF_DETERMINISTIC_OPS": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",
}


def run_python(
    code: str, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a fresh interpreter with an explicit startup environment.

    Parameters
    ----------
    code : str
        Python source passed to ``python -c``.
    environment : dict of str to str
        Complete child process environment.

    Returns
    -------
    subprocess.CompletedProcess of str
        Captured process result.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class ProtocolRuntimeTests(unittest.TestCase):
    def child_environment(self) -> dict[str, str]:
        """Return the current environment with all frozen startup values.

        Returns
        -------
        dict of str to str
            Environment suitable for a conforming fresh interpreter.
        """
        environment = os.environ.copy()
        environment.update(STARTUP_ENVIRONMENT)
        return environment

    def test_fresh_process_requires_every_startup_value(self) -> None:
        for name in STARTUP_ENVIRONMENT:
            with self.subTest(name=name):
                environment = self.child_environment()
                environment.pop(name)
                result = run_python("import src.protocol_runtime", environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{name}: expected", result.stderr)

    def test_fresh_process_rejects_conflicts_and_late_hash_seed(self) -> None:
        environment = self.child_environment()
        environment["KERAS_BACKEND"] = "jax"
        result = run_python("import src.protocol_runtime", environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("KERAS_BACKEND: expected tensorflow, got jax", result.stderr)

        environment = self.child_environment()
        environment["PYTHONHASHSEED"] = "1"
        result = run_python(
            "import os; os.environ['PYTHONHASHSEED']='0'; import src.protocol_runtime",
            environment,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not applied before interpreter startup", result.stderr)

    def test_fresh_process_validates_versions_and_registers_determinism(self) -> None:
        result = run_python(
            "import tensorflow as tf; calls=[]; "
            "tf.config.experimental.enable_op_determinism=lambda: calls.append(True); "
            "from src.protocol_runtime import validate_protocol_runtime; "
            "protocol = validate_protocol_runtime(); "
            "print(protocol['aggregation']['flower_version'], len(calls))",
            self.child_environment(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1.32.1 1")

        mismatch = run_python(
            "import importlib.metadata; original=importlib.metadata.version; "
            "importlib.metadata.version=lambda name: '0.0.0' if name == 'flwr' else original(name); "
            "from src.protocol_runtime import validate_protocol_runtime; "
            "validate_protocol_runtime()",
            self.child_environment(),
        )
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("flower: expected 1.32.1, got 0.0.0", mismatch.stderr)

    def test_startup_contract_and_flower_pin_match_frozen_protocol(self) -> None:
        protocol = tomllib.loads(
            (ROOT / "docs/scientific-protocol-v1.toml").read_text(encoding="utf-8")
        )
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
        env_values = dict(
            line.split("=", maxsplit=1)
            for line in (ROOT / ".env.protocol")
            .read_text(encoding="utf-8")
            .splitlines()
        )

        self.assertEqual(
            protocol["framework"]["execution_environment_before_import"],
            STARTUP_ENVIRONMENT,
        )
        self.assertEqual(env_values, STARTUP_ENVIRONMENT)
        self.assertIn("flwr[simulation]==1.32.1", project["project"]["dependencies"])
        flower = next(
            package for package in lock["package"] if package["name"] == "flwr"
        )
        self.assertEqual(flower["version"], protocol["aggregation"]["flower_version"])
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        for name, value in STARTUP_ENVIRONMENT.items():
            self.assertIn(f"{name}={value}", dockerfile)


if __name__ == "__main__":
    unittest.main()
