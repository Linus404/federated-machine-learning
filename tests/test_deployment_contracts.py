import unittest
from pathlib import Path


def service_block(compose: str, service: str) -> str:
    lines = compose.splitlines()
    start = lines.index(f"  {service}:") + 1
    end = next(
        (
            index
            for index in range(start, len(lines))
            if lines[index].startswith("  ") and not lines[index].startswith("    ")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


class DistributedDeploymentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server_compose = Path("deploy/server.compose.yaml").read_text(
            encoding="utf-8"
        )
        cls.client_compose = Path("deploy/client.compose.yaml").read_text(
            encoding="utf-8"
        )

    def test_distributed_stack_uses_separate_flower_roles_with_one_image(self) -> None:
        expected_commands = {
            "superlink": "flower-superlink",
            "serverapp": "flwr-serverapp",
            "dashboard": "streamlit run dashboard.py",
            "supernode": "flower-supernode",
            "clientapp": "flwr-clientapp",
        }

        for service, command in expected_commands.items():
            with self.subTest(service=service):
                compose = (
                    self.server_compose
                    if service in {"superlink", "serverapp", "dashboard"}
                    else self.client_compose
                )
                block = service_block(compose, service)
                self.assertIn("image: federated-machine-learning:latest", block)
                self.assertIn(command, block)
                self.assertNotIn("flwr run", block)

    def test_only_clientapp_receives_one_private_read_only_shard(self) -> None:
        supernode = service_block(self.client_compose, "supernode")
        clientapp = service_block(self.client_compose, "clientapp")

        self.assertNotIn("CLIENT_SHARD_DIR", supernode)
        self.assertNotIn("target: /app/client-data", supernode)
        self.assertEqual(
            self.client_compose.count(
                "source: ${CLIENT_SHARD_DIR:?set CLIENT_SHARD_DIR}"
            ),
            1,
        )
        self.assertEqual(self.client_compose.count("target: /app/client-data"), 1)
        self.assertIn("read_only: true", clientapp)
        self.assertNotIn("CLIENT_SHARD_DIR", self.server_compose)
        self.assertNotIn("client-data", self.server_compose)

    def test_public_artifacts_are_read_only_for_every_consuming_role(self) -> None:
        for service in ("supernode", "clientapp"):
            with self.subTest(service=service):
                self.assertIn(
                    "../artifacts/public:/app/artifacts/public:ro",
                    service_block(self.client_compose, service),
                )
        for service in ("superlink", "serverapp", "dashboard"):
            with self.subTest(service=service):
                self.assertIn(
                    "../artifacts/public:/app/artifacts/public:ro",
                    service_block(self.server_compose, service),
                )

    def test_gce_capability_is_retained_behind_deployment_boundary(self) -> None:
        retained_files = {
            Path("deploy/README.md"),
            Path("deploy/gce-bootstrap.sh"),
            Path("deploy/gce-run.sh"),
            Path("deploy/server.compose.yaml"),
            Path("deploy/client.compose.yaml"),
        }

        self.assertTrue(all(path.is_file() for path in retained_files))
        self.assertFalse(
            any("gce" in path.name.lower() for path in Path("src").glob("*.py"))
        )
        root_compose = Path("compose.yaml").read_text(encoding="utf-8").lower()
        self.assertNotIn("gce", root_compose)
        self.assertNotIn("gcloud", root_compose)


if __name__ == "__main__":
    unittest.main()
