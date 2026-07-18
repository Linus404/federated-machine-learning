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
        cls.compose = Path("compose.yaml").read_text(encoding="utf-8")

    def test_local_stack_has_exact_service_topology(self) -> None:
        services_block = self.compose.split("services:\n", maxsplit=1)[1]
        services = {
            line.strip().removesuffix(":")
            for line in services_block.splitlines()
            if line.startswith("  ")
            and not line.startswith("    ")
            and line.endswith(":")
        }
        expected = {"superlink", "serverapp", "dashboard"}
        expected.update(f"supernode-{index}" for index in range(4))
        expected.update(f"clientapp-{index}" for index in range(4))
        self.assertEqual(services, expected)

    def test_host_ports_are_published_on_loopback_only(self) -> None:
        self.assertIn('"127.0.0.1:9093:9093"', service_block(self.compose, "superlink"))
        self.assertIn('"127.0.0.1:8501:8501"', service_block(self.compose, "dashboard"))

    def test_distributed_stack_uses_separate_flower_roles_with_one_image(self) -> None:
        expected_commands = {
            "superlink": "flower-superlink",
            "serverapp": "flwr-serverapp",
            "dashboard": "streamlit run dashboard.py",
            **{f"supernode-{index}": "flower-supernode" for index in range(4)},
            **{f"clientapp-{index}": "flwr-clientapp" for index in range(4)},
        }

        self.assertEqual(
            self.compose.count("image: federated-machine-learning:latest"), 1
        )
        for service, command in expected_commands.items():
            with self.subTest(service=service):
                block = service_block(self.compose, service)
                self.assertRegex(block, r"<<: \*(flower-service|clientapp)")
                self.assertIn(command, block)
                self.assertNotIn("flwr run", block)

    def test_only_clientapp_receives_one_private_read_only_shard(self) -> None:
        for index in range(4):
            clientapp = service_block(self.compose, f"clientapp-{index}")
            with self.subTest(index=index):
                self.assertIn(f"source: ./artifacts/clients/client-{index}", clientapp)
                self.assertEqual(
                    self.compose.count(f"source: ./artifacts/clients/client-{index}"),
                    1,
                )
                self.assertIn("target: /app/client-data", clientapp)
                self.assertIn("read_only: true", clientapp)
                for other_index in set(range(4)) - {index}:
                    self.assertNotIn(f"client-{other_index}", clientapp)

        self.assertEqual(self.compose.count("target: /app/client-data"), 4)
        for service in ("superlink", "serverapp", "dashboard"):
            self.assertNotIn("client-data", service_block(self.compose, service))
        for index in range(4):
            self.assertNotIn(
                "client-data", service_block(self.compose, f"supernode-{index}")
            )

    def test_supernodes_map_partitions_to_matching_clientapps(self) -> None:
        for index in range(4):
            with self.subTest(index=index):
                supernode = service_block(self.compose, f"supernode-{index}")
                clientapp = service_block(self.compose, f"clientapp-{index}")
                self.assertIn("--superlink superlink:9092", supernode)
                self.assertIn(f"--node-config partition-id={index}", supernode)
                self.assertIn(
                    f"--clientappio-api-address supernode-{index}:9094", clientapp
                )
                self.assertIn(f"- supernode-{index}", clientapp)

    def test_public_artifacts_are_read_only_for_every_consuming_role(self) -> None:
        consumers = ["serverapp", "dashboard"] + [
            f"clientapp-{index}" for index in range(4)
        ]
        for service in consumers:
            with self.subTest(service=service):
                self.assertIn(
                    "./artifacts/public:/app/artifacts/public:ro",
                    service_block(self.compose, service),
                )
        for service in ["superlink"] + [f"supernode-{index}" for index in range(4)]:
            with self.subTest(service=service):
                self.assertNotIn(
                    "/app/artifacts/public", service_block(self.compose, service)
                )

    def test_server_outputs_have_one_writer_and_read_only_dashboard(self) -> None:
        self.assertIn(
            "./artifacts/server:/app/artifacts/server\n",
            service_block(self.compose, "serverapp"),
        )
        self.assertIn(
            "./artifacts/server:/app/artifacts/server:ro",
            service_block(self.compose, "dashboard"),
        )

    def test_local_superlink_profile_uses_user_flower_config(self) -> None:
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertNotIn("[tool.flwr.federations.local-docker]", pyproject)
        self.assertIn("uv run python -m src.flower_config", readme)
        self.assertNotIn("grep -q", readme)

    def test_gce_capability_is_retained_behind_deployment_boundary(self) -> None:
        retained_files = {
            Path("deploy/gce/README.md"),
            Path("deploy/gce/gce-bootstrap.sh"),
            Path("deploy/gce/gce-run.sh"),
            Path("deploy/gce/server.compose.yaml"),
            Path("deploy/gce/client.compose.yaml"),
        }

        self.assertTrue(all(path.is_file() for path in retained_files))
        self.assertEqual(
            {path.name for path in Path("deploy").iterdir()},
            {"gce"},
        )
        self.assertFalse(
            any("gce" in path.name.lower() for path in Path("src").glob("*.py"))
        )
        self.assertNotIn("gce", self.compose.lower())
        self.assertNotIn("gcloud", self.compose.lower())

        gce_readme = Path("deploy/gce/README.md").read_text(encoding="utf-8")
        self.assertIn("cloud credits are unavailable", gce_readme)
        self.assertIn("docker compose config", gce_readme)
        self.assertIn("bash -n", gce_readme)

    def test_gce_control_plane_has_no_artifact_access(self) -> None:
        server_compose = Path("deploy/gce/server.compose.yaml").read_text(
            encoding="utf-8"
        )
        client_compose = Path("deploy/gce/client.compose.yaml").read_text(
            encoding="utf-8"
        )

        for compose, service in (
            (server_compose, "superlink"),
            (client_compose, "supernode"),
        ):
            with self.subTest(service=service):
                block = service_block(compose, service)
                self.assertNotIn("/app/artifacts", block)
                self.assertNotIn("CLIENT_DATA_DIR", block)

    def test_gce_app_roles_receive_only_required_artifacts(self) -> None:
        server_compose = Path("deploy/gce/server.compose.yaml").read_text(
            encoding="utf-8"
        )
        client_compose = Path("deploy/gce/client.compose.yaml").read_text(
            encoding="utf-8"
        )

        clientapp = service_block(client_compose, "clientapp")
        self.assertIn("source: ${CLIENT_SHARD_DIR:?set CLIENT_SHARD_DIR}", clientapp)
        self.assertIn("target: /app/client-data", clientapp)
        self.assertIn("read_only: true", clientapp)
        self.assertIn("../../artifacts/public:/app/artifacts/public:ro", clientapp)

        serverapp = service_block(server_compose, "serverapp")
        dashboard = service_block(server_compose, "dashboard")
        self.assertIn("../../artifacts/server:/app/artifacts/server\n", serverapp)
        self.assertIn("../../artifacts/server:/app/artifacts/server:ro", dashboard)


if __name__ == "__main__":
    unittest.main()
