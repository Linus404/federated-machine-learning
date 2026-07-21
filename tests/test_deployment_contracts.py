import unittest
from pathlib import Path


def service_block(compose: str, service: str) -> str:
    """Return one service definition from Compose source text.

    Parameters
    ----------
    compose : str
        Complete Compose YAML source.
    service : str
        Service name whose indented block should be returned.

    Returns
    -------
    str
        The service body without its top-level service key.
    """
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
        services_block = self.compose.split("services:\n", maxsplit=1)[1].split(
            "\nvolumes:\n", maxsplit=1
        )[0]
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

    def test_superlink_state_survives_container_restarts(self) -> None:
        superlink = service_block(self.compose, "superlink")

        self.assertIn("--database /app/state/state.db", superlink)
        self.assertIn("superlink-state:/app/state", superlink)
        self.assertIn("superlink-state:", self.compose)

    def test_flower_dependencies_wait_for_local_api_availability(self) -> None:
        superlink = service_block(self.compose, "superlink")
        self.assertIn("healthcheck:", superlink)
        supernode_defaults = self.compose.split("services:\n", maxsplit=1)[0]
        self.assertIn("x-supernode: &supernode", supernode_defaults)
        self.assertIn("healthcheck:", supernode_defaults)
        self.assertIn('127.0.0.1", 9094', supernode_defaults)
        self.assertNotIn('127.0.0.1", 9099', supernode_defaults)

        for service in ["serverapp"] + [f"supernode-{index}" for index in range(4)]:
            with self.subTest(service=service):
                block = service_block(self.compose, service)
                self.assertIn("superlink:", block)
                self.assertIn("condition: service_healthy", block)

        for index in range(4):
            with self.subTest(index=index):
                supernode = service_block(self.compose, f"supernode-{index}")
                clientapp = service_block(self.compose, f"clientapp-{index}")
                self.assertIn("<<: *supernode", supernode)
                self.assertIn(f"supernode-{index}:", clientapp)
                self.assertIn("condition: service_healthy", clientapp)

    def test_run_submission_waits_for_registered_supernodes(self) -> None:
        flower_config = Path("src/flower_config.py").read_text(encoding="utf-8")

        self.assertIn("wait_for_online_supernodes(", flower_config)

    def test_distributed_stack_uses_separate_flower_roles_with_one_image(self) -> None:
        expected_commands = {
            "superlink": "flower-superlink",
            "serverapp": "flower-superexec",
            "dashboard": "streamlit run dashboard.py",
            **{f"supernode-{index}": "flower-supernode" for index in range(4)},
            **{f"clientapp-{index}": "flower-superexec" for index in range(4)},
        }

        self.assertEqual(
            self.compose.count("image: federated-machine-learning:latest"), 1
        )
        for service, command in expected_commands.items():
            with self.subTest(service=service):
                block = service_block(self.compose, service)
                self.assertRegex(block, r"<<: \*(flower-service|clientapp|supernode)")
                self.assertIn(command, block)
                self.assertNotIn("flwr run", block)

        self.assertIn(
            "--plugin-type serverapp", service_block(self.compose, "serverapp")
        )
        for index in range(4):
            self.assertIn(
                "--plugin-type clientapp",
                service_block(self.compose, f"clientapp-{index}"),
            )

        for deprecated_entrypoint in ("flwr-serverapp", "flwr-clientapp"):
            self.assertNotIn(deprecated_entrypoint, self.compose)

    def test_training_services_receive_documented_host_git_revision(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        revision_environment = 'FML_CODE_REVISION: "${FML_CODE_REVISION:-}"'
        clientapp_defaults = self.compose.split("x-supernode:", maxsplit=1)[0]

        self.assertIn(revision_environment, clientapp_defaults)
        self.assertIn(revision_environment, service_block(self.compose, "serverapp"))
        for index in range(4):
            with self.subTest(index=index):
                self.assertIn(
                    "<<: *clientapp",
                    service_block(self.compose, f"clientapp-{index}"),
                )
        self.assertIn(
            'FML_CODE_REVISION="$(git rev-parse HEAD)" docker compose up --build -d',
            readme,
        )

    def test_container_build_context_excludes_git_metadata(self) -> None:
        dockerignore = Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

        self.assertIn(".git", dockerignore)
        self.assertNotIn("COPY .git", dockerfile)

    def test_contract_helpers_follow_project_docstring_conventions(self) -> None:
        docstring = service_block.__doc__ or ""

        self.assertIn("Parameters\n", docstring)
        self.assertIn("Returns\n", docstring)

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
                self.assertIn(f"--appio-api-address supernode-{index}:9094", clientapp)
                self.assertIn(f"supernode-{index}:", clientapp)

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

    def test_regeneration_uses_matching_non_destructive_artifact_paths(self) -> None:
        compatibility = Path("COMPATIBILITY.md").read_text(encoding="utf-8")
        root = "artifacts/regenerated-schema-1"

        self.assertIn(f"--client-shard-dir {root}/clients", compatibility)
        self.assertIn(f"--public-artifact-dir {root}/public", compatibility)
        self.assertIn(
            f"client-data-dir='{root}/clients/client-{{partition}}'", compatibility
        )
        self.assertIn(f"public-artifact-dir='{root}/public'", compatibility)
        self.assertIn(f"server-artifact-dir='{root}/server'", compatibility)


if __name__ == "__main__":
    unittest.main()
