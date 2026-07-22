from __future__ import annotations

import unittest
from importlib.metadata import version


class SecureAggregationApiAvailabilityTests(unittest.TestCase):
    def test_flower_1321_api_availability_not_protocol_validation(self) -> None:
        """Probe imports and constructors, not the SecAgg+ protocol."""
        from flwr.client import ClientApp
        from flwr.client.mod import secaggplus_mod
        from flwr.server import ServerApp
        from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

        self.assertEqual(version("flwr"), "1.32.1")
        self.assertIsInstance(ServerApp(), ServerApp)
        self.assertIsInstance(
            ClientApp(client_fn=lambda context: None, mods=[secaggplus_mod]),
            ClientApp,
        )
        self.assertIsInstance(
            DefaultWorkflow(
                fit_workflow=SecAggPlusWorkflow(
                    num_shares=3,
                    reconstruction_threshold=2,
                )
            ),
            DefaultWorkflow,
        )


if __name__ == "__main__":
    unittest.main()
