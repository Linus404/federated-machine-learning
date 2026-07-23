from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flwr.clientapp import ClientApp
from flwr.common import Context
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

from src.client_app import client_fn as production_client_fn
from src.server_app import server_fn as production_server_fn

E2E_CONFIG = {
    "num-server-rounds": 1,
    "expected-client-count": 2,
    "local-epochs": 1,
    "batch-size": 16_384,
    "validation-split": 0.2,
    "artifact-retention-runs": 1,
    "proximal-mu": 0.1,
    "use-huber": False,
    "huber-threshold": 10.0,
    "use-update-noise": False,
}


def _with_e2e_config(context: Context) -> Context:
    """Return an equivalent context carrying the bounded E2E run config.

    Parameters
    ----------
    context : flwr.common.Context
        Runtime-owned Flower context, which must remain immutable.

    Returns
    -------
    flwr.common.Context
        New context preserving runtime identity, node state, and node config.
    """
    return Context(
        run_id=context.run_id,
        node_id=context.node_id,
        node_config=context.node_config,
        state=context.state,
        run_config={**context.run_config, **E2E_CONFIG},
        series_id=context.series_id,
    )


def _server_fn(context):
    """Run the production server with the bounded E2E configuration.

    Parameters
    ----------
    context : flwr.common.Context
        Flower simulation context.

    Returns
    -------
    flwr.server.ServerAppComponents
        Production server components configured for one round and two clients.
    """
    return production_server_fn(_with_e2e_config(context))


def _client_fn(context):
    """Run a production client with the bounded E2E configuration.

    Parameters
    ----------
    context : flwr.common.Context
        Flower simulation context containing its assigned partition ID.

    Returns
    -------
    flwr.client.Client
        Production Flower client for the assigned private shard.
    """
    return production_client_fn(_with_e2e_config(context))


@unittest.skipUnless(
    os.environ.get("RUN_FLOWER_E2E") == "1",
    "set RUN_FLOWER_E2E=1 and FML_E2E_ARTIFACT_ROOT to run real Flower E2E",
)
class FlowerEndToEndTests(unittest.TestCase):
    def test_two_clients_complete_one_real_training_round(self) -> None:
        artifact_root = Path(os.environ["FML_E2E_ARTIFACT_ROOT"]).resolve()
        self.assertTrue((artifact_root / "public").is_dir())
        self.assertTrue((artifact_root / "clients" / "client-0").is_dir())
        self.assertTrue((artifact_root / "clients" / "client-1").is_dir())

        with tempfile.TemporaryDirectory() as tmpdir:
            server_artifacts = Path(tmpdir) / "server"
            environment = {
                "CLIENT_DATA_DIR": str(
                    artifact_root / "clients" / "client-{partition}"
                ),
                "FML_PUBLIC_ARTIFACT_DIR": str(artifact_root / "public"),
                "FML_SERVER_ARTIFACT_DIR": str(server_artifacts),
                "TF_CPP_MIN_LOG_LEVEL": "2",
            }
            with patch.dict(os.environ, environment):
                run_simulation(
                    server_app=ServerApp(server_fn=_server_fn),
                    client_app=ClientApp(client_fn=_client_fn),
                    num_supernodes=2,
                    backend_config={
                        "init_args": {
                            "include_dashboard": False,
                            "log_to_driver": True,
                        },
                        "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
                        "actor": {"tensorflow": 0},
                    },
                )

            run_dirs = list((server_artifacts / "runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "global_model.keras").is_file())
            with (run_dir / "client_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual({int(row["client_id"]) for row in rows}, {0, 1})
            self.assertEqual({int(row["round"]) for row in rows}, {1})


if __name__ == "__main__":
    unittest.main()
