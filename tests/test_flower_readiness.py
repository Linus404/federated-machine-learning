import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import src.flower_readiness as flower_readiness


class FlowerReadinessTests(unittest.TestCase):
    def test_count_online_supernodes_uses_control_api_status(self) -> None:
        channel = Mock()
        ready = Mock()
        stub = Mock()
        stub.ListNodes.return_value = SimpleNamespace(
            nodes_info=[
                SimpleNamespace(status="online"),
                SimpleNamespace(status="offline"),
                SimpleNamespace(status="online"),
            ]
        )

        with (
            patch.object(
                flower_readiness.grpc, "insecure_channel", return_value=channel
            ),
            patch.object(
                flower_readiness.grpc, "channel_ready_future", return_value=ready
            ),
            patch.object(flower_readiness, "ControlStub", return_value=stub),
        ):
            count = flower_readiness.count_online_supernodes(
                "superlink:9093", timeout_seconds=1.5
            )

        self.assertEqual(count, 2)
        ready.result.assert_called_once_with(timeout=1.5)
        stub.ListNodes.assert_called_once()
        self.assertEqual(stub.ListNodes.call_args.kwargs, {"timeout": 1.5})
        channel.close.assert_called_once_with()

    def test_wait_requires_exact_expected_online_count(self) -> None:
        with (
            patch.object(
                flower_readiness,
                "count_online_supernodes",
                side_effect=[3, 5, 4],
            ) as count_online,
            patch.object(flower_readiness.time, "sleep"),
        ):
            flower_readiness.wait_for_online_supernodes(
                expected_online=4,
                timeout_seconds=5,
                retry_interval_seconds=0.01,
            )

        self.assertEqual(count_online.call_count, 3)

    def test_wait_times_out_with_last_observed_count(self) -> None:
        with (
            patch.object(
                flower_readiness.time,
                "monotonic",
                side_effect=[0.0, 0.0, 1.0],
            ),
            patch.object(flower_readiness, "count_online_supernodes", return_value=2),
            patch.object(flower_readiness.time, "sleep"),
        ):
            with self.assertRaisesRegex(
                TimeoutError, "last observed online count was 2"
            ):
                flower_readiness.wait_for_online_supernodes(
                    expected_online=4,
                    timeout_seconds=1,
                    retry_interval_seconds=0.01,
                )


if __name__ == "__main__":
    unittest.main()
