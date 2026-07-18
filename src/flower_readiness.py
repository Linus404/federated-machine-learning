"""Verify that the expected Flower SuperNodes are registered and online."""

from __future__ import annotations

import argparse
import time

import grpc
from flwr.proto.control_pb2 import ListNodesRequest  # pylint: disable=E0611
from flwr.proto.control_pb2_grpc import ControlStub

DEFAULT_CONTROL_API_ADDRESS = "127.0.0.1:9093"
DEFAULT_EXPECTED_ONLINE_SUPERNODES = 4
DEFAULT_READINESS_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRY_INTERVAL_SECONDS = 1.0
RPC_TIMEOUT_SECONDS = 2.0


def count_online_supernodes(address: str, timeout_seconds: float) -> int:
    """Return the number of online SuperNodes reported by SuperLink.

    Parameters
    ----------
    address : str
        Flower Control API address.
    timeout_seconds : float
        Maximum time for connection establishment and the list request.

    Returns
    -------
    int
        Number of SuperNodes whose reported status is ``online``.

    Raises
    ------
    grpc.FutureTimeoutError
        If the Control API channel is not ready before the deadline.
    grpc.RpcError
        If the Control API request fails.
    """
    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_seconds)
        response = ControlStub(channel).ListNodes(
            ListNodesRequest(), timeout=timeout_seconds
        )
        return sum(node.status == "online" for node in response.nodes_info)
    finally:
        channel.close()


def wait_for_online_supernodes(
    address: str = DEFAULT_CONTROL_API_ADDRESS,
    expected_online: int = DEFAULT_EXPECTED_ONLINE_SUPERNODES,
    timeout_seconds: float = DEFAULT_READINESS_TIMEOUT_SECONDS,
    retry_interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
) -> None:
    """Wait until SuperLink reports exactly the expected online SuperNodes.

    Parameters
    ----------
    address : str, optional
        Flower Control API address.
    expected_online : int, optional
        Exact number of online SuperNodes required by the federation contract.
    timeout_seconds : float, optional
        Overall readiness deadline in seconds.
    retry_interval_seconds : float, optional
        Delay between unsuccessful checks in seconds.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a numeric argument is not positive.
    TimeoutError
        If the expected online count is not observed before the deadline.
    """
    if expected_online <= 0:
        raise ValueError("expected_online must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if retry_interval_seconds <= 0:
        raise ValueError("retry_interval_seconds must be positive")

    deadline = time.monotonic() + timeout_seconds
    last_count: int | None = None
    last_error: grpc.RpcError | grpc.FutureTimeoutError | None = None

    while (remaining := deadline - time.monotonic()) > 0:
        try:
            last_count = count_online_supernodes(
                address, timeout_seconds=min(RPC_TIMEOUT_SECONDS, remaining)
            )
            last_error = None
            if last_count == expected_online:
                return
        except (grpc.RpcError, grpc.FutureTimeoutError) as error:
            last_error = error

        time.sleep(min(retry_interval_seconds, max(0.0, remaining)))

    detail = (
        f"last observed online count was {last_count}"
        if last_error is None
        else f"last Control API error was {last_error}"
    )
    raise TimeoutError(
        f"Flower federation at {address} did not reach exactly {expected_online} "
        f"online SuperNodes within {timeout_seconds:g} seconds; {detail}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse readiness command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. ``None`` reads from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed command-line values.
    """
    parser = argparse.ArgumentParser(
        description="Wait for the fixed Flower federation to register all SuperNodes."
    )
    parser.add_argument("--address", default=DEFAULT_CONTROL_API_ADDRESS)
    parser.add_argument(
        "--expected-online",
        type=int,
        default=DEFAULT_EXPECTED_ONLINE_SUPERNODES,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_READINESS_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_RETRY_INTERVAL_SECONDS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Wait for federation readiness and print the verified state.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. ``None`` reads from ``sys.argv``.

    Returns
    -------
    None
    """
    args = parse_args(argv)
    try:
        wait_for_online_supernodes(
            address=args.address,
            expected_online=args.expected_online,
            timeout_seconds=args.timeout,
            retry_interval_seconds=args.interval,
        )
    except (TimeoutError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"validated exactly {args.expected_online} online SuperNodes at {args.address}"
    )


if __name__ == "__main__":
    main()
