"""Configure the local Flower connection and verify federation readiness."""

from __future__ import annotations

import argparse
import os
import stat
import tempfile
import tomllib
from pathlib import Path

from src.flower_readiness import (
    DEFAULT_EXPECTED_ONLINE_SUPERNODES,
    DEFAULT_READINESS_TIMEOUT_SECONDS,
    wait_for_online_supernodes,
)

LOCAL_SUPERLINK_PROFILE = "local-docker"
LOCAL_SUPERLINK_ADDRESS = "127.0.0.1:9093"
LOCAL_SUPERLINK_SETTINGS = {
    "address": LOCAL_SUPERLINK_ADDRESS,
    "insecure": True,
}


def default_flower_config_path() -> Path:
    """Return the Flower configuration path used by the CLI.

    Returns
    -------
    pathlib.Path
        The ``config.toml`` path below ``FLWR_HOME`` or ``~/.flwr``.
    """
    flower_home = Path(os.environ.get("FLWR_HOME", Path.home() / ".flwr"))
    return flower_home.expanduser() / "config.toml"


def _write_text_atomically(path: Path, content: str) -> None:
    """Replace a text file atomically while preserving its existing mode.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    content : str
        Complete UTF-8 text to write.

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        if path.exists():
            temporary_path.chmod(stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def ensure_local_superlink_profile(config_path: str | Path) -> bool:
    """Create or validate the loopback-only local Docker SuperLink profile.

    Existing profiles are never rewritten implicitly. A conflicting profile raises
    an error so the caller cannot silently connect to a remote or TLS-enabled
    SuperLink.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Flower ``config.toml`` file to inspect or create.

    Returns
    -------
    bool
        ``True`` when the profile was added, otherwise ``False`` when an identical
        profile already existed.

    Raises
    ------
    ValueError
        If the configuration is invalid TOML or the existing profile conflicts with
        the required local settings.
    """
    requested_path = Path(config_path).expanduser()
    path = requested_path.resolve() if requested_path.is_symlink() else requested_path
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    try:
        config = tomllib.loads(content)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Flower configuration is not valid TOML: {path}") from error

    superlink_profiles = config.get("superlink", {})
    if not isinstance(superlink_profiles, dict):
        raise ValueError(f"Flower configuration has an invalid superlink table: {path}")
    profile = superlink_profiles.get(LOCAL_SUPERLINK_PROFILE)
    if profile is not None:
        if profile != LOCAL_SUPERLINK_SETTINGS:
            raise ValueError(
                f"[superlink.{LOCAL_SUPERLINK_PROFILE}] in {path} does not match "
                f"{LOCAL_SUPERLINK_SETTINGS!r}; update or remove that table explicitly"
            )
        return False

    separator = "" if not content else ("\n" if content.endswith("\n") else "\n\n")
    profile_text = (
        f"[superlink.{LOCAL_SUPERLINK_PROFILE}]\n"
        f'address = "{LOCAL_SUPERLINK_ADDRESS}"\n'
        "insecure = true\n"
    )
    _write_text_atomically(path, f"{content}{separator}{profile_text}")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

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
        description="Configure local Docker Flower and wait for its four SuperNodes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_flower_config_path(),
        help="Flower config.toml path (default: FLWR_HOME/config.toml or ~/.flwr/config.toml)",
    )
    parser.add_argument(
        "--readiness-timeout",
        type=float,
        default=DEFAULT_READINESS_TIMEOUT_SECONDS,
        help="seconds to wait for all four SuperNodes (default: 120)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Configure the local Docker Flower profile and verify readiness.

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
        created = ensure_local_superlink_profile(args.config)
        wait_for_online_supernodes(
            address=LOCAL_SUPERLINK_ADDRESS,
            expected_online=DEFAULT_EXPECTED_ONLINE_SUPERNODES,
            timeout_seconds=args.readiness_timeout,
        )
    except (TimeoutError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    action = "created" if created else "validated"
    print(f"{action} [superlink.{LOCAL_SUPERLINK_PROFILE}] in {args.config}")
    print(
        f"validated exactly {DEFAULT_EXPECTED_ONLINE_SUPERNODES} online SuperNodes "
        f"at {LOCAL_SUPERLINK_ADDRESS}"
    )


if __name__ == "__main__":
    main()
