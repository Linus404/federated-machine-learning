"""Define the deterministic training seed contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

DEFAULT_MASTER_SEED = 67
MASTER_SEED_CONFIG_KEY = "master-seed"
SERVER_ROUND_CONFIG_KEY = "server_round"
SEED_DERIVATION_DOMAIN = "fml-training-seed-v1"
SEED_DERIVATION = {
    "algorithm": "sha256",
    "domain": SEED_DERIVATION_DOMAIN,
    "encoding": "canonical-json-array",
    "output_bits": 32,
}
SEED_NAMESPACES = {
    "client_model": "client/<client_id>/model-construction",
    "client_round_dropout": "client/<client_id>/round/<server_round>/dropout/<layer_index>",
    "client_round_training_order": "client/<client_id>/round/<server_round>/training-order",
    "client_round_update_noise": "client/<client_id>/round/<server_round>/update-noise",
    "local_model": "local/model-construction",
    "local_round_dropout": "local/round/1/dropout/<layer_index>",
    "local_round_training_order": "local/round/1/training-order",
    "server_initial_model": "server/initial/model-construction",
    "server_round_model": "server/round/<server_round>/model-construction",
}


def effective_master_seed(run_config: Mapping[str, Any]) -> int:
    """Return the validated configured or default master seed.

    Parameters
    ----------
    run_config : mapping of str to Any
        Training run configuration.

    Returns
    -------
    int
        Effective unsigned 32-bit master seed.

    Raises
    ------
    ValueError
        If the configured value is not an unsigned 32-bit built-in integer.
    """
    seed = run_config.get(MASTER_SEED_CONFIG_KEY, DEFAULT_MASTER_SEED)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError(f"{MASTER_SEED_CONFIG_KEY} must be an unsigned 32-bit integer")
    return seed


def derive_seed(master_seed: int, *namespace: str | int) -> int:
    """Derive one stable unsigned 32-bit seed from a namespaced master seed.

    Parameters
    ----------
    master_seed : int
        Validated effective master seed.
    *namespace : str or int
        Ordered namespace components that identify one independent stream.

    Returns
    -------
    int
        Deterministically derived unsigned 32-bit seed.

    Raises
    ------
    ValueError
        If the master seed or a namespace component is invalid.
    """
    effective_master_seed({MASTER_SEED_CONFIG_KEY: master_seed})
    if not namespace:
        raise ValueError("seed namespace must not be empty")
    for component in namespace:
        if type(component) is int:
            if component < 0:
                raise ValueError(
                    "integer seed namespace components must be non-negative"
                )
        elif not isinstance(component, str) or not component:
            raise ValueError(
                "seed namespace must contain non-empty strings or non-negative integers"
            )
    document = json.dumps(
        [master_seed, *namespace],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(SEED_DERIVATION_DOMAIN.encode() + b"\0" + document)
    return int.from_bytes(digest.digest()[:4], "big")
