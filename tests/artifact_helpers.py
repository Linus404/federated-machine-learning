from types import SimpleNamespace
from typing import Any


def fake_app_manifest() -> SimpleNamespace:
    """Return a minimal immutable-byte-compatible app-manifest test double.

    Returns
    -------
    types.SimpleNamespace
        App-manifest-shaped object with public bytes and model dimensions.
    """
    return SimpleNamespace(
        payload={
            "vocabulary_size": 20_000,
            "sequence_length": 500,
            "embedding_dim": 100,
        },
        manifest_bytes=b"canonical public manifest\n",
        vocabulary_bytes=b"\n[UNK]\n",
    )


def compatible_model() -> Any:
    """Return a model-shaped test double matching ``fake_app_manifest``.

    Returns
    -------
    Any
        Object exposing the Keras dimension attributes used by the dashboard.
    """
    embedding = SimpleNamespace(
        get_config=lambda: {"input_dim": 20_000, "output_dim": 100}
    )
    return SimpleNamespace(
        input_shape=(None, 500),
        output_shape=(None, 1),
        get_layer=lambda name: embedding if name == "token_embedding" else None,
    )
