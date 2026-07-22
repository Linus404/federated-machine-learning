from __future__ import annotations

import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from src.protocol_runtime import validate_protocol_runtime

import keras
import tensorflow as tf

from src.app_manifest import (
    AppManifest,
    load_app_manifest,
    resolve_public_artifact_dir,
)
from src.artifact_history import CURRENT_RUN_FILENAME, load_current_run_snapshot
from src.artifact_compatibility import (
    SERVER_ARTIFACT_MANIFEST_FILENAME,
    SERVER_ARTIFACTS,
    load_server_artifact_snapshot,
)
from src.paths import default_public_artifact_dir, default_server_artifact_dir
from src.text_preprocessing import create_text_vectorizer


PUBLIC_ARTIFACT_DIR: Path = default_public_artifact_dir()
ARTIFACT_ROOT: Path = default_server_artifact_dir()
DEFAULT_REFRESH_SECONDS = 6
IDLE_STOP_AFTER = 7


def _is_fresh_artifact_root(root: Path) -> bool:
    """Return whether no current or legacy server artifacts exist.

    Parameters
    ----------
    root : pathlib.Path
        Configured server artifact root.

    Returns
    -------
    bool
        ``True`` only for an absent or artifact-empty regular directory.
    """
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        return False
    filenames = {
        CURRENT_RUN_FILENAME,
        SERVER_ARTIFACT_MANIFEST_FILENAME,
        "run_manifest.json",
        *(str(layout["filename"]) for layout in SERVER_ARTIFACTS.values()),
    }
    return not any(
        (root / filename).exists() or (root / filename).is_symlink()
        for filename in filenames
    )


def load_model(path: Path | None = None) -> Any:
    """Load the trained global sentiment model.

    Parameters
    ----------
    path : pathlib.Path or None, optional
        Explicit legacy model path, or ``None`` to load the current run snapshot.

    Returns
    -------
    Any
        Loaded Keras model.

    Raises
    ------
    FileNotFoundError
        If no trained model artifact exists.
    """
    validate_protocol_runtime()
    app_manifest = load_public_snapshot()
    return _load_bound_model(app_manifest, path)


@st.cache_resource
def _load_public_snapshot_at(public_dir: Path) -> AppManifest:
    """Load and cache one immutable public generation directory.

    Parameters
    ----------
    public_dir : pathlib.Path
        Canonical immutable public generation directory.

    Returns
    -------
    AppManifest
        Validated public artifact snapshot.
    """
    return load_app_manifest(public_artifact_dir=public_dir)


def load_public_snapshot() -> AppManifest:
    """Load the currently selected validated public artifact snapshot.

    Returns
    -------
    AppManifest
        Cached snapshot keyed by its immutable generation directory.

    Raises
    ------
    ValueError
        If the selected generation or public artifacts are unsafe or incompatible.
    """
    public_dir = resolve_public_artifact_dir(
        public_artifact_dir=PUBLIC_ARTIFACT_DIR
    ).resolve(strict=True)
    return _load_public_snapshot_at(public_dir)


def _load_bound_model(app_manifest: AppManifest, path: Path | None = None) -> Any:
    """Load one server model bound to a validated public snapshot.

    Parameters
    ----------
    app_manifest : AppManifest
        Validated public app-manifest snapshot retained by the caller.
    path : pathlib.Path or None, optional
        Explicit server model path, or the selected current run when omitted.

    Returns
    -------
    Any
        Loaded dimension-validated Keras model.

    Raises
    ------
    FileNotFoundError
        If the bound server snapshot contains no model.
    ValueError
        If the server binding or serialized model dimensions are incompatible.
    """
    snapshot = (
        load_current_run_snapshot(ARTIFACT_ROOT, app_manifest=app_manifest)
        if path is None
        else load_server_artifact_snapshot(path.parent, app_manifest=app_manifest)
    )
    model_bytes = snapshot.files.get("global_model.keras")
    if model_bytes is None:
        raise FileNotFoundError(
            f"No model found in {snapshot.directory}. "
            "Bitte zuerst (federated) Training starten, damit der Server "
            "global_model.keras speichert."
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "global_model.keras"
        model_path.write_bytes(model_bytes)
        model = keras.models.load_model(model_path)
    _validate_model_dimensions(model, app_manifest.payload)
    return model


def _validate_model_dimensions(model: Any, public_manifest: Mapping[str, Any]) -> None:
    """Require a loaded model to match its bound public dimensions.

    Parameters
    ----------
    model : Any
        Loaded Keras model candidate.
    public_manifest : Any
        Validated public manifest payload.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If input, output, vocabulary, or embedding dimensions differ.
    """
    try:
        embedding = model.get_layer("token_embedding")
        config = embedding.get_config()
        dimensions = {
            "vocabulary_size": config["input_dim"],
            "embedding_dim": config["output_dim"],
            "sequence_length": model.input_shape[-1],
        }
        output_units = model.output_shape[-1]
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("server model has no compatible dimension contract") from error
    expected = {
        name: public_manifest[name]
        for name in ("vocabulary_size", "embedding_dim", "sequence_length")
    }
    if dimensions != expected or output_units != 1:
        raise ValueError(
            "server model dimensions do not match its bound public artifacts; "
            "regenerate the server run"
        )


@st.cache_resource
def load_vectorizer() -> keras.layers.TextVectorization:
    """Load the checksum-verified public vocabulary into the shared vectorizer.

    Returns
    -------
    keras.layers.TextVectorization
        Vocabulary-bound vectorizer implementing the frozen protocol.

    Raises
    ------
    ValueError
        If the public manifest or vocabulary artifact is invalid.
    """
    app_manifest = load_public_snapshot()
    return create_text_vectorizer(
        sequence_length=app_manifest.payload["sequence_length"],
        vocabulary=app_manifest.vocabulary_terms[2:],
    )


def load_metrics(
    path: Path | None = None,
    *,
    filename: str = "metrics.csv",
    app_manifest: AppManifest | None = None,
) -> pd.DataFrame | None:
    """Load metrics from a compatible server artifact directory.

    Parameters
    ----------
    path : pathlib.Path, optional
        Metrics CSV path.
    filename : str, optional
        Current-run artifact filename used when ``path`` is ``None``.
    app_manifest : AppManifest or None, optional
        Validated public snapshot that the server artifacts must match. The
        currently selected snapshot is loaded when omitted.

    Returns
    -------
    pandas.DataFrame or None
        Non-empty metrics, or ``None`` when the CSV is absent or empty.

    Raises
    ------
    ValueError
        If the server artifact schema is unsupported.
    """
    if path is None and _is_fresh_artifact_root(ARTIFACT_ROOT):
        return None
    public_snapshot = app_manifest or load_public_snapshot()
    snapshot = (
        load_current_run_snapshot(ARTIFACT_ROOT, app_manifest=public_snapshot)
        if path is None
        else load_server_artifact_snapshot(path.parent, app_manifest=public_snapshot)
    )
    content = snapshot.files.get(path.name if path is not None else filename)
    if content is None:
        return None
    try:
        df = pd.read_csv(BytesIO(content))
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    return df


def predict_sentiment(review_text: str) -> tuple[float, str]:
    """Predict sentiment with one mutually bound model and vocabulary snapshot.

    Parameters
    ----------
    review_text : str
        Review text to classify.

    Returns
    -------
    tuple of float and str
        Positive-class probability and thresholded sentiment label.

    Raises
    ------
    ValueError
        If runtime or artifact compatibility validation fails.
    """
    validate_protocol_runtime()
    app_manifest = load_public_snapshot()
    model = _load_bound_model(app_manifest)
    vectorizer = create_text_vectorizer(
        sequence_length=app_manifest.payload["sequence_length"],
        vocabulary=app_manifest.vocabulary_terms[2:],
    )
    inputs = tf.constant([review_text])
    token_ids = vectorizer(inputs)
    preds = model.predict(token_ids, verbose=0)
    positive_prob = float(preds[0][0])
    label = "positive" if positive_prob >= 0.5 else "negative"
    return positive_prob, label


def file_signature(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


def main() -> None:
    """Render the Streamlit dashboard.

    Returns
    -------
    None
    """
    st.set_page_config(page_title="Federated Sentiment Dashboard", layout="wide")

    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True
    if "refresh_seconds" not in st.session_state:
        st.session_state.refresh_seconds = DEFAULT_REFRESH_SECONDS
    if "last_signature" not in st.session_state:
        st.session_state.last_signature = None
    if "idle_cycles" not in st.session_state:
        st.session_state.idle_cycles = 0

    current_path = ARTIFACT_ROOT / "current.json"
    current_signature = file_signature(current_path)
    if current_signature == st.session_state.last_signature:
        st.session_state.idle_cycles += 1
    else:
        st.session_state.idle_cycles = 0
        st.session_state.last_signature = current_signature

    if st.session_state.idle_cycles >= IDLE_STOP_AFTER:
        st.session_state.auto_refresh = False

    st.title("Review inference & Training metrics")

    with st.sidebar:
        st.subheader("Dashboard settings")
        st.session_state.auto_refresh = st.toggle(
            "Auto refresh", value=st.session_state.auto_refresh
        )
        st.session_state.refresh_seconds = st.slider(
            "Refresh interval (seconds)",
            1,
            30,
            value=st.session_state.refresh_seconds,
        )
        if st.button("Refresh now"):
            st.session_state.idle_cycles = 0
            st.rerun()

        st.caption(f"Artifact root: {ARTIFACT_ROOT}")
        if st.session_state.auto_refresh:
            st.caption("Auto refresh aktiv")
        else:
            st.caption("Auto refresh aus")

    col_left, col_right = st.columns([1.1, 1.3])

    with col_left:
        st.subheader("Review inference")
        st.caption("Enter a movie review")

        default_text = "This movie was surprisingly thoughtful and fun to watch."
        review = st.text_area(
            label="Test",
            label_visibility="collapsed",
            value=default_text,
            height=200,
            placeholder="Type your review here...",
        )

        if st.button("Predict sentiment") and review.strip():
            try:
                positive_prob, label = predict_sentiment(review)
                st.markdown("### Positive sentiment probability")
                st.markdown(f"**{positive_prob * 100:.1f}%**")
                st.markdown(f"**Prediction:** {label}")
            except (FileNotFoundError, OSError, ValueError, tf.errors.OpError) as error:
                st.error(f"Fehler bei der Vorhersage: {error}")

    with col_right:
        st.subheader("Training metrics")

        app_manifest = load_public_snapshot()
        df_metrics = load_metrics(app_manifest=app_manifest)
        if df_metrics is None:
            st.info(
                "Noch keine Trainingsmetriken gefunden. Wenn gerade ein neuer Lauf startet, "
                "ist das am Anfang normal."
            )
        else:
            st.caption(f"Loaded {len(df_metrics)} metric rows")
            st.dataframe(df_metrics, width="stretch", hide_index=True)

            expected_cols = {"round", "loss", "accuracy"}
            if expected_cols <= set(df_metrics.columns):
                chart_data = df_metrics.sort_values("round").set_index("round")[
                    ["accuracy", "loss"]
                ]
                st.line_chart(chart_data)
            else:
                st.warning(
                    "metrics.csv hat nicht die erwarteten Spalten ('round', 'loss', 'accuracy')."
                )

            st.subheader("Client evaluation accuracy")
            df_client_metrics = load_metrics(
                filename="client_metrics.csv", app_manifest=app_manifest
            )
            client_columns = {"round", "client_id", "accuracy"}
            if df_client_metrics is None:
                st.info("Per-client metrics will appear during the next training run.")
            elif client_columns <= set(df_client_metrics.columns):
                client_chart = df_client_metrics.pivot(
                    index="round", columns="client_id", values="accuracy"
                ).sort_index()
                client_chart.columns = [
                    f"Client {int(client_id)}" for client_id in client_chart.columns
                ]
                st.line_chart(client_chart)
            else:
                st.warning(
                    "client_metrics.csv is missing the expected round, client_id, "
                    "or accuracy columns."
                )

    if st.session_state.auto_refresh:
        time.sleep(st.session_state.refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
