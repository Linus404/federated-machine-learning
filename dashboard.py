from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import streamlit as st

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras.layers import TextVectorization
import tensorflow as tf

from src.app_manifest import load_app_manifest
from src.paths import (
    client_metrics_path,
    default_public_artifact_dir,
    default_server_artifact_dir,
    global_model_path,
    metrics_path,
)


PUBLIC_ARTIFACT_DIR: Path = default_public_artifact_dir()
ARTIFACT_DIR: Path = default_server_artifact_dir()
MODEL_PATH: Path = global_model_path(ARTIFACT_DIR)
METRICS_PATH: Path = metrics_path(ARTIFACT_DIR)
CLIENT_METRICS_PATH: Path = client_metrics_path(ARTIFACT_DIR)
DEFAULT_REFRESH_SECONDS = 6
IDLE_STOP_AFTER = 7


class _PredictiveModel(Protocol):
    def predict(self, x: Any, verbose: int = 0) -> Any: ...


def load_model() -> _PredictiveModel:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Kein Modell gefunden unter {MODEL_PATH}. "
            "Bitte zuerst (federated) Training starten, damit der Server "
            "global_model.keras speichert."
        )
    return cast(_PredictiveModel, keras.models.load_model(MODEL_PATH))


@st.cache_resource
def load_vectorizer():
    app_manifest = load_app_manifest(public_artifact_dir=PUBLIC_ARTIFACT_DIR)
    vocab_path = app_manifest.vocabulary_path
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Keine Vokabular-Datei gefunden unter {vocab_path}. "
            "Bitte zuerst Stage 1 Datenvorbereitung ausführen."
        )

    saved_vocab = vocab_path.read_text(encoding="utf-8").splitlines()
    # TextVectorization adds "" and "[UNK]" itself when a vocabulary is supplied.
    vocab = [term for term in saved_vocab[2:] if term]

    seq_len = int(app_manifest.payload["sequence_length"])
    return TextVectorization(
        output_mode="int",
        output_sequence_length=seq_len,
        vocabulary=vocab,
    )


def load_metrics(path: Path = METRICS_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    return df


def predict_sentiment(review_text: str) -> tuple[float, str]:
    model = load_model()
    vectorizer = load_vectorizer()
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


st.set_page_config(page_title="Federated Sentiment Dashboard", layout="wide")

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True
if "refresh_seconds" not in st.session_state:
    st.session_state.refresh_seconds = DEFAULT_REFRESH_SECONDS
if "last_signature" not in st.session_state:
    st.session_state.last_signature = None
if "idle_cycles" not in st.session_state:
    st.session_state.idle_cycles = 0

current_signature = file_signature(METRICS_PATH)
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

    st.caption(f"Artifact dir: {ARTIFACT_DIR}")
    st.caption(f"Metrics file: {METRICS_PATH}")
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

    df_metrics = load_metrics()
    if df_metrics is None:
        st.info(
            "Noch keine Trainingsmetriken gefunden. Wenn gerade ein neuer Lauf startet, "
            "ist das am Anfang normal."
        )
    else:
        st.caption(f"Loaded {len(df_metrics)} rows from {METRICS_PATH}")
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
        df_client_metrics = load_metrics(CLIENT_METRICS_PATH)
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
