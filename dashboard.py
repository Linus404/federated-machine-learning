from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras.layers import TextVectorization
import tensorflow as tf

from src.local_training import sequence_length
from src.paths import (
    default_artifact_dir,
    default_data_dir,
    global_model_path,
    metrics_path,
    vocab_path,
)


DATA_DIR: Path = default_data_dir()
VOCAB_PATH: Path = vocab_path(DATA_DIR)
ARTIFACT_DIR: Path = default_artifact_dir()
MODEL_PATH: Path = global_model_path(ARTIFACT_DIR)
METRICS_PATH: Path = metrics_path(ARTIFACT_DIR)
DEFAULT_REFRESH_SECONDS = 6
IDLE_STOP_AFTER = 7


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Kein Modell gefunden unter {MODEL_PATH}. "
            "Bitte zuerst (federated) Training starten, damit der Server "
            "global_model.keras speichert."
        )
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_vectorizer():
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Keine Vokabular-Datei gefunden unter {VOCAB_PATH}. "
            "Bitte zuerst Stage 1 Datenvorbereitung ausführen."
        )

    saved_vocab = VOCAB_PATH.read_text(encoding="utf-8").splitlines()
    # TextVectorization adds "" and "[UNK]" itself when a vocabulary is supplied.
    vocab = [term for term in saved_vocab[2:] if term]

    seq_len = sequence_length(DATA_DIR, partition=0)
    return TextVectorization(
        output_mode="int",
        output_sequence_length=seq_len,
        vocabulary=vocab,
    )


def load_metrics() -> pd.DataFrame | None:
    if not METRICS_PATH.exists():
        return None
    try:
        df = pd.read_csv(METRICS_PATH)
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

if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_seconds)
    st.rerun()
