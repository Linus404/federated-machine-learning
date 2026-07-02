from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras.layers import TextVectorization
import tensorflow as tf

from src.paths import default_data_dir, vocab_path
from src.local_training import sequence_length


DATA_DIR: Path = default_data_dir()
VOCAB_PATH: Path = vocab_path(DATA_DIR)
ARTIFACT_DIR: Path = Path("artifacts")
MODEL_PATH: Path = ARTIFACT_DIR / "global_model.keras"
METRICS_PATH: Path = ARTIFACT_DIR / "metrics.csv"


# --------- Hilfsfunktionen ---------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Kein Modell gefunden unter {MODEL_PATH}. "
            "Bitte zuerst (federated) Training starten, damit der Server "
            "globalmodel.keras speichert."
        )
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_vectorizer():
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Keine Vokabular-Datei gefunden unter {VOCAB_PATH}. "
            "Bitte zuerst Stage 1 Datenvorbereitung ausführen."
        )

    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]

    seq_len = sequence_length(DATA_DIR, partition=0)

    vectorizer = TextVectorization(
        output_mode="int",
        output_sequence_length=seq_len,
        vocabulary=vocab,
    )
    return vectorizer


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    df = pd.read_csv(METRICS_PATH)

    return df


def predict_sentiment(review_text: str) -> tuple[float, str]:
    """Berechne p(positiv) und Label."""
    model = load_model()
    vectorizer = load_vectorizer()

    inputs = tf.constant([review_text])
    token_ids = vectorizer(inputs)
    preds = model.predict(token_ids, verbose=0)
    positive_prob = float(preds[0][0])

    label = "positive" if positive_prob >= 0.5 else "negative"
    return positive_prob, label


# --------- Streamlit Layout ---------
st.set_page_config(
    page_title="Federated Sentiment Dashboard",
    layout="wide",
)

st.title("Review inference & Training metrics")

col_left, col_right = st.columns([1.1, 1.3])

# ----- Linke Seite: Review Inference -----
with col_left:
    st.subheader("Review inference")

    st.caption("Enter a movie review")
    default_text = "This movie was surprisingly thoughtful and fun to watch."
    review = st.text_area(
        label="Test",
        label_visibility="collapsed",   # Label verstecken
        value=default_text,
        height=200,
        placeholder="Type your review here...",
    )

    predict_button = st.button("Predict sentiment")

    if predict_button and review.strip():
        try:
            positive_prob, label = predict_sentiment(review)
            st.markdown("### Positive sentiment probability")
            st.markdown(f"**{positive_prob * 100:.1f}%**")
            st.markdown(f"**Prediction:** {label}")
        except Exception as e:
            st.error(f"Fehler bei der Vorhersage: {e}")

# ----- Rechte Seite: Trainingsmetriken -----
with col_right:
    st.subheader("Training metrics")

    df_metrics = load_metrics()
    if df_metrics is None:
        st.info(
            "Keine Trainingsmetriken gefunden. "
            "Stelle sicher, dass der Server gelaufen ist und "
            "`artifacts/metrics.csv` erzeugt wurde."
        )
    else:
        st.dataframe(
            df_metrics,
            width="stretch",
            hide_index=True,
        )

        if {"round", "loss", "accuracy"} <= set(df_metrics.columns):
            chart_data = df_metrics.set_index("round")[["accuracy", "loss"]]
            st.line_chart(chart_data)
        else:
            st.warning(
                "metrics.csv hat nicht die erwarteten Spalten "
                "('round', 'loss', 'accuracy')."
            )