"""Shared text preprocessing for the frozen scientific protocol."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from typing import Any

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(
    package="federated_imdb", name="protocol_standardize"
)
def protocol_standardize(text: Any) -> Any:
    """Apply the frozen HTML, casing, and punctuation standardization.

    Parameters
    ----------
    text : Any
        TensorFlow string tensor accepted by ``tf.strings`` operations.

    Returns
    -------
    Any
        Standardized TensorFlow string tensor with Unicode code points preserved.
    """
    text = tf.strings.regex_replace(text, "<[^>]+>", " ")
    text = tf.strings.lower(text)
    return tf.strings.regex_replace(text, "[" + re.escape(string.punctuation) + "]", "")


def create_text_vectorizer(
    *,
    sequence_length: int,
    max_tokens: int | None = None,
    vocabulary: Sequence[str] | None = None,
) -> keras.layers.TextVectorization:
    """Construct a vectorizer that implements the frozen token-ID protocol.

    Parameters
    ----------
    sequence_length : int
        Fixed number of token IDs emitted for each input.
    max_tokens : int or None, optional
        Maximum vocabulary size used during adaptation.
    vocabulary : sequence of str or None, optional
        Learned terms excluding the padding and unknown reserved tokens.

    Returns
    -------
    keras.layers.TextVectorization
        Unadapted producer vectorizer or vocabulary-bound consumer vectorizer.
    """
    return keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=protocol_standardize,
        split="whitespace",
        output_mode="int",
        output_sequence_length=sequence_length,
        pad_to_max_tokens=False,
        vocabulary=vocabulary,
    )
