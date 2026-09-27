"""Keras model construction."""

from __future__ import annotations

import tensorflow as tf

from src.config import DROPOUT_RATE, EMBEDDING_DIM, LSTM_UNITS


def build_baseline_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int = EMBEDDING_DIM,
    lstm_units: int = LSTM_UNITS,
    dropout_rate: float = DROPOUT_RATE,
) -> tf.keras.Model:
    """Build the baseline Embedding -> LSTM -> Dropout -> Softmax model."""
    if vocab_size <= 2:
        raise ValueError("vocab_size must be greater than 2.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length,), dtype=tf.int32),
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="embedding",
            ),
            tf.keras.layers.LSTM(lstm_units, name="lstm"),
            tf.keras.layers.Dropout(dropout_rate, name="dropout"),
            tf.keras.layers.Dense(vocab_size, activation="softmax", name="output"),
        ],
        name="shakespeare_lstm_generator",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def build_deeper_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int = EMBEDDING_DIM,
    first_lstm_units: int = LSTM_UNITS,
    second_lstm_units: int = 128,
    dropout_rate: float = DROPOUT_RATE,
) -> tf.keras.Model:
    """Build a two-LSTM experiment model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length,), dtype=tf.int32),
            tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding"),
            tf.keras.layers.LSTM(
                first_lstm_units,
                return_sequences=True,
                name="lstm_1",
            ),
            tf.keras.layers.Dropout(dropout_rate, name="dropout_1"),
            tf.keras.layers.LSTM(second_lstm_units, name="lstm_2"),
            tf.keras.layers.Dropout(dropout_rate, name="dropout_2"),
            tf.keras.layers.Dense(vocab_size, activation="softmax", name="output"),
        ],
        name="shakespeare_deeper_lstm_generator",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
