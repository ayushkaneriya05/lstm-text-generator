"""Train the baseline LSTM text generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    DATA_PATH,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    EMBEDDING_DIM,
    HISTORY_PATH,
    LSTM_UNITS,
    MAX_VOCAB_SIZE,
    MODEL_PATH,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TRAIN_SPLIT,
    VOCAB_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from src.data_preprocessing import prepare_datasets, save_vocabulary, set_random_seed
from src.model import build_baseline_model


def save_history_plot(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save loss and accuracy plots as a single figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse categorical crossentropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_model(
    data_path: Path = DATA_PATH,
    sequence_length: int = SEQUENCE_LENGTH,
    max_vocab_size: int = MAX_VOCAB_SIZE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[tf.keras.Model, dict, tf.keras.callbacks.History]:
    """Prepare data, train the baseline model and persist artifacts."""
    set_random_seed()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = prepare_datasets(
        data_path=data_path,
        train_split=TRAIN_SPLIT,
        sequence_length=sequence_length,
        max_vocab_size=max_vocab_size,
    )

    save_vocabulary(
        VOCAB_PATH,
        dataset["word_to_id"],
        dataset["id_to_word"],
        sequence_length,
    )

    print(f"Vocabulary size: {dataset['vocab_size']:,}")
    print(f"Training sequences: {len(dataset['x_train']):,}")
    print(f"Validation sequences: {len(dataset['x_val']):,}")

    model = build_baseline_model(
        vocab_size=dataset["vocab_size"],
        sequence_length=sequence_length,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
    )
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        dataset["x_train"],
        dataset["y_train"],
        validation_data=(dataset["x_val"], dataset["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1,
    )

    save_history_plot(history, HISTORY_PATH)

    # Save the restored best weights even if checkpointing was interrupted.
    model.save(MODEL_PATH)

    return model, dataset, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LSTM Shakespeare text generator.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--max-vocab-size", type=int, default=MAX_VOCAB_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        data_path=args.data,
        sequence_length=args.sequence_length,
        max_vocab_size=args.max_vocab_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
