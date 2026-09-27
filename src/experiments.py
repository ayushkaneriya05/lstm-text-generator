"""Run bonus architecture/sequence-length experiments and save a comparison table."""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    EXPERIMENTS_PATH,
    LSTM_UNITS,
    MAX_VOCAB_SIZE,
    MODELS_DIR,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TRAIN_SPLIT,
    DATA_PATH,
    OUTPUTS_DIR,
    EMBEDDING_DIM,
    DROPOUT_RATE,
)
from src.data_preprocessing import prepare_datasets, set_random_seed
from src.model import build_baseline_model, build_deeper_model


@dataclass(frozen=True)
class Experiment:
    name: str
    sequence_length: int
    kind: str


EXPERIMENTS = [
    Experiment("baseline_seq30", 30, "baseline"),
    Experiment("deeper_seq30", 30, "deeper"),
    Experiment("baseline_seq50", 50, "baseline"),
]


def run_experiment(experiment: Experiment) -> dict:
    set_random_seed()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    dataset = prepare_datasets(
        data_path=DATA_PATH,
        train_split=TRAIN_SPLIT,
        sequence_length=experiment.sequence_length,
        max_vocab_size=MAX_VOCAB_SIZE,
    )

    if experiment.kind == "baseline":
        model = build_baseline_model(
            vocab_size=dataset["vocab_size"],
            sequence_length=experiment.sequence_length,
            embedding_dim=EMBEDDING_DIM,
            lstm_units=LSTM_UNITS,
            dropout_rate=DROPOUT_RATE,
        )
    elif experiment.kind == "deeper":
        model = build_deeper_model(
            vocab_size=dataset["vocab_size"],
            sequence_length=experiment.sequence_length,
            embedding_dim=EMBEDDING_DIM,
            first_lstm_units=LSTM_UNITS,
            second_lstm_units=128,
            dropout_rate=DROPOUT_RATE,
        )
    else:
        raise ValueError(f"Unknown experiment kind: {experiment.kind}")

    run_dir = MODELS_DIR / "experiments"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"{experiment.name}.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        dataset["x_train"],
        dataset["y_train"],
        validation_data=(dataset["x_val"], dataset["y_val"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    duration = time.perf_counter() - start

    best_idx = int(tf.math.argmin(history.history["val_loss"]).numpy())

    return {
        "experiment": experiment.name,
        "sequence_length": experiment.sequence_length,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_layers": 1 if experiment.kind == "baseline" else 2,
        "vocab_size": dataset["vocab_size"],
        "parameters": model.count_params(),
        "best_train_loss": history.history["loss"][best_idx],
        "best_val_loss": history.history["val_loss"][best_idx],
        "best_train_accuracy": history.history["accuracy"][best_idx],
        "best_val_accuracy": history.history["val_accuracy"][best_idx],
        "training_time_seconds": round(duration, 2),
    }


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for experiment in EXPERIMENTS:
        print(f"\n=== Running {experiment.name} ===")
        results.append(run_experiment(experiment))

    fieldnames = list(results[0].keys())
    with EXPERIMENTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved experiment results to: {EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()
