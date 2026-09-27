"""Generate a reproducible set of sample outputs for the final submission."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.config import DEFAULT_TEMPERATURE, MODEL_PATH, OUTPUTS_DIR, SAMPLES_PATH, VOCAB_PATH
from src.data_preprocessing import load_vocabulary
from src.generate import generate_text


SEEDS = [
    "to be or not to be",
    "the king",
    "love is",
    "what is the",
    "shall i compare",
]


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train the model first.")
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocabulary not found: {VOCAB_PATH}. Train the model first.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_vocabulary(VOCAB_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    lines: list[str] = []
    for seed in SEEDS:
        generated = generate_text(
            model=model,
            seed_text=seed,
            word_to_id=metadata["word_to_id"],
            id_to_word=metadata["id_to_word"],
            sequence_length=metadata["sequence_length"],
            num_words=50,
            temperature=DEFAULT_TEMPERATURE,
        )
        lines.extend(
            [
                "=" * 60,
                f"Seed: {seed}",
                f"Temperature: {DEFAULT_TEMPERATURE}",
                "Words generated: 50",
                "-" * 60,
                generated,
                "",
            ]
        )

    SAMPLES_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved sample outputs to: {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
