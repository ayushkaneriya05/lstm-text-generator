"""Seed-based text generation with temperature sampling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import tensorflow as tf

from src.config import (
    DEFAULT_GENERATION_LENGTH,
    DEFAULT_TEMPERATURE,
    MODEL_PATH,
    VOCAB_PATH,
)
from src.data_preprocessing import load_vocabulary


def sample_from_probabilities(probabilities: np.ndarray, temperature: float) -> int:
    """Sample a token id after applying temperature scaling."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0.")

    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, None)
    logits = np.log(probabilities) / temperature
    logits -= np.max(logits)
    scaled = np.exp(logits)
    scaled /= np.sum(scaled)

    return int(np.random.choice(len(scaled), p=scaled))


def generate_text(
    model: tf.keras.Model,
    seed_text: str,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    sequence_length: int,
    num_words: int = DEFAULT_GENERATION_LENGTH,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Generate text iteratively from a seed prompt."""
    if not seed_text.strip():
        raise ValueError("seed_text cannot be empty.")
    if num_words < 1:
        raise ValueError("num_words must be at least 1.")

    unk_id = word_to_id["<UNK>"]
    generated_tokens = seed_text.lower().split()

    for _ in range(num_words):
        context_tokens = generated_tokens[-sequence_length:]
        encoded = [word_to_id.get(token, unk_id) for token in context_tokens]

        if len(encoded) < sequence_length:
            encoded = [word_to_id.get("<PAD>", 0)] * (sequence_length - len(encoded)) + encoded

        model_input = np.asarray([encoded], dtype=np.int32)
        probabilities = model.predict(model_input, verbose=0)[0]
        next_id = sample_from_probabilities(probabilities, temperature)
        next_word = id_to_word.get(next_id, "<UNK>")

        generated_tokens.append(next_word)

    return " ".join(generated_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Shakespeare-style text with an LSTM.")
    parser.add_argument("--seed", required=True, help="Seed text, e.g. 'to be or not to be'.")
    parser.add_argument("--words", type=int, default=DEFAULT_GENERATION_LENGTH)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found at {args.model}. Train it first.")

    metadata = load_vocabulary(args.vocab)
    model = tf.keras.models.load_model(args.model)

    result = generate_text(
        model=model,
        seed_text=args.seed,
        word_to_id=metadata["word_to_id"],
        id_to_word=metadata["id_to_word"],
        sequence_length=metadata["sequence_length"],
        num_words=args.words,
        temperature=args.temperature,
    )
    print(result)


if __name__ == "__main__":
    main()
