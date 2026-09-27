"""Data loading, cleaning, vocabulary building and sequence preparation."""

from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config import MAX_VOCAB_SIZE, RANDOM_SEED, SPECIAL_TOKENS


def load_text(path: Path) -> str:
    """Load UTF-8 text from disk with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it first; see data/README.md."
        )
    return path.read_text(encoding="utf-8")


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation and normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize cleaned text with whitespace-based word tokenization."""
    if not text:
        return []
    return text.split()


def split_tokens(tokens: list[str], train_split: float) -> tuple[list[str], list[str]]:
    """Split token stream before creating sliding windows to reduce leakage."""
    if not 0 < train_split < 1:
        raise ValueError("train_split must be between 0 and 1.")
    split_index = int(len(tokens) * train_split)
    if split_index < 2 or len(tokens) - split_index < 2:
        raise ValueError("Dataset is too small for the requested train/validation split.")
    return tokens[:split_index], tokens[split_index:]


def build_vocabulary(tokens: Iterable[str], max_vocab_size: int) -> tuple[dict[str, int], dict[int, str]]:
    """Build word-to-id and id-to-word mappings with PAD and UNK tokens."""
    if max_vocab_size < 2:
        raise ValueError("max_vocab_size must be at least 2.")

    counts = Counter(tokens)
    word_to_id = dict(SPECIAL_TOKENS)

    # Reserve ids 0 and 1 for PAD and UNK.
    most_common = counts.most_common(max_vocab_size - len(SPECIAL_TOKENS))
    next_id = len(SPECIAL_TOKENS)
    for word, _ in most_common:
        if word not in word_to_id:
            word_to_id[word] = next_id
            next_id += 1

    id_to_word = {idx: word for word, idx in word_to_id.items()}
    return word_to_id, id_to_word


def encode_tokens(tokens: list[str], word_to_id: dict[str, int]) -> np.ndarray:
    """Convert words into integer ids, mapping unseen words to <UNK>."""
    unk_id = word_to_id["<UNK>"]
    return np.asarray([word_to_id.get(token, unk_id) for token in tokens], dtype=np.int32)


def create_sequences(encoded_tokens: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-length next-token training examples."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if len(encoded_tokens) <= sequence_length:
        raise ValueError("Not enough tokens to create sequences.")

    num_sequences = len(encoded_tokens) - sequence_length
    x = np.empty((num_sequences, sequence_length), dtype=np.int32)
    y = np.empty((num_sequences,), dtype=np.int32)

    for i in range(num_sequences):
        x[i] = encoded_tokens[i : i + sequence_length]
        y[i] = encoded_tokens[i + sequence_length]

    return x, y


def save_vocabulary(path: Path, word_to_id: dict[str, int], id_to_word: dict[int, str], sequence_length: int) -> None:
    """Persist vocabulary metadata for reproducible generation."""
    payload = {
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "sequence_length": sequence_length,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(payload, file)


def load_vocabulary(path: Path) -> dict:
    """Load persisted vocabulary metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


def prepare_datasets(
    data_path: Path,
    train_split: float,
    sequence_length: int,
    max_vocab_size: int = MAX_VOCAB_SIZE,
) -> dict:
    """Run the complete preprocessing pipeline and return arrays + metadata."""
    raw_text = load_text(data_path)
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    train_tokens, val_tokens = split_tokens(tokens, train_split)

    word_to_id, id_to_word = build_vocabulary(train_tokens, max_vocab_size)
    train_encoded = encode_tokens(train_tokens, word_to_id)
    val_encoded = encode_tokens(val_tokens, word_to_id)

    x_train, y_train = create_sequences(train_encoded, sequence_length)
    x_val, y_val = create_sequences(val_encoded, sequence_length)

    return {
        "raw_text_length": len(raw_text),
        "cleaned_word_count": len(tokens),
        "train_word_count": len(train_tokens),
        "val_word_count": len(val_tokens),
        "vocab_size": len(word_to_id),
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
    }


def set_random_seed() -> None:
    """Set NumPy's seed; TensorFlow's seed is set in train.py/model.py."""
    np.random.seed(RANDOM_SEED)
