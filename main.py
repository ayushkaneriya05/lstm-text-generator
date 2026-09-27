"""Interactive entry point for the hiring-task project."""

from __future__ import annotations

from pathlib import Path

from src.config import DATA_PATH, DEFAULT_GENERATION_LENGTH, DEFAULT_TEMPERATURE, MODEL_PATH, VOCAB_PATH
from src.data_preprocessing import load_vocabulary
from src.generate import generate_text
from src.train import train_model


def generate_from_saved_model() -> None:
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        print("A trained model is required. Train the model first.")
        return

    seed = input("Enter seed text: ").strip()
    if not seed:
        print("Seed text cannot be empty.")
        return

    try:
        words = int(input(f"Number of new words [{DEFAULT_GENERATION_LENGTH}]: ") or DEFAULT_GENERATION_LENGTH)
        temperature = float(input(f"Temperature [{DEFAULT_TEMPERATURE}]: ") or DEFAULT_TEMPERATURE)

        import tensorflow as tf

        metadata = load_vocabulary(VOCAB_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)
        text = generate_text(
            model=model,
            seed_text=seed,
            word_to_id=metadata["word_to_id"],
            id_to_word=metadata["id_to_word"],
            sequence_length=metadata["sequence_length"],
            num_words=words,
            temperature=temperature,
        )
        print("\nGenerated text:\n")
        print(text)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")


def main() -> None:
    while True:
        print("\n======================================")
        print(" LSTM Shakespeare Text Generator")
        print("======================================")
        print("1. Train model")
        print("2. Generate text")
        print("3. Run bonus experiments")
        print("4. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            train_model(data_path=DATA_PATH)
        elif choice == "2":
            generate_from_saved_model()
        elif choice == "3":
            from src.experiments import main as experiments_main

            experiments_main()
        elif choice == "4":
            print("Goodbye.")
            break
        else:
            print("Please select a valid option.")


if __name__ == "__main__":
    main()
