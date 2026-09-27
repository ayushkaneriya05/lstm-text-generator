# LSTM Shakespeare Text Generator

A complete implementation of the hiring task **"Generative AI with LSTM - Text Generation"**.
The project trains an LSTM language model on a public-domain Shakespeare corpus and generates new text from a seed prompt.

## 1. Task Coverage

This repository implements the requested pipeline:

- Load a large `.txt` text corpus.
- Lowercase and remove punctuation.
- Tokenize at the word level.
- Create next-token input/target sequences.
- Split the token stream into training and validation portions before creating sliding windows.
- Use an Embedding + LSTM + Dense/Softmax architecture.
- Train with Adam and sparse categorical crossentropy.
- Use validation data, early stopping, and checkpointing.
- Generate text iteratively from a seed prompt.
- Support temperature sampling.
- Save model/vocabulary artifacts for reproducible generation.
- Run bonus experiments with a deeper LSTM and a longer sequence length.

## 2. Dataset

The dataset is Shakespeare's complete works from Project Gutenberg:

https://www.gutenberg.org/files/100/100-0.txt

The project includes a small download helper. Run:

```bash
python download_dataset.py
```

The resulting file should be:

```text
data/shakespeare.txt
```

## 3. Project Structure

```text
lstm-text-generator/
├── data/
│   ├── README.md
│   └── shakespeare.txt        # created by download_dataset.py
│
├── models/
│   ├── best_model.keras       # created after training
│   ├── tokenizer.pkl          # created after training
│   └── experiments/           # created by bonus experiments
│
├── outputs/
│   ├── training_history.png   # created after training
│   ├── generated_samples.txt  # create from sample prompts
│   └── experiment_results.csv # created after experiments
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── experiments.py
│
├── download_dataset.py
├── main.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 4. Environment Setup

Recommended Python version: **3.11**.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 5. Download the Dataset

```bash
python download_dataset.py
```

## 6. Train the Baseline Model

From the project root:

```bash
python -m src.train
```

The baseline uses:

| Hyperparameter | Value |
|---|---:|
| Train split | 90% |
| Validation split | 10% |
| Sequence length | 30 |
| Maximum vocabulary | 20,000 |
| Embedding dimension | 128 |
| LSTM units | 256 |
| Dropout | 0.20 |
| Batch size | 128 |
| Maximum epochs | 20 |
| Early stopping patience | 3 |
| Optimizer | Adam |
| Loss | Sparse categorical crossentropy |

The split happens **before** sequence generation to reduce leakage from overlapping sliding windows.

## 7. Baseline Architecture

```text
Input sequence (30 word IDs)
          │
          ▼
Embedding (128)
          │
          ▼
LSTM (256)
          │
          ▼
Dropout (0.20)
          │
          ▼
Dense (vocab_size, softmax)
          │
          ▼
Next-word probability distribution
```

## 8. Generate Text

After training:

```bash
python -m src.generate --seed "to be or not to be" --words 50 --temperature 0.8
```

Try additional seeds:

```bash
python -m src.generate --seed "the king" --words 50 --temperature 0.8
python -m src.generate --seed "love is" --words 50 --temperature 0.8
python -m src.generate --seed "what is the" --words 50 --temperature 0.8
```

### Temperature

Temperature controls sampling diversity:

- `0.5` → more deterministic and conservative.
- `0.8` → balanced starting point.
- `1.0` → standard sampling scale.
- `1.2` → more diverse/random output.

The actual best setting should be judged from the generated samples rather than assumed in advance.

## 9. Bonus Experiments

Run:

```bash
python -m src.experiments
```

Experiments included:

1. Baseline, sequence length 30.
2. Deeper model, sequence length 30, using two LSTM layers.
3. Baseline model with sequence length 50.

Results are written to:

```text
outputs/experiment_results.csv
```

The table records validation loss/accuracy, parameter count, and training time so the architecture trade-offs can be discussed from actual measurements.

## 10. Generated Samples

For the final hiring submission, create `outputs/generated_samples.txt` containing outputs from at least three to five seed prompts. Keep the exact seed, temperature, number of generated words, and generated result together.

A suggested format:

```text
============================================================
Seed: to be or not to be
Temperature: 0.8
Words generated: 50
------------------------------------------------------------
<actual model output>

============================================================
Seed: the king
Temperature: 0.8
Words generated: 50
------------------------------------------------------------
<actual model output>
```

Do not insert fabricated output. Generate this file from the trained model.

## 11. Training Output

Training generates:

```text
models/best_model.keras
models/tokenizer.pkl
outputs/training_history.png
```

The plot shows training vs. validation loss and accuracy.

## 12. Problem-Solving Decisions

### Validation leakage

The raw token stream is split before sequence windows are created. This is preferable to randomly splitting overlapping windows because adjacent sequences share most of their tokens.

### Unknown words

The vocabulary contains `<UNK>`. Validation or generation words that are not in the training vocabulary map to that token.

### Sampling diversity

Greedy `argmax` generation is avoided in the normal generation path. Temperature-based sampling allows controlled randomness.

### Reproducibility

A fixed random seed is used in NumPy and TensorFlow for repeatable experiments where the runtime allows deterministic behavior.

