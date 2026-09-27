from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

DATA_PATH = DATA_DIR / "shakespeare.txt"
MODEL_PATH = MODELS_DIR / "best_model.keras"
VOCAB_PATH = MODELS_DIR / "tokenizer.pkl"
HISTORY_PATH = OUTPUTS_DIR / "training_history.png"
SAMPLES_PATH = OUTPUTS_DIR / "generated_samples.txt"
EXPERIMENTS_PATH = OUTPUTS_DIR / "experiment_results.csv"

RANDOM_SEED = 42
TRAIN_SPLIT = 0.90
SEQUENCE_LENGTH = 30
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
LSTM_UNITS = 256
DROPOUT_RATE = 0.20
BATCH_SIZE = 128
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
DEFAULT_GENERATION_LENGTH = 50
DEFAULT_TEMPERATURE = 0.80

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
}
