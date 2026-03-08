# config.py

from pathlib import Path

# Racine du projet (là où est gpt_config.py)
ROOT = Path(__file__).parent

# Chemins
DATA_PATH = ROOT / "data" / "input.txt"
MODEL_PATH = ROOT / "best_model.pth"
block_size = 256
batch_size = 32
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
learning_rate = 1e-3
max_iters = 2000
vocab_size = 65 