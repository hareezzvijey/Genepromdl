# =============================================================================
# config.py — v2 (Corrected fixes — training dynamics, not data)
# Changes from v1:
#   REMOVED: POS_WEIGHT (data is balanced ~52/48, not needed)
#   CHANGED: LEARNING_RATE 1e-4 → 5e-4 (model too deep, needs larger steps)
#   CHANGED: DROPOUT 0.25 → 0.10 (reduce noise during early learning)
#   CHANGED: FUSION_DROPOUT 0.35 → 0.10 (same reason)
#   CHANGED: ATTN_HEADS 2 → 4 (sync with model — was mismatched)
#   CHANGED: EMBED_DIM 32 → 16 (sync with model — was mismatched)
#   CHANGED: PIN_MEMORY True (faster GPU transfer on CUDA)
#   NOTE: NUM_WORKERS stays 0 on Windows (multiprocessing issues)
# =============================================================================

import os
import torch

# ─── PATHS ───────────────────────────────────────────────────────────────────
DATA_DIR    = r"D:\Hareezzvijey\genepromdl\data"
RESULTS_DIR = r"D:\Hareezzvijey\genepromdl\data\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

X_DRUG_PATH  = os.path.join(DATA_DIR, "X_drug.dat")
X_SEQ_PATH   = os.path.join(DATA_DIR, "X_seq.dat")
Y_PATH       = os.path.join(DATA_DIR, "y.npy")
SHAPES_PATH  = os.path.join(DATA_DIR, "array_shapes.pkl")
TABLE_PATH   = os.path.join(DATA_DIR, "training_table.csv")
MODEL_PATH   = os.path.join(RESULTS_DIR, "best_model.pt")

# ─── GPU ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── TRAINING ────────────────────────────────────────────────────────────────
BATCH_SIZE     = 64
NUM_WORKERS    = 0
PIN_MEMORY     = True          # FIX: faster CPU→GPU transfer
LEARNING_RATE  = 5e-4          # FIX: was 1e-4, too slow for this depth
EPOCHS         = 50
PATIENCE       = 7
LR_PATIENCE    = 3
LR_FACTOR      = 0.5
MIN_LR         = 1e-7
RANDOM_SEED    = 42

# ─── MODEL ───────────────────────────────────────────────────────────────────
DRUG_DIM       = 2048
SEQ_LEN        = 1200
N_BASES        = 4
EMBED_DIM      = 16            # FIX: synced with model (was 32 in config, 16 in model)
CONV_FILTERS   = [64, 128, 256]
DILATION_RATES = [1, 2, 4]
ATTN_HEADS     = 4             # FIX: synced with model (was 2 in config, 4 in model)
ATTN_KEY_DIM   = 64
DENSE_UNITS    = [512, 256]
FUSION_UNITS   = [256, 64]
DROPOUT        = 0.10          # FIX: was 0.25, reduce during early learning
FUSION_DROPOUT = 0.10          # FIX: was 0.35, too aggressive at init

# ─── DATASET ─────────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# ─── CELL LINE INFO (NEW) ───────────────────────────────────────
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(TABLE_PATH)

le = LabelEncoder()
df["cell_line_id"] = le.fit_transform(df["cell_line"])

NUM_CELL_LINES = df["cell_line_id"].nunique()