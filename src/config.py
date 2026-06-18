# =============================================================================
# config.py — ALL PATHS AND HYPERPARAMETERS IN ONE PLACE
# Edit this file only. All other scripts import from here.
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

# ─── GPU CONFIG ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── TRAINING HYPERPARAMETERS ────────────────────────────────────────────────
BATCH_SIZE     = 64     # RTX 3050 4GB VRAM — safe batch size
NUM_WORKERS    = 0
PIN_MEMORY     = False     
LEARNING_RATE  = 1e-4
EPOCHS         = 50
PATIENCE       = 5        # EarlyStopping patience
LR_PATIENCE    = 3        # ReduceLROnPlateau patience
LR_FACTOR      = 0.5
MIN_LR         = 1e-7
RANDOM_SEED    = 42

# ─── MODEL HYPERPARAMETERS ───────────────────────────────────────────────────
DRUG_DIM       = 2048
SEQ_LEN        = 1200
N_BASES        = 4
EMBED_DIM      = 32
CONV_FILTERS   = [64, 128, 256]
DILATION_RATES = [1, 2, 4]
ATTN_HEADS     = 2
ATTN_KEY_DIM   = 64
DENSE_UNITS    = [512, 256]
FUSION_UNITS   = [256, 64]
DROPOUT        = 0.25
FUSION_DROPOUT = 0.35

# ─── DATASET SPLIT ───────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15