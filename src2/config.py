# =============================================================================
# config.py — V2: Drug Sensitivity Prediction (using methylation as covariate)
# All paths and hyperparameters in one place.
# =============================================================================

import os
import torch

# ─── PATHS ───────────────────────────────────────────────────────────────────
DATA_DIR    = r"D:\Hareezzvijey\genepromdl\data2"
RESULTS_DIR = r"D:\Hareezzvijey\genepromdl\results_v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Original files (already have these)
GDSC1_PATH       = os.path.join(DATA_DIR, "GDSC_DATASET.csv")
GDSC2_PATH       = os.path.join(DATA_DIR, "GDSC2-dataset.csv")
COMPOUNDS_PATH   = os.path.join(DATA_DIR, "Compounds-annotation.csv")
MODEL_CSV_PATH   = os.path.join(DATA_DIR, "Model.csv")
METH_PATH        = os.path.join(DATA_DIR, "CCLE_RRBS_TSS_1kb_20180614.txt")

# Cached pickle files (already have these — reused, no re-fetching)
SMILES_CACHE     = os.path.join(DATA_DIR, "smiles_cache.pkl")
DRUG_FPS_CACHE   = os.path.join(DATA_DIR, "drug_fps.pkl")
SEQ_CACHE        = os.path.join(DATA_DIR, "seq_cache.pkl")
GENE_OHE_CACHE   = os.path.join(DATA_DIR, "gene_ohe.pkl")

# New V2 outputs
V2_TABLE_PATH    = os.path.join(DATA_DIR, "training_table_v2.csv")
X_DRUG_V2_PATH   = os.path.join(DATA_DIR, "X_drug_v2.dat")
X_SEQ_V2_PATH    = os.path.join(DATA_DIR, "X_seq_v2.dat")
X_METH_V2_PATH   = os.path.join(DATA_DIR, "X_meth_v2.npy")   # scalar methylation feature
Y_V2_PATH        = os.path.join(DATA_DIR, "y_v2.npy")
SHAPES_V2_PATH   = os.path.join(DATA_DIR, "array_shapes_v2.pkl")
MODEL_PATH       = os.path.join(RESULTS_DIR, "best_model_v2.pt")

# ─── GPU CONFIG ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── TASK CONFIG (V2: regression on drug sensitivity) ────────────────────────
TARGET_COLUMN   = "Z_SCORE"     # could also use "LN_IC50"
TASK_TYPE       = "regression"  # was "classification" in V1

# ─── DATA BUILD CONFIG ────────────────────────────────────────────────────────
N_GENES_USE       = 100          # same gene set as before
SAMPLE_FRACTION   = 0.10         # same sampling fraction as before
METH_THRESHOLD    = 0.6          # used only for the optional binary meth feature

# ─── TRAINING HYPERPARAMETERS ────────────────────────────────────────────────
BATCH_SIZE     = 128
NUM_WORKERS    = 0
PIN_MEMORY     = True
LEARNING_RATE  = 1e-4
EPOCHS         = 50
PATIENCE       = 5
LR_PATIENCE    = 3
LR_FACTOR      = 0.5
MIN_LR         = 1e-7
RANDOM_SEED    = 42

# ─── MODEL HYPERPARAMETERS ───────────────────────────────────────────────────
DRUG_DIM       = 2048
SEQ_LEN        = 1200
N_BASES        = 4
EMBED_DIM      = 16
CONV_FILTERS   = [64, 128, 256]
DILATION_RATES = [1, 2, 4]
ATTN_HEADS     = 4
ATTN_KEY_DIM   = 64
DENSE_UNITS    = [512, 256]
FUSION_UNITS   = [256, 64]
DROPOUT        = 0.3
FUSION_DROPOUT = 0.4

# ─── DATASET SPLIT ───────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15