# =============================================================================
# config.py — V3: Drug Sensitivity + Tissue/Cancer-Type Biological Features
# =============================================================================

import os
import torch

# ─── PATHS ───────────────────────────────────────────────────────────────────
DATA_DIR    = r"D:\Hareezzvijey\genepromdl\data"
RESULTS_DIR = r"D:\Hareezzvijey\genepromdl\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GDSC1_PATH       = os.path.join(DATA_DIR, "GDSC_DATASET.csv")
GDSC2_PATH       = os.path.join(DATA_DIR, "GDSC2-dataset.csv")
MODEL_CSV_PATH   = os.path.join(DATA_DIR, "Model.csv")
METH_PATH        = os.path.join(DATA_DIR, "CCLE_RRBS_TSS_1kb_20180614.txt")
CELL_DETAILS_PATH = os.path.join(DATA_DIR, "Cell_Lines_Details.xlsx")

SMILES_CACHE     = os.path.join(DATA_DIR, "smiles_cache.pkl")
DRUG_FPS_CACHE   = os.path.join(DATA_DIR, "drug_fps.pkl")
SEQ_CACHE        = os.path.join(DATA_DIR, "seq_cache.pkl")
GENE_OHE_CACHE   = os.path.join(DATA_DIR, "gene_ohe.pkl")

V3_TABLE_PATH    = os.path.join(DATA_DIR, "training_table_v3.csv")
X_DRUG_V3_PATH   = os.path.join(DATA_DIR, "X_drug_v3.dat")
X_SEQ_V3_PATH    = os.path.join(DATA_DIR, "X_seq_v3.dat")
X_METH_V3_PATH   = os.path.join(DATA_DIR, "X_meth_v3.npy")
X_TISSUE_V3_PATH = os.path.join(DATA_DIR, "X_tissue_v3.npy")
TISSUE_ENCODER_PATH = os.path.join(DATA_DIR, "tissue_encoder_v3.pkl")
Y_V3_PATH        = os.path.join(DATA_DIR, "y_v3.npy")
SHAPES_V3_PATH   = os.path.join(DATA_DIR, "array_shapes_v3.pkl")
MODEL_PATH       = os.path.join(RESULTS_DIR, "best_model_v3.pt")

# ─── GPU CONFIG ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── TASK CONFIG ──────────────────────────────────────────────────────────────
TARGET_COLUMN   = "Z_SCORE"
TASK_TYPE       = "regression"

# ─── DATA BUILD CONFIG ────────────────────────────────────────────────────────
N_GENES_USE       = 100
SAMPLE_FRACTION   = 0.10
RANDOM_SEED       = 42

# ─── TRAINING HYPERPARAMETERS ────────────────────────────────────────────────
BATCH_SIZE     = 512
NUM_WORKERS    = 0
PIN_MEMORY     = True
LEARNING_RATE  = 1e-4
EPOCHS         = 50
PATIENCE       = 5
LR_PATIENCE    = 3
LR_FACTOR      = 0.5
MIN_LR         = 1e-7

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

# TISSUE_DIM is loaded automatically from the shapes file built in Step 0.
# Falls back to None if Step 0 hasn't been run yet (only 04_train_v3.py etc need this).
try:
    import pickle as _pickle
    with open(SHAPES_V3_PATH, "rb") as _f:
        _shapes = _pickle.load(_f)
    TISSUE_DIM = _shapes["tissue_dim"]
except FileNotFoundError:
    TISSUE_DIM = None  # Step 0 not run yet

# ─── DATASET SPLIT ───────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15