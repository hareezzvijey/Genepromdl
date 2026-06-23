# =============================================================================
# dataset_module_v2.py — FINAL VERSION (FAST + CORRECT)
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


class GenePromDLDatasetV2(Dataset):
    def __init__(self, x_drug, x_seq, x_meth, y, indices):
        self.X_drug = x_drug
        self.X_seq  = x_seq
        self.X_meth = x_meth
        self.y      = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        drug = torch.from_numpy(self.X_drug[i]).float()
        seq  = torch.from_numpy(self.X_seq[i]).float().permute(1, 0)  # (4, 1200)
        meth = torch.tensor(self.X_meth[i], dtype=torch.float32)
        target = torch.tensor(self.y[i], dtype=torch.float32)

        return drug, seq, meth, target


def get_dataloaders_v2(data_dir=DATA_DIR,
                       batch_size=BATCH_SIZE,
                       num_workers=NUM_WORKERS,
                       pin_memory=PIN_MEMORY):

    # ─────────────────────────────────────────────
    # LOAD SHAPES
    # ─────────────────────────────────────────────
    with open(SHAPES_V2_PATH, "rb") as f:
        shapes = pickle.load(f)

    N = shapes["n_samples"]

    # ─────────────────────────────────────────────
    # LOAD MEMMAP ONCE (IMPORTANT)
    # ─────────────────────────────────────────────
    X_drug = np.memmap(X_DRUG_V2_PATH, dtype="uint8", mode="r", shape=(N, DRUG_DIM))
    X_seq  = np.memmap(X_SEQ_V2_PATH,  dtype="uint8", mode="r", shape=(N, SEQ_LEN, N_BASES))
    X_meth = np.memmap(X_METH_V2_PATH, dtype="float32", mode="r", shape=(N,))
    y      = np.memmap(Y_V2_PATH,      dtype="float32", mode="r", shape=(N,))

    # ─────────────────────────────────────────────
    # LOAD SPLITS
    # ─────────────────────────────────────────────
    train_idx = np.load(os.path.join(data_dir, "train_idx_v2.npy"))
    val_idx   = np.load(os.path.join(data_dir, "val_idx_v2.npy"))
    test_idx  = np.load(os.path.join(data_dir, "test_idx_v2.npy"))

    # ─────────────────────────────────────────────
    # DATASETS
    # ─────────────────────────────────────────────
    train_ds = GenePromDLDatasetV2(X_drug, X_seq, X_meth, y, train_idx)
    val_ds   = GenePromDLDatasetV2(X_drug, X_seq, X_meth, y, val_idx)
    test_ds  = GenePromDLDatasetV2(X_drug, X_seq, X_meth, y, test_idx)

    # ─────────────────────────────────────────────
    # DATALOADERS
    # ─────────────────────────────────────────────
    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )

    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_dl   = DataLoader(val_ds, shuffle=False, **kw)
    test_dl  = DataLoader(test_ds, shuffle=False, **kw)

    return train_dl, val_dl, test_dl