# =============================================================================
# dataset_module_v3.py — PyTorch Dataset for V3
# Four inputs: drug fingerprint, DNA sequence, methylation, tissue features
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


class GenePromDLDatasetV3(Dataset):
    def __init__(self, x_drug_path, x_seq_path, x_meth, x_tissue, y, indices, shapes_path):
        with open(shapes_path, "rb") as f:
            shapes = pickle.load(f)
        N = shapes["n_samples"]
        self.X_drug   = np.memmap(x_drug_path, dtype="uint8", mode="r", shape=(N, DRUG_DIM))
        self.X_seq    = np.memmap(x_seq_path,  dtype="uint8", mode="r", shape=(N, SEQ_LEN, N_BASES))
        self.X_meth   = x_meth
        self.X_tissue = x_tissue
        self.y        = y
        self.indices  = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        drug   = torch.from_numpy(self.X_drug[i].astype(np.float32))
        seq    = torch.from_numpy(self.X_seq[i].astype(np.float32)).permute(1, 0)
        meth   = torch.tensor(float(self.X_meth[i]), dtype=torch.float32)
        tissue = torch.from_numpy(self.X_tissue[i].astype(np.float32))
        target = torch.tensor(float(self.y[i]), dtype=torch.float32)
        return drug, seq, meth, tissue, target


def get_dataloaders_v3(data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY):
    y         = np.load(Y_V3_PATH)
    x_meth    = np.load(X_METH_V3_PATH)
    x_tissue  = np.load(X_TISSUE_V3_PATH)
    train_idx = np.load(os.path.join(data_dir, "train_idx_v3.npy"))
    val_idx   = np.load(os.path.join(data_dir, "val_idx_v3.npy"))
    test_idx  = np.load(os.path.join(data_dir, "test_idx_v3.npy"))

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=pin_memory and torch.cuda.is_available())

    train_ds = GenePromDLDatasetV3(X_DRUG_V3_PATH, X_SEQ_V3_PATH, x_meth, x_tissue, y, train_idx, SHAPES_V3_PATH)
    val_ds   = GenePromDLDatasetV3(X_DRUG_V3_PATH, X_SEQ_V3_PATH, x_meth, x_tissue, y, val_idx,   SHAPES_V3_PATH)
    test_ds  = GenePromDLDatasetV3(X_DRUG_V3_PATH, X_SEQ_V3_PATH, x_meth, x_tissue, y, test_idx,  SHAPES_V3_PATH)

    train_dl = DataLoader(train_ds, shuffle=True,  drop_last=True, **kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, **kw)
    test_dl  = DataLoader(test_ds,  shuffle=False, **kw)
    return train_dl, val_dl, test_dl