# dataset_module.py — imported by train, evaluate, ablation scripts
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


class GenePromDLDataset(Dataset):
    def __init__(self, x_drug_path, x_seq_path, y, indices, shapes_path):
        with open(shapes_path, "rb") as f:
            shapes = pickle.load(f)
        N = shapes["n_samples"]
        self.X_drug  = np.memmap(x_drug_path, dtype="uint8", mode="r", shape=(N, 2048))
        self.X_seq   = np.memmap(x_seq_path,  dtype="uint8", mode="r", shape=(N, 1200, 4))
        self.y       = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i    = self.indices[idx]
        drug = torch.from_numpy(self.X_drug[i].astype(np.float32))
        seq  = torch.from_numpy(self.X_seq[i].astype(np.float32)).permute(1, 0)
        lbl  = torch.tensor(float(self.y[i]), dtype=torch.float32)
        return drug, seq, lbl


def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY):
    y         = np.load(os.path.join(data_dir, "y.npy"))
    train_idx = np.load(os.path.join(data_dir, "train_idx.npy"))
    val_idx   = np.load(os.path.join(data_dir, "val_idx.npy"))
    test_idx  = np.load(os.path.join(data_dir, "test_idx.npy"))

    xd = os.path.join(data_dir, "X_drug.dat")
    xs = os.path.join(data_dir, "X_seq.dat")
    sh = os.path.join(data_dir, "array_shapes.pkl")

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=pin_memory and torch.cuda.is_available())

    train_dl = DataLoader(GenePromDLDataset(xd, xs, y, train_idx, sh),
                          shuffle=True,  **kw, drop_last=True)
    val_dl   = DataLoader(GenePromDLDataset(xd, xs, y, val_idx,   sh),
                          shuffle=False, **kw)
    test_dl  = DataLoader(GenePromDLDataset(xd, xs, y, test_idx,  sh),
                          shuffle=False, **kw)
    return train_dl, val_dl, test_dl