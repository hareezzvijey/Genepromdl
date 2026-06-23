# =============================================================================
# STEP 3 — PYTORCH DATASET AND DATALOADER
# File: src/03_dataset.py
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# 🔥 NEW: Load table + encode cell_line
df = pd.read_csv(TABLE_PATH)

le = LabelEncoder()
df["cell_line_id"] = le.fit_transform(df["cell_line"])

NUM_CELL_LINES = df["cell_line_id"].nunique()


class GenePromDLDataset(Dataset):

    def __init__(self, x_drug_path, x_seq_path, y, indices, shapes_path):
        with open(shapes_path, "rb") as f:
            shapes = pickle.load(f)

        N = shapes["n_samples"]

        self.drug_dim = 2048
        self.seq_len = 1200

        self.X_drug = np.memmap(
            x_drug_path, dtype="uint8", mode="r",
            shape=(N, self.drug_dim)
        )

        self.X_seq = np.memmap(
            x_seq_path, dtype="uint8", mode="r",
            shape=(N, self.seq_len, 4)
        )

        self.y = y
        self.indices = indices

        # 🔥 NEW: store cell ids
        self.cell_ids = df["cell_line_id"].values

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        drug = torch.from_numpy(self.X_drug[i].astype(np.float32))
        seq = torch.from_numpy(self.X_seq[i].astype(np.float32))
        seq = seq.permute(1, 0)

        label = torch.tensor(float(self.y[i]), dtype=torch.float32)

        cell = torch.tensor(self.cell_ids[i], dtype=torch.long)

        return drug, seq, cell, label


def get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
):

    y = np.load(os.path.join(data_dir, "y.npy"))

    train_idx = np.load(os.path.join(data_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(data_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(data_dir, "test_idx.npy"))

    x_drug_path = os.path.join(data_dir, "X_drug.dat")
    x_seq_path = os.path.join(data_dir, "X_seq.dat")
    shapes_path = os.path.join(data_dir, "array_shapes.pkl")

    train_ds = GenePromDLDataset(x_drug_path, x_seq_path, y, train_idx, shapes_path)
    val_ds   = GenePromDLDataset(x_drug_path, x_seq_path, y, val_idx, shapes_path)
    test_ds  = GenePromDLDataset(x_drug_path, x_seq_path, y, test_idx, shapes_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              pin_memory=pin_memory, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=pin_memory)

    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("=" * 60)
    print("GenePromDL — Dataset Test")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    drug, seq, cell, label = next(iter(train_loader))

    print("\nOne batch shapes:")
    print(f"  drug  : {drug.shape}")
    print(f"  seq   : {seq.shape}")
    print(f"  cell  : {cell.shape}")
    print(f"  label : {label.shape}")

    print("\nSample values:")
    print(f"  cell unique: {torch.unique(cell)[:10]}")

# """
# STEP 3 — PYTORCH DATASET AND DATALOADER
# File: src/03_dataset.py
# Memory-safe Dataset that reads memmaps on demand.
# DataLoader uses multiple workers + pin_memory for fast GPU transfer.
# """

# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import pickle
# import os
# import sys
# import platform

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from config import *

# # Windows fix: set multiprocessing start method
# if platform.system() == 'Windows':
#     # Use 0 workers on Windows to avoid pickle issues
#     WINDOWS_WORKERS = 0
# else:
#     WINDOWS_WORKERS = NUM_WORKERS


# class GenePromDLDataset(Dataset):
#     """
#     PyTorch Dataset for GenePromDL.
#     Reads X_drug and X_seq from numpy memmaps ON DEMAND.
#     Only the requested rows are loaded — no full array in RAM.
    
#     Windows-compatible: uses lazy loading + picklable state.
#     """

#     def __init__(self, x_drug_path, x_seq_path, y, indices, shapes_path):
#         # Store paths (not memmap objects)
#         self.x_drug_path = x_drug_path
#         self.x_seq_path = x_seq_path
#         self.y = y
#         self.indices = indices
        
#         # Load shapes
#         with open(shapes_path, "rb") as f:
#             shapes = pickle.load(f)
#         self.n_samples = shapes["n_samples"]
#         self.drug_dim = shapes["drug_dim"]  # 2048
#         self.seq_dim = shapes["seq_dim"]    # (1200, 4)
        
#         # Lazy memmap objects (created on first access)
#         self._X_drug = None
#         self._X_seq = None

#     def _ensure_memmaps(self):
#         """Create memmap objects only when needed (lazy loading)."""
#         if self._X_drug is None:
#             self._X_drug = np.memmap(self.x_drug_path, dtype="uint8",
#                                      mode="r", shape=(self.n_samples, 2048))
#         if self._X_seq is None:
#             self._X_seq = np.memmap(self.x_seq_path, dtype="uint8",
#                                     mode="r", shape=(self.n_samples, 1200, 4))
#         return self._X_drug, self._X_seq

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         i = self.indices[idx]
#         X_drug, X_seq = self._ensure_memmaps()

#         drug = torch.from_numpy(X_drug[i].astype(np.float32))  # (2048,)
#         seq = torch.from_numpy(X_seq[i].astype(np.float32))    # (1200, 4)
#         seq = seq.permute(1, 0)                                 # → (4, 1200)
#         label = torch.tensor(float(self.y[i]), dtype=torch.float32)

#         return drug, seq, label

#     # Windows multiprocessing support
#     def __getstate__(self):
#         """What to pickle when spawning workers."""
#         state = self.__dict__.copy()
#         # Don't pickle memmap objects (not picklable on Windows)
#         state['_X_drug'] = None
#         state['_X_seq'] = None
#         return state

#     def __setstate__(self, state):
#         """Restore state when unpickling."""
#         self.__dict__.update(state)
#         # Memmaps will be recreated on first access
#         self._X_drug = None
#         self._X_seq = None


# def get_dataloaders(data_dir=DATA_DIR,
#                     batch_size=BATCH_SIZE,
#                     num_workers=None,
#                     pin_memory=PIN_MEMORY):
#     """Build train, val, test DataLoaders."""
    
#     # Use Windows-safe workers
#     if num_workers is None:
#         num_workers = WINDOWS_WORKERS
    
#     y = np.load(os.path.join(data_dir, "y.npy"))
#     train_idx = np.load(os.path.join(data_dir, "train_idx.npy"))
#     val_idx = np.load(os.path.join(data_dir, "val_idx.npy"))
#     test_idx = np.load(os.path.join(data_dir, "test_idx.npy"))

#     x_drug_path = os.path.join(data_dir, "X_drug.dat")
#     x_seq_path = os.path.join(data_dir, "X_seq.dat")
#     shapes_path = os.path.join(data_dir, "array_shapes.pkl")

#     train_ds = GenePromDLDataset(x_drug_path, x_seq_path, y, train_idx, shapes_path)
#     val_ds = GenePromDLDataset(x_drug_path, x_seq_path, y, val_idx, shapes_path)
#     test_ds = GenePromDLDataset(x_drug_path, x_seq_path, y, test_idx, shapes_path)

#     train_loader = DataLoader(train_ds, batch_size=batch_size,
#                               shuffle=True, num_workers=num_workers,
#                               pin_memory=pin_memory, drop_last=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size,
#                             shuffle=False, num_workers=num_workers,
#                             pin_memory=pin_memory)
#     test_loader = DataLoader(test_ds, batch_size=batch_size,
#                              shuffle=False, num_workers=num_workers,
#                              pin_memory=pin_memory)

#     return train_loader, val_loader, test_loader


# # ─── STANDALONE TEST ─────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("=" * 60)
#     print("GenePromDL — PyTorch DataLoader Test")
#     print("=" * 60)
#     print(f"Platform: {platform.system()}")
#     print(f"Workers: {WINDOWS_WORKERS}")

#     train_loader, val_loader, test_loader = get_dataloaders()

#     print(f"\nTrain batches : {len(train_loader)}")
#     print(f"Val batches   : {len(val_loader)}")
#     print(f"Test batches  : {len(test_loader)}")

#     # Test one batch
#     drug, seq, label = next(iter(train_loader))
#     print(f"\nOne batch shapes:")
#     print(f"  drug  : {drug.shape}   dtype: {drug.dtype}")
#     print(f"  seq   : {seq.shape}  dtype: {seq.dtype}")
#     print(f"  label : {label.shape}   dtype: {label.dtype}")
#     print(f"  drug value range: [{drug.min():.1f}, {drug.max():.1f}]")
#     print(f"  seq  value range: [{seq.min():.1f}, {seq.max():.1f}]")

#     # Test GPU transfer
#     drug = drug.to(DEVICE)
#     seq = seq.to(DEVICE)
#     label = label.to(DEVICE)
#     print(f"\nTransferred to {DEVICE}:")
#     print(f"  drug device: {drug.device}")
#     print(f"  seq  device: {seq.device}")

#     print("\n" + "=" * 60)
#     print("DataLoader working correctly.")
#     print("Next: run src/04_model.py")
#     print("=" * 60)