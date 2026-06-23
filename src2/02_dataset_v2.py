# =============================================================================
# STEP 2 — TEST DATALOADER (V2)
# File: src/02_dataset_v2.py
# =============================================================================

import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module_v2 import get_dataloaders_v2

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

print("=" * 60)
print("V2 — DataLoader Test")
print("=" * 60)

train_loader, val_loader, test_loader = get_dataloaders_v2()
print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches  : {len(val_loader)}")
print(f"Test batches : {len(test_loader)}")

drug, seq, meth, target = next(iter(train_loader))
print(f"\nOne batch:")
print(f"  drug   : {drug.shape}   dtype: {drug.dtype}")
print(f"  seq    : {seq.shape}  dtype: {seq.dtype}")
print(f"  meth   : {meth.shape}     dtype: {meth.dtype}  range: [{meth.min():.3f}, {meth.max():.3f}]")
print(f"  target : {target.shape}     dtype: {target.dtype}  range: [{target.min():.3f}, {target.max():.3f}]")

drug, seq, meth, target = drug.to(DEVICE), seq.to(DEVICE), meth.to(DEVICE), target.to(DEVICE)
print(f"\nTransferred to {DEVICE}: OK")


if __name__ == "__main__":

    print("=" * 60)
    print("V2 — DataLoader Test")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders_v2()

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")

    drug, seq, meth, target = next(iter(train_loader))

    print(f"\nOne batch:")
    print(f"  drug   : {drug.shape}   dtype: {drug.dtype}")
    print(f"  seq    : {seq.shape}    dtype: {seq.dtype}")
    print(f"  meth   : {meth.shape}   dtype: {meth.dtype}")
    print(f"  target : {target.shape} dtype: {target.dtype}")

    drug = drug.to(DEVICE)
    seq = seq.to(DEVICE)
    meth = meth.to(DEVICE)
    target = target.to(DEVICE)

    print(f"\nTransferred to {DEVICE}: OK")

    print("\n" + "=" * 60)
    print("DataLoader working. Next: run src/03_model_v2.py")
    print("=" * 60)