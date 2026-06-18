# =============================================================================
# STEP 1 — VERIFY DATA AND GPU
# File: src/01_verify.py
# Checks all data files are valid AND confirms GPU is detected.
# Run this FIRST every session.
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import os
import sys
import torch
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

print("=" * 60)
print("GenePromDL (PyTorch) — System and Data Verification")
print("=" * 60)

# ── GPU check ─────────────────────────────────────────────────────────────────
print("\n[GPU]")
if torch.cuda.is_available():
    gpu_name  = torch.cuda.get_device_name(0)
    gpu_mem   = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU detected   : {gpu_name}")
    print(f"  VRAM           : {gpu_mem:.1f} GB")
    print(f"  CUDA version   : {torch.version.cuda}")
    print(f"  Training device: CUDA (GPU)")
else:
    print("  No GPU detected — training on CPU (slower)")
print(f"  torch.device   : {DEVICE}")

# ── RAM check ──────────────────────────────────────────────────────────────────
ram = psutil.virtual_memory()
print(f"\n[RAM]  {ram.total/1e9:.1f} GB total  |  {ram.available/1e9:.1f} GB available")

# ── File check ─────────────────────────────────────────────────────────────────
print("\n[FILES]")
files = {
    "X_drug.dat"       : X_DRUG_PATH,
    "X_seq.dat"        : X_SEQ_PATH,
    "y.npy"            : Y_PATH,
    "array_shapes.pkl" : SHAPES_PATH,
    "training_table.csv": TABLE_PATH,
}
all_ok = True
for name, path in files.items():
    exists = os.path.isfile(path)
    size   = f"{os.path.getsize(path)/1e9:.2f} GB" if exists else "MISSING"
    print(f"  {'OK' if exists else 'FAIL'}  {name:<25} {size}")
    if not exists:
        all_ok = False

if not all_ok:
    print("\nERROR: Missing files. Check DATA_DIR in config.py")
    raise SystemExit(1)

# ── Load shapes ────────────────────────────────────────────────────────────────
print("\n[SHAPES]")
with open(SHAPES_PATH, "rb") as f:
    shapes = pickle.load(f)
N = shapes["n_samples"]
print(f"  N samples : {N:,}")
print(f"  X_drug    : {shapes['X_drug_shape']}")
print(f"  X_seq     : {shapes['X_seq_shape']}")

# ── Load memmaps ───────────────────────────────────────────────────────────────
print("\n[VALIDATION]")
X_drug = np.memmap(X_DRUG_PATH, dtype="uint8", mode="r", shape=(N, 2048))
X_seq  = np.memmap(X_SEQ_PATH,  dtype="uint8", mode="r", shape=(N, 1200, 4))
y      = np.load(Y_PATH)

# Value checks
print(f"  X_drug unique: {np.unique(X_drug[:1000])}  {'OK' if set(np.unique(X_drug[:1000])) <= {0,1} else 'FAIL'}")
print(f"  X_seq  unique: {np.unique(X_seq[:200])}    {'OK' if set(np.unique(X_seq[:200])) <= {0,1} else 'FAIL'}")
print(f"  y      unique: {np.unique(y)}              {'OK' if set(np.unique(y)) <= {0,1} else 'FAIL'}")

# Active bits check
bits = X_drug[:500].sum(axis=1)
print(f"  Drug active bits: min={bits.min()}, max={bits.max()}, mean={bits.mean():.1f}  {'OK' if 10 < bits.mean() < 300 else 'CHECK'}")

# One-hot check
sums = X_seq[:200].sum(axis=(1,2))
print(f"  Seq row sums (expect ~1200): min={sums.min()}, max={sums.max()}, mean={sums.mean():.1f}")

# Label balance
print(f"  Positive rate: {y.mean():.2%}")

# DNA decode
bases = ['A','T','G','C']
dna = ''.join([bases[np.argmax(p)] if p.sum() > 0 else 'N' for p in X_seq[0][:20]])
print(f"  DNA sample 0: {dna}")

# ── Quick GPU tensor test ──────────────────────────────────────────────────────
print("\n[GPU TRANSFER TEST]")
dummy = torch.zeros(256, 2048, dtype=torch.float32)
dummy = dummy.to(DEVICE)
print(f"  Tensor shape on {DEVICE}: {dummy.shape}")
del dummy
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("Verification complete. System is ready.")
print("Next: run src/02_split_by_drug.py")
print("=" * 60)