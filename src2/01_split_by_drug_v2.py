# =============================================================================
# STEP 1 — SPLIT BY DRUG (V2) — FINAL VERSION
# =============================================================================

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

print("=" * 60)
print("V2 — Drug-Based Train/Val/Test Split")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD ONLY REQUIRED COLUMN (FAST + MEMORY SAFE)
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(V2_TABLE_PATH, usecols=['drug'])

print(f"\nTotal samples : {len(df):,}")

# ─────────────────────────────────────────────────────────────
# SPLIT DRUGS
# ─────────────────────────────────────────────────────────────
all_drugs = df['drug'].unique()

np.random.seed(RANDOM_SEED)
np.random.shuffle(all_drugs)

n_total = len(all_drugs)
n_train = int(n_total * TRAIN_FRAC)
n_val   = int(n_total * VAL_FRAC)

train_drugs = set(all_drugs[:n_train])
val_drugs   = set(all_drugs[n_train:n_train + n_val])
test_drugs  = set(all_drugs[n_train + n_val:])

# ─────────────────────────────────────────────────────────────
# CREATE INDICES (FAST)
# ─────────────────────────────────────────────────────────────
train_idx = df.index[df['drug'].isin(train_drugs)].to_numpy()
val_idx   = df.index[df['drug'].isin(val_drugs)].to_numpy()
test_idx  = df.index[df['drug'].isin(test_drugs)].to_numpy()

# ─────────────────────────────────────────────────────────────
# PRINT STATS
# ─────────────────────────────────────────────────────────────
print(f"\n  Train: {len(train_drugs)} drugs, {len(train_idx):,} rows ({len(train_idx)/len(df):.1%})")
print(f"  Val  : {len(val_drugs)} drugs, {len(val_idx):,} rows ({len(val_idx)/len(df):.1%})")
print(f"  Test : {len(test_drugs)} drugs, {len(test_idx):,} rows ({len(test_idx)/len(df):.1%})")

# ─────────────────────────────────────────────────────────────
# SAFETY CHECKS
# ─────────────────────────────────────────────────────────────
assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

print("\nNo overlap: OK")

# Drug-level disjoint check (VERY IMPORTANT)
print("\nDrug overlap check:",
      len(train_drugs & val_drugs),
      len(train_drugs & test_drugs),
      len(val_drugs & test_drugs))

# ─────────────────────────────────────────────────────────────
# LABEL DISTRIBUTION
# ─────────────────────────────────────────────────────────────
y = np.memmap(Y_V2_PATH, dtype="float32", mode="r")

print(f"\nZ_SCORE distribution:")
print(f"Train: mean={y[train_idx].mean():.4f}, std={y[train_idx].std():.4f}")
print(f"Val  : mean={y[val_idx].mean():.4f}, std={y[val_idx].std():.4f}")
print(f"Test : mean={y[test_idx].mean():.4f}, std={y[test_idx].std():.4f}")

# Extra safety
assert len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
np.save(os.path.join(DATA_DIR, "train_idx_v2.npy"), train_idx)
np.save(os.path.join(DATA_DIR, "val_idx_v2.npy"), val_idx)
np.save(os.path.join(DATA_DIR, "test_idx_v2.npy"), test_idx)

print("\n" + "=" * 60)
print("Saved: train_idx_v2.npy, val_idx_v2.npy, test_idx_v2.npy")
print("Next: run src/02_dataset_v2.py")
print("=" * 60)