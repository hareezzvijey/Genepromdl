# =============================================================================
# STEP 1 — SPLIT BY DRUG (V3)
# File: src/01_split_by_drug_v3.py
# =============================================================================

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

print("=" * 60)
print("V3 — Drug-Based Train/Val/Test Split")
print("=" * 60)

df = pd.read_csv(V3_TABLE_PATH)
print(f"\nTotal samples : {len(df):,}")

all_drugs = df['drug'].unique()
np.random.seed(RANDOM_SEED)
np.random.shuffle(all_drugs)

n_total = len(all_drugs)
n_train = int(n_total * TRAIN_FRAC)
n_val   = int(n_total * VAL_FRAC)

train_drugs = set(all_drugs[:n_train])
val_drugs   = set(all_drugs[n_train:n_train + n_val])
test_drugs  = set(all_drugs[n_train + n_val:])

train_idx = np.where(df['drug'].isin(train_drugs).values)[0]
val_idx   = np.where(df['drug'].isin(val_drugs).values)[0]
test_idx  = np.where(df['drug'].isin(test_drugs).values)[0]

print(f"\n  Train: {len(train_drugs)} drugs, {len(train_idx):,} rows ({len(train_idx)/len(df):.1%})")
print(f"  Val  : {len(val_drugs)} drugs, {len(val_idx):,} rows ({len(val_idx)/len(df):.1%})")
print(f"  Test : {len(test_drugs)} drugs, {len(test_idx):,} rows ({len(test_idx)/len(df):.1%})")

assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0
print("\n  No overlap: OK")

y = np.load(Y_V3_PATH)
print(f"\n  Z_SCORE distribution:")
print(f"    Train: mean={y[train_idx].mean():.4f}  std={y[train_idx].std():.4f}")
print(f"    Val  : mean={y[val_idx].mean():.4f}  std={y[val_idx].std():.4f}")
print(f"    Test : mean={y[test_idx].mean():.4f}  std={y[test_idx].std():.4f}")

np.save(os.path.join(DATA_DIR, "train_idx_v3.npy"), train_idx)
np.save(os.path.join(DATA_DIR, "val_idx_v3.npy"),   val_idx)
np.save(os.path.join(DATA_DIR, "test_idx_v3.npy"),  test_idx)

print("\n" + "=" * 60)
print("Saved split indices. Next: run src/02_model_v3.py")
print("=" * 60)