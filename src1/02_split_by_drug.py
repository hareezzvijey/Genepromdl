# =============================================================================
# STEP 2 — SPLIT BY DRUG
# File: src/02_split_by_drug.py
# Splits dataset by DRUG NAME so test set has completely unseen drugs.
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

print("=" * 60)
print("GenePromDL — Drug-Based Train/Val/Test Split")
print("=" * 60)

df = pd.read_csv(TABLE_PATH)
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

print(f"\n  Train drugs: {len(train_drugs)}  rows: {len(train_idx):,} ({len(train_idx)/len(df):.1%})")
print(f"  Val drugs  : {len(val_drugs)}  rows: {len(val_idx):,} ({len(val_idx)/len(df):.1%})")
print(f"  Test drugs : {len(test_drugs)}  rows: {len(test_idx):,} ({len(test_idx)/len(df):.1%})")

# Verify no overlap
assert len(set(train_idx) & set(val_idx))  == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx)   & set(test_idx)) == 0
print("\n  No index overlap: OK")

y = np.load(Y_PATH)
print(f"\n  Label balance:")
print(f"    Train: {y[train_idx].mean():.2%} positive")
print(f"    Val  : {y[val_idx].mean():.2%} positive")
print(f"    Test : {y[test_idx].mean():.2%} positive")

print(f"\n  Test drugs (unseen during training): {sorted(test_drugs)[:10]} ...")

np.save(os.path.join(DATA_DIR, "train_idx.npy"), train_idx)
np.save(os.path.join(DATA_DIR, "val_idx.npy"),   val_idx)
np.save(os.path.join(DATA_DIR, "test_idx.npy"),  test_idx)

split_info = {"train_drugs": list(train_drugs),
              "val_drugs":   list(val_drugs),
              "test_drugs":  list(test_drugs)}
with open(os.path.join(DATA_DIR, "split_info.pkl"), "wb") as f:
    pickle.dump(split_info, f)

print("\n" + "=" * 60)
print("Saved: train_idx.npy, val_idx.npy, test_idx.npy")
print("Next: run src/03_dataset.py")
print("=" * 60)