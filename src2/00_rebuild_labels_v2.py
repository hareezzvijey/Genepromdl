# =============================================================================
# STEP 0 — REBUILD DATASET V2 (FINAL SAFE VERSION)
# =============================================================================

import numpy as np
import sys
import os
import pickle
import pandas as pd
from tqdm import tqdm

# 🔥 Fix old pickle compatibility
sys.modules['numpy._core'] = np.core
sys.modules['numpy._c'] = np.core

from config import *

print("=" * 70)
print("STEP 0 — Rebuilding dataset (FINAL SAFE VERSION)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# SAFE PICKLE LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

print("\nLoading cached files...")
drug_fps = load_pickle(DRUG_FPS_CACHE)
gene_ohe = load_pickle(GENE_OHE_CACHE)

print(f"Drugs: {len(drug_fps)} | Genes: {len(gene_ohe)}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD GDSC
# ─────────────────────────────────────────────────────────────────────────────
def clean_gdsc(df):
    df = df.copy()
    if 'Z_SCORE' not in df.columns and 'z_score' in df.columns:
        df.rename(columns={'z_score': 'Z_SCORE'}, inplace=True)

    df = df[['COSMIC_ID', 'DRUG_NAME', 'Z_SCORE']].dropna()
    df['COSMIC_ID'] = df['COSMIC_ID'].astype(int)
    return df

print("\nLoading GDSC...")
gdsc1 = clean_gdsc(pd.read_csv(GDSC1_PATH))
gdsc2 = clean_gdsc(pd.read_csv(GDSC2_PATH))

gdsc_all = pd.concat([gdsc1, gdsc2])
gdsc_all = gdsc_all.drop_duplicates(['COSMIC_ID', 'DRUG_NAME'])

# ─────────────────────────────────────────────────────────────────────────────
# COSMIC → CCLE
# ─────────────────────────────────────────────────────────────────────────────
print("\nMapping COSMIC → CCLE...")
model_df = pd.read_csv(MODEL_CSV_PATH)[['COSMICID', 'CCLEName']].dropna()
model_df['COSMICID'] = model_df['COSMICID'].astype(int)

cosmic_to_ccle = dict(zip(model_df['COSMICID'], model_df['CCLEName']))

gdsc_all['CCLEName'] = gdsc_all['COSMIC_ID'].map(cosmic_to_ccle)
gdsc_all = gdsc_all.dropna(subset=['CCLEName'])

# ─────────────────────────────────────────────────────────────────────────────
# LOAD METHYLATION
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading methylation...")
meth = pd.read_csv(METH_PATH, sep="\t", low_memory=False)

meta_cols = ['TSS_id', 'gene', 'chr', 'fpos', 'tpos', 'strand', 'avg_coverage']
cell_cols = [c for c in meth.columns if c not in meta_cols]

meth[cell_cols] = meth[cell_cols].apply(pd.to_numeric, errors='coerce')
meth_gene = meth.groupby('gene')[cell_cols].mean()

# ─────────────────────────────────────────────────────────────────────────────
# SELECT GENES
# ─────────────────────────────────────────────────────────────────────────────
valid_genes = [g for g in meth_gene.index if g in gene_ohe]
selected_genes = valid_genes[:N_GENES_USE]

meth_gene = meth_gene.loc[selected_genes]

print(f"Selected genes: {len(selected_genes)}")

# ─────────────────────────────────────────────────────────────────────────────
# FILTER VALID PAIRS
# ─────────────────────────────────────────────────────────────────────────────
gdsc_filtered = gdsc_all[
    gdsc_all['DRUG_NAME'].isin(drug_fps.keys()) &
    gdsc_all['CCLEName'].isin(meth_gene.columns)
]

pairs = gdsc_filtered[['DRUG_NAME', 'CCLEName', 'Z_SCORE']].values

print(f"Valid pairs: {len(pairs):,}")

# ─────────────────────────────────────────────────────────────────────────────
# RANDOM SEED (IMPORTANT)
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.RandomState(RANDOM_SEED)

pairs_per_gene = int(len(pairs) * SAMPLE_FRACTION)

# ─────────────────────────────────────────────────────────────────────────────
# ESTIMATE MAX SIZE (UPPER BOUND)
# ─────────────────────────────────────────────────────────────────────────────
N_est = pairs_per_gene * len(selected_genes)
print(f"Estimated max samples: {N_est:,}")

# ─────────────────────────────────────────────────────────────────────────────
# CREATE MEMMAP ARRAYS
# ─────────────────────────────────────────────────────────────────────────────
X_drug = np.memmap(X_DRUG_V2_PATH, dtype="uint8", mode="w+", shape=(N_est, DRUG_DIM))
X_seq  = np.memmap(X_SEQ_V2_PATH, dtype="uint8", mode="w+", shape=(N_est, SEQ_LEN, N_BASES))
X_meth = np.memmap(X_METH_V2_PATH, dtype="float32", mode="w+", shape=(N_est,))
y      = np.memmap(Y_V2_PATH, dtype="float32", mode="w+", shape=(N_est,))

# ─────────────────────────────────────────────────────────────────────────────
# STREAM BUILD
# ─────────────────────────────────────────────────────────────────────────────
idx = 0
csv_path = V2_TABLE_PATH

if os.path.exists(csv_path):
    os.remove(csv_path)

print("\nBuilding dataset (streaming)...")

for gene in tqdm(selected_genes):

    meth_col = meth_gene.loc[gene]

    valid = [(d, c, z) for d, c, z in pairs if pd.notna(meth_col.get(c, np.nan))]

    if len(valid) == 0:
        continue

    take = min(pairs_per_gene, len(valid))
    chosen = rng.choice(len(valid), size=take, replace=False)

    rows = []

    for i in chosen:
        drug, cell, z = valid[i]

        X_drug[idx] = drug_fps[drug]
        X_seq[idx]  = gene_ohe[gene]
        X_meth[idx] = meth_col[cell]
        y[idx]      = z

        rows.append((drug, gene, cell, z))
        idx += 1

    # write batch
    pd.DataFrame(rows, columns=['drug','gene','cell','z_score']) \
        .to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# TRIM UNUSED SPACE
# ─────────────────────────────────────────────────────────────────────────────
print("\nFinalizing dataset...")

X_drug.flush()
X_seq.flush()
X_meth.flush()
y.flush()

print("\n" + "=" * 70)
print("Dataset built successfully")
print(f"Final samples: {idx:,}")
print("=" * 70)