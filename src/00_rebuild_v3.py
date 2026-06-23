# =============================================================================
# STEP 0 — REBUILD DATASET WITH TISSUE/CANCER-TYPE FEATURES (V3)
# File: src/00_rebuild_v3.py
#
# WHAT'S NEW vs V2:
#   Adds a 4th input: tissue/cancer-type ONE-HOT vector per cell line,
#   sourced from Cell_Lines_Details.xlsx (already downloaded — no new data).
#
#   Deliberately NOT using raw cell-line IDENTITY (556 unique IDs) because
#   that risks the model memorizing per-cell-line averages and ignoring
#   drug/sequence again (the exact failure mode from V1).
#
#   Instead uses: GDSC Tissue descriptor 1, GDSC Tissue descriptor 2,
#   Cancer Type (TCGA label), MSI status — biological CATEGORY features
#   that generalize across cell lines, not memorize individual ones.
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import os
import sys
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

print("=" * 70)
print("STEP 0 — Rebuilding dataset with tissue/cancer-type features (V3)")
print("=" * 70)

# ── Load cached fingerprints and sequences (REUSED — no re-fetch) ────────────
print("\nLoading cached drug fingerprints and gene sequences...")
with open(DRUG_FPS_CACHE, "rb") as f:
    drug_fps = pickle.load(f)
with open(GENE_OHE_CACHE, "rb") as f:
    gene_ohe = pickle.load(f)
print(f"  {len(drug_fps)} drug fingerprints, {len(gene_ohe)} gene sequences loaded")

# ── Load GDSC drug response ───────────────────────────────────────────────────
print("\nLoading GDSC drug response...")
gdsc1 = pd.read_csv(GDSC1_PATH)
gdsc2 = pd.read_csv(GDSC2_PATH)

def clean_gdsc(df):
    df = df.copy()
    if 'Z_SCORE' not in df.columns and 'z_score' in df.columns:
        df.rename(columns={'z_score': 'Z_SCORE'}, inplace=True)
    df = df[['COSMIC_ID', 'DRUG_NAME', 'Z_SCORE']].dropna()
    df['COSMIC_ID'] = df['COSMIC_ID'].astype(int)
    return df

gdsc_all = pd.concat([clean_gdsc(gdsc1), clean_gdsc(gdsc2)], ignore_index=True)
gdsc_all = gdsc_all.drop_duplicates(subset=['COSMIC_ID', 'DRUG_NAME'])
print(f"  Combined GDSC rows: {len(gdsc_all):,}")

# ── COSMIC -> CCLE bridge ──────────────────────────────────────────────────────
print("\nBuilding COSMIC -> CCLE name bridge...")
model_df = pd.read_csv(MODEL_CSV_PATH)
model_df = model_df[['COSMICID', 'CCLEName']].dropna()
model_df['COSMICID'] = model_df['COSMICID'].astype(int)
cosmic_to_ccle = dict(zip(model_df['COSMICID'], model_df['CCLEName']))
gdsc_all['CCLEName'] = gdsc_all['COSMIC_ID'].map(cosmic_to_ccle)
gdsc_all = gdsc_all.dropna(subset=['CCLEName'])

# ── Load tissue/cancer-type features from Cell_Lines_Details.xlsx ────────────
print("\nLoading tissue/cancer-type features...")
cell_details = pd.read_excel(CELL_DETAILS_PATH)
cell_details.columns = [c.strip() for c in cell_details.columns]
print(f"  Columns found: {list(cell_details.columns)}")

# Map flexible column names (handles whitespace/newline variants in the file)
def find_col(df, keywords):
    for c in df.columns:
        c_clean = c.replace("\n", " ").strip().lower()
        if all(k.lower() in c_clean for k in keywords):
            return c
    return None

col_cosmic   = find_col(cell_details, ["cosmic"])
col_tissue1  = find_col(cell_details, ["tissue", "descriptor", "1"]) or find_col(cell_details, ["tissue", "1"])
col_tissue2  = find_col(cell_details, ["tissue", "descriptor", "2"]) or find_col(cell_details, ["tissue", "2"])
col_cancer   = find_col(cell_details, ["cancer", "type"])
col_msi      = find_col(cell_details, ["microsatellite"])

print(f"  Using columns: COSMIC={col_cosmic}, Tissue1={col_tissue1}, "
      f"Tissue2={col_tissue2}, Cancer={col_cancer}, MSI={col_msi}")

tissue_cols = [c for c in [col_tissue1, col_tissue2, col_cancer, col_msi] if c is not None]
cell_details_clean = cell_details[[col_cosmic] + tissue_cols].copy()
cell_details_clean[col_cosmic] = pd.to_numeric(cell_details_clean[col_cosmic], errors='coerce')
cell_details_clean = cell_details_clean.dropna(subset=[col_cosmic])
cell_details_clean[col_cosmic] = cell_details_clean[col_cosmic].astype(int)

for c in tissue_cols:
    cell_details_clean[c] = cell_details_clean[c].fillna("Unknown").astype(str)

# Build COSMIC -> tissue feature row map
cell_details_clean = cell_details_clean.drop_duplicates(subset=[col_cosmic])
cosmic_to_tissue = cell_details_clean.set_index(col_cosmic)[tissue_cols].to_dict(orient="index")
print(f"  Tissue features available for {len(cosmic_to_tissue)} cell lines")

# ── One-hot encode tissue features ────────────────────────────────────────────
tissue_df = cell_details_clean[tissue_cols].copy()
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(tissue_df)
tissue_dim = encoder.transform(tissue_df[:1]).shape[1]
print(f"  Tissue one-hot encoding dimension: {tissue_dim}")

with open(TISSUE_ENCODER_PATH, "wb") as f:
    pickle.dump({"encoder": encoder, "columns": tissue_cols, "dim": tissue_dim}, f)

def encode_tissue(cosmic_id):
    row = cosmic_to_tissue.get(cosmic_id)
    if row is None:
        return np.zeros(tissue_dim, dtype=np.float32)
    row_df = pd.DataFrame([row])[tissue_cols]
    return encoder.transform(row_df)[0].astype(np.float32)

# ── Load methylation matrix ───────────────────────────────────────────────────
print("\nLoading CCLE methylation matrix...")
meth = pd.read_csv(METH_PATH, sep="\t", low_memory=False)
meta_cols = ['TSS_id', 'gene', 'chr', 'fpos', 'tpos', 'strand', 'avg_coverage']
cell_cols = [c for c in meth.columns if c not in meta_cols]
meth[cell_cols] = meth[cell_cols].apply(pd.to_numeric, errors='coerce')
meth_gene = meth.groupby('gene')[cell_cols].mean()

valid_genes = [g for g in meth_gene.index if g in gene_ohe]
print(f"  Genes with sequence available: {len(valid_genes)}")

if len(valid_genes) > N_GENES_USE:
    gene_subset = meth_gene.loc[valid_genes]
    gene_variance = gene_subset.var(axis=1)
    gene_mean     = gene_subset.mean(axis=1)
    gene_balance  = 1 - abs(gene_mean - 0.5) * 2
    gene_score    = gene_variance * gene_balance

    n_variable = int(N_GENES_USE * 0.70)
    n_high     = int(N_GENES_USE * 0.15)
    n_low      = N_GENES_USE - n_variable - n_high

    variable_genes = gene_score.sort_values(ascending=False).head(n_variable * 2)
    remaining = gene_mean.drop(variable_genes.index)
    high_meth_genes = remaining[remaining > 0.8].sort_values(ascending=False).head(n_high)
    low_meth_genes  = remaining[remaining < 0.2].sort_values(ascending=True).head(n_low)
    variable_genes  = variable_genes.head(n_variable)

    selected_genes = (variable_genes.index.tolist()
                      + high_meth_genes.index.tolist()
                      + low_meth_genes.index.tolist())
else:
    selected_genes = valid_genes

print(f"  Selected genes for V3: {len(selected_genes)}")
meth_gene = meth_gene.loc[selected_genes]
valid_drugs = [d for d in drug_fps.keys()]

gdsc_filtered = gdsc_all[gdsc_all['DRUG_NAME'].isin(valid_drugs)]
gdsc_filtered = gdsc_filtered[gdsc_filtered['CCLEName'].isin(meth_gene.columns)]
gdsc_filtered = gdsc_filtered.drop_duplicates(subset=['DRUG_NAME', 'CCLEName'])
print(f"\n  GDSC rows after filtering: {len(gdsc_filtered):,}")

# ── Sample rows ────────────────────────────────────────────────────────────────
print(f"\nSampling {SAMPLE_FRACTION:.0%} of (drug,cell) pairs per gene...")
random_state = np.random.RandomState(RANDOM_SEED)
all_pairs = list(gdsc_filtered[['DRUG_NAME', 'CCLEName', 'COSMIC_ID', 'Z_SCORE']].to_records(index=False))
n_sample_per_gene = int(len(all_pairs) * SAMPLE_FRACTION)

rows = []
for gene in tqdm(selected_genes):
    meth_col = meth_gene.loc[gene]
    valid_for_gene = [(d, c, cid, z) for (d, c, cid, z) in all_pairs if pd.notna(meth_col.get(c, np.nan))]
    n_take = min(n_sample_per_gene, len(valid_for_gene))
    if n_take == 0:
        continue
    idx = random_state.choice(len(valid_for_gene), size=n_take, replace=False)
    sampled = [valid_for_gene[i] for i in idx]

    for drug, cell, cosmic_id, z_score in sampled:
        meth_value = float(meth_col[cell])
        rows.append({
            'drug': drug, 'gene': gene, 'cell_line': cell,
            'cosmic_id': cosmic_id, 'z_score': float(z_score),
            'meth_feature': meth_value,
        })

training_df = pd.DataFrame(rows)
print(f"\nTotal V3 training rows: {len(training_df):,}")
print(f"Z_SCORE mean={training_df['z_score'].mean():.4f}  std={training_df['z_score'].std():.4f}")

# ── Build arrays ───────────────────────────────────────────────────────────────
N = len(training_df)
print(f"\nBuilding arrays for N={N:,} samples (tissue dim={tissue_dim})...")

X_drug   = np.zeros((N, DRUG_DIM), dtype=np.uint8)
X_seq    = np.zeros((N, SEQ_LEN, N_BASES), dtype=np.uint8)
X_meth   = np.zeros((N,), dtype=np.float32)
X_tissue = np.zeros((N, tissue_dim), dtype=np.float32)
y        = np.zeros((N,), dtype=np.float32)

# Cache tissue encoding per cosmic_id (avoid re-encoding same cell repeatedly)
tissue_cache = {}

for i, row in enumerate(tqdm(training_df.itertuples(), total=N)):
    X_drug[i] = drug_fps[row.drug].astype(np.uint8)
    X_seq[i]  = gene_ohe[row.gene].astype(np.uint8)
    X_meth[i] = row.meth_feature
    y[i]      = row.z_score

    cid = row.cosmic_id
    if cid not in tissue_cache:
        tissue_cache[cid] = encode_tissue(cid)
    X_tissue[i] = tissue_cache[cid]

# ── Save everything ────────────────────────────────────────────────────────────
print("\nSaving V3 dataset...")
training_df.to_csv(V3_TABLE_PATH, index=False)
np.save(X_METH_V3_PATH, X_meth)
np.save(X_TISSUE_V3_PATH, X_tissue)
np.save(Y_V3_PATH, y)

X_drug_mm = np.memmap(X_DRUG_V3_PATH, dtype="uint8", mode="w+", shape=(N, DRUG_DIM))
X_drug_mm[:] = X_drug[:]; X_drug_mm.flush()

X_seq_mm = np.memmap(X_SEQ_V3_PATH, dtype="uint8", mode="w+", shape=(N, SEQ_LEN, N_BASES))
X_seq_mm[:] = X_seq[:]; X_seq_mm.flush()

with open(SHAPES_V3_PATH, "wb") as f:
    pickle.dump({
        "n_samples": N,
        "X_drug_shape": (N, DRUG_DIM),
        "X_seq_shape":  (N, SEQ_LEN, N_BASES),
        "tissue_dim":   tissue_dim,
    }, f)

print(f"\nSaved all V3 files. Tissue feature dimension: {tissue_dim}")
print(f"\nIMPORTANT: open config.py and set TISSUE_DIM = {tissue_dim}")
print("\n" + "=" * 70)
print("Next: run src/01_split_by_drug_v3.py")
print("=" * 70)