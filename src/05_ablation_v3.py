# =============================================================================
# STEP 5 — ABLATION STUDY (V3 — 4 inputs, 15 combinations)
# File: src/05_ablation_v3.py
# Tests every combination of {drug, seq, meth, tissue} -> strongest evidence
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module_v3 import get_dataloaders_v3
from model_module_v3 import DrugBranch, SequenceBranch, MethylationBranch, TissueBranch

ABLATION_EPOCHS = 12   # 15 combinations is a lot of training — keep epochs modest

print("=" * 60)
print("V3 — Ablation Study (all combinations of 4 inputs)")
print("=" * 60)

train_loader, val_loader, test_loader = get_dataloaders_v3()


class FlexibleVariantV3(nn.Module):
    def __init__(self, use_drug, use_seq, use_meth, use_tissue, tissue_dim):
        super().__init__()
        self.use_drug   = use_drug
        self.use_seq    = use_seq
        self.use_meth   = use_meth
        self.use_tissue = use_tissue

        in_dim = 0
        if use_drug:
            self.drug_branch = DrugBranch(); in_dim += DENSE_UNITS[-1]
        if use_seq:
            self.seq_branch = SequenceBranch(); in_dim += CONV_FILTERS[-1]
        if use_meth:
            self.meth_branch = MethylationBranch(out_dim=16); in_dim += 16
        if use_tissue:
            self.tissue_branch = TissueBranch(tissue_dim, out_dim=32); in_dim += 32

        self.head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, drug, seq, meth, tissue):
        parts = []
        if self.use_drug:   parts.append(self.drug_branch(drug))
        if self.use_seq:    parts.append(self.seq_branch(seq))
        if self.use_meth:   parts.append(self.meth_branch(meth))
        if self.use_tissue: parts.append(self.tissue_branch(tissue))
        f = torch.cat(parts, dim=1)
        return self.head(f).squeeze(1)


def train_and_test(model, name, epochs=ABLATION_EPOCHS):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_rmse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for drug, seq, meth, tissue, target in train_loader:
            drug, seq, meth, tissue, target = (
                drug.to(DEVICE), seq.to(DEVICE), meth.to(DEVICE),
                tissue.to(DEVICE), target.to(DEVICE))
            optimizer.zero_grad()
            loss = criterion(model(drug, seq, meth, tissue), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for drug, seq, meth, tissue, target in val_loader:
                drug, seq, meth, tissue = drug.to(DEVICE), seq.to(DEVICE), meth.to(DEVICE), tissue.to(DEVICE)
                preds.extend(model(drug, seq, meth, tissue).cpu().numpy())
                trues.extend(target.numpy())
        val_rmse = np.sqrt(mean_squared_error(trues, preds))
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for drug, seq, meth, tissue, target in test_loader:
            drug, seq, meth, tissue = drug.to(DEVICE), seq.to(DEVICE), meth.to(DEVICE), tissue.to(DEVICE)
            preds.extend(model(drug, seq, meth, tissue).cpu().numpy())
            trues.extend(target.numpy())

    preds, trues = np.array(preds), np.array(trues)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2   = r2_score(trues, preds)
    params = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] Test RMSE={rmse:.4f}  R2={r2:.4f}  params={params:,}")

    torch.cuda.empty_cache()
    return {"Model": name, "Test_RMSE": round(rmse, 4),
            "Test_R2": round(r2, 4), "Parameters": params}


# ── Generate all 15 non-empty combinations of 4 inputs ────────────────────────
input_names = ["Drug", "Sequence", "Methylation", "Tissue"]
all_combos = []
for r in range(1, 5):
    all_combos.extend(combinations(range(4), r))

results = []
for combo in all_combos:
    flags = [i in combo for i in range(4)]
    name = " + ".join([input_names[i] for i in combo])
    print(f"\n{'─'*60}")
    print(f"Training: {name}")
    print(f"{'─'*60}")
    model = FlexibleVariantV3(flags[0], flags[1], flags[2], flags[3], tissue_dim=TISSUE_DIM)
    r = train_and_test(model, name)
    results.append(r)

print("\n" + "=" * 70)
print("FULL ABLATION RESULTS (V3) — 15 combinations")
print("=" * 70)
df_abl = pd.DataFrame(results).sort_values("Test_R2", ascending=False)
print(df_abl.to_string(index=False))
df_abl.to_csv(os.path.join(RESULTS_DIR, "ablation_results_v3.csv"), index=False)

fig, ax = plt.subplots(figsize=(12, 6))
df_sorted = df_abl.sort_values("Test_R2")
colors = ["#1F4E79" if "Drug + Sequence + Methylation + Tissue" == m else "#AACCEE"
          for m in df_sorted["Model"]]
ax.barh(df_sorted["Model"], df_sorted["Test_R2"], color=colors)
ax.set_xlabel("Test R²")
ax.set_title("GenePromDL V3 — Full Ablation (all input combinations)", fontweight="bold")
ax.axvline(0, color="red", linestyle="--", lw=1)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ablation_chart_v3_full.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nFull ablation study complete. Saved to results_v3/")
print("=" * 60)