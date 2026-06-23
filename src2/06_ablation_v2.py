# =============================================================================
# STEP 6 — ABLATION STUDY (V2 — Regression)
# File: src/06_ablation_v2.py
# Variants: Drug only / Seq only / Meth only / Drug+Seq / Drug+Meth /
#           Seq+Meth / Full model
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module_v2 import get_dataloaders_v2
from model_module_v2 import GenePromDLv2, DrugBranch, SequenceBranch, MethylationBranch

ABLATION_EPOCHS = 15

print("=" * 60)
print("V2 — Ablation Study (Regression)")
print("=" * 60)

train_loader, val_loader, test_loader = get_dataloaders_v2()


class FlexibleVariant(nn.Module):
    """Generic variant that uses any subset of {drug, seq, meth} branches."""
    def __init__(self, use_drug=True, use_seq=True, use_meth=True):
        super().__init__()
        self.use_drug = use_drug
        self.use_seq  = use_seq
        self.use_meth = use_meth

        in_dim = 0
        if use_drug:
            self.drug_branch = DrugBranch()
            in_dim += DENSE_UNITS[-1]
        if use_seq:
            self.seq_branch = SequenceBranch()
            in_dim += CONV_FILTERS[-1]
        if use_meth:
            self.meth_branch = MethylationBranch(out_dim=16)
            in_dim += 16

        self.head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, drug, seq, meth):
        parts = []
        if self.use_drug: parts.append(self.drug_branch(drug))
        if self.use_seq:  parts.append(self.seq_branch(seq))
        if self.use_meth: parts.append(self.meth_branch(meth))
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
        for drug, seq, meth, target in train_loader:
            drug, seq, meth, target = drug.to(DEVICE), seq.to(DEVICE), meth.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(drug, seq, meth), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for drug, seq, meth, target in val_loader:
                drug, seq, meth = drug.to(DEVICE), seq.to(DEVICE), meth.to(DEVICE)
                preds.extend(model(drug, seq, meth).cpu().numpy())
                trues.extend(target.numpy())
        val_rmse = np.sqrt(mean_squared_error(trues, preds))
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch:>2}/{epochs}  val_rmse={val_rmse:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for drug, seq, meth, target in test_loader:
            drug, seq, meth = drug.to(DEVICE), seq.to(DEVICE), meth.to(DEVICE)
            preds.extend(model(drug, seq, meth).cpu().numpy())
            trues.extend(target.numpy())

    preds, trues = np.array(preds), np.array(trues)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2   = r2_score(trues, preds)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Test RMSE={rmse:.4f}  R2={r2:.4f}")

    torch.cuda.empty_cache()
    return {"Model": name, "Test_RMSE": round(rmse, 4),
            "Test_R2": round(r2, 4), "Parameters": params}


variants = [
    ("Drug only",          FlexibleVariant(True,  False, False)),
    ("Sequence only",      FlexibleVariant(False, True,  False)),
    ("Methylation only",   FlexibleVariant(False, False, True)),
    ("Drug + Sequence",    FlexibleVariant(True,  True,  False)),
    ("Drug + Methylation", FlexibleVariant(True,  False, True)),
    ("Sequence + Methylation", FlexibleVariant(False, True, True)),
    ("Full model (all 3)", FlexibleVariant(True,  True,  True)),
]

results = []
for name, model in variants:
    print(f"\n{'─'*60}")
    print(f"Training: {name}")
    print(f"{'─'*60}")
    r = train_and_test(model, name)
    results.append(r)

print("\n" + "=" * 70)
print("ABLATION RESULTS (V2) — Table for paper")
print("=" * 70)
df_abl = pd.DataFrame(results)
print(df_abl.to_string(index=False))
df_abl.to_csv(os.path.join(RESULTS_DIR, "ablation_results_v2.csv"), index=False)

fig, ax = plt.subplots(figsize=(11, 5))
labels = [r["Model"] for r in results]
r2s    = [r["Test_R2"] for r in results]
colors = ["#AACCEE"] * 6 + ["#1F4E79"]
bars = ax.bar(labels, r2s, color=colors, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Test R²", fontsize=12)
ax.set_title("GenePromDL V2 — Ablation Study (Drug Sensitivity Regression)", fontsize=12, fontweight="bold")
ax.axhline(0, color="red", linestyle="--", lw=1)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=20, ha="right")
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ablation_chart_v2.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAblation study complete. Saved to results_v2/")
print("=" * 60)