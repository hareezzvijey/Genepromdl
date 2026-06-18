# =============================================================================
# STEP 7 — ABLATION STUDY (PyTorch)
# File: src/07_ablation.py
# 4 variants → proves each component contributes → Table 2 in paper
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module import get_dataloaders
from model_module import GenePromDL, DrugBranch, SequenceBranch, SENetBlock, SelfAttentionBlock

ABLATION_EPOCHS = 15   # fewer epochs for ablation — enough to compare

print("=" * 60)
print("GenePromDL — Ablation Study")
print("=" * 60)

train_loader, val_loader, test_loader = get_dataloaders()


# ─── VARIANT DEFINITIONS ──────────────────────────────────────────────────────

class Variant1_DrugOnly(nn.Module):
    """Drug fingerprint only — sequence input accepted but ignored."""
    def __init__(self):
        super().__init__()
        self.drug_branch = DrugBranch()
        self.head = nn.Sequential(
            nn.Linear(DENSE_UNITS[-1], 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, drug, seq):
        d = self.drug_branch(drug)
        return torch.sigmoid(self.head(d)).squeeze(1)


class Variant2_SeqOnly(nn.Module):
    """DNA sequence only — drug input accepted but ignored."""
    def __init__(self):
        super().__init__()
        self.seq_branch = SequenceBranch()
        self.head = nn.Sequential(
            nn.Linear(CONV_FILTERS[-1], 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, drug, seq):
        s = self.seq_branch(seq)
        return torch.sigmoid(self.head(s)).squeeze(1)


class Variant3_DualNoAttnSENET(nn.Module):
    """Both branches, but NO self-attention and NO SENET."""
    def __init__(self):
        super().__init__()
        self.drug_branch = DrugBranch()

        # Plain dilated CNN — no attention, no SENET
        n_bases = N_BASES; embed_dim = EMBED_DIM
        self.embed = nn.Conv1d(n_bases, embed_dim, 1)
        self.convs = nn.ModuleList()
        in_ch = embed_dim
        for i, (f, d) in enumerate(zip(CONV_FILTERS, DILATION_RATES)):
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_ch, f, 8 if i==0 else 4, dilation=d, padding="same"),
                nn.BatchNorm1d(f), nn.ReLU(), nn.Dropout(DROPOUT)
            ))
            in_ch = f
        self.pool = nn.AdaptiveMaxPool1d(1)

        fuse_in = DENSE_UNITS[-1] + CONV_FILTERS[-1]
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, FUSION_UNITS[0]), nn.ReLU(), nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_UNITS[0], FUSION_UNITS[1]), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(FUSION_UNITS[1], 1)
        )

    def forward(self, drug, seq):
        d = self.drug_branch(drug)
        s = self.embed(seq)
        for c in self.convs: s = c(s)
        s = self.pool(s).squeeze(-1)
        return torch.sigmoid(self.fusion(torch.cat([d, s], dim=1))).squeeze(1)


# ─── TRAIN AND EVALUATE ONE VARIANT ──────────────────────────────────────────
def train_variant(model, train_loader, val_loader, test_loader, name, epochs=ABLATION_EPOCHS):
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_auroc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for drug, seq, labels in train_loader:
            drug, seq, labels = drug.to(DEVICE), seq.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(drug, seq), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Val AUROC
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for drug, seq, labels in val_loader:
                drug, seq = drug.to(DEVICE), seq.to(DEVICE)
                preds.extend(model(drug, seq).cpu().numpy())
                trues.extend(labels.numpy())
        val_auroc = roc_auc_score(trues, preds)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch:>2}/{epochs}  val_auroc={val_auroc:.4f}")

    # Test with best weights
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for drug, seq, labels in test_loader:
            drug, seq = drug.to(DEVICE), seq.to(DEVICE)
            preds.extend(model(drug, seq).cpu().numpy())
            trues.extend(labels.numpy())

    preds, trues = np.array(preds), np.array(trues)
    auroc = roc_auc_score(trues, preds)
    auprc = average_precision_score(trues, preds)
    f1    = f1_score(trues, (preds >= 0.5).astype(int))
    params = sum(p.numel() for p in model.parameters())
    print(f"  Test AUROC={auroc:.4f}  AUPRC={auprc:.4f}  F1={f1:.4f}")

    torch.cuda.empty_cache()
    return {"Model": name, "Test AUROC": round(auroc,4),
            "Test AUPRC": round(auprc,4), "Test F1": round(f1,4),
            "Parameters": params}


# ─── RUN ALL VARIANTS ─────────────────────────────────────────────────────────
variants = [
    ("Variant 1: Drug Only",                    Variant1_DrugOnly()),
    ("Variant 2: Sequence Only",                Variant2_SeqOnly()),
    ("Variant 3: Dual (No Attention + SENET)",  Variant3_DualNoAttnSENET()),
    ("Variant 4: Full Model (Attn + SENET)",    GenePromDL()),
]

results = []
for name, model in variants:
    print(f"\n{'─'*60}")
    print(f"Training: {name}")
    print(f"{'─'*60}")
    r = train_variant(model, train_loader, val_loader, test_loader, name)
    results.append(r)

# ─── PRINT AND SAVE TABLE ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ABLATION RESULTS — Table 2 in your paper")
print("=" * 70)
df_abl = pd.DataFrame(results)
print(df_abl.to_string(index=False))

df_abl.to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)

# ─── BAR CHART ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
labels  = [r["Model"].split(":")[0] for r in results]
aurocs  = [r["Test AUROC"] for r in results]
colors  = ["#AACCEE", "#AACCEE", "#AACCEE", "#1F4E79"]

bars = ax.bar(labels, aurocs, color=colors, edgecolor="white", linewidth=1.5)
ax.set_ylim(max(0, min(aurocs) - 0.05), 1.0)
ax.set_ylabel("Test AUROC", fontsize=12)
ax.set_title("GenePromDL — Ablation Study", fontsize=13, fontweight="bold")
ax.axhline(0.5, color="red", linestyle="--", lw=1, label="Random (0.5)")
ax.legend(); ax.grid(axis="y", alpha=0.3)

for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ablation_chart.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nAblation results saved to results/")
print("=" * 60)