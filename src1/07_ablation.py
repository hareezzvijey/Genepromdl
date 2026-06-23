# =============================================================================
# STEP 7 — ABLATION STUDY (FIXED)
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
from model_module import GenePromDL, DrugBranch, SequenceBranch

ABLATION_EPOCHS = 10   # reduced (faster, enough for comparison)

print("=" * 60)
print("GenePromDL — Ablation Study")
print("=" * 60)

train_loader, val_loader, test_loader = get_dataloaders()


# ─── VARIANTS ─────────────────────────────────────────────

class Variant1_DrugOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_branch = DrugBranch()
        self.head = nn.Sequential(
            nn.Linear(DENSE_UNITS[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, drug, seq, cell=None):
        d = self.drug_branch(drug)
        return self.head(d).view(-1)   # logits


class Variant2_SeqOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_branch = SequenceBranch()
        self.head = nn.Sequential(
            nn.Linear(CONV_FILTERS[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, drug, seq, cell=None):
        s = self.seq_branch(seq)
        return self.head(s).view(-1)


class Variant3_DualNoAttnSENET(nn.Module):
    def __init__(self):
        super().__init__()

        self.drug_branch = DrugBranch()

        self.embed = nn.Conv1d(N_BASES, EMBED_DIM, 1)
        self.convs = nn.ModuleList()

        in_ch = EMBED_DIM
        for i, (f, d) in enumerate(zip(CONV_FILTERS, DILATION_RATES)):
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_ch, f, 8 if i==0 else 4,
                          dilation=d, padding="same"),
                nn.BatchNorm1d(f),
                nn.ReLU(),
                nn.Dropout(DROPOUT)
            ))
            in_ch = f

        self.pool = nn.AdaptiveMaxPool1d(1)

        fusion_in = DENSE_UNITS[-1] + CONV_FILTERS[-1]

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, FUSION_UNITS[0]),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_UNITS[0], FUSION_UNITS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FUSION_UNITS[1], 1)
        )

    def forward(self, drug, seq, cell=None):
        d = self.drug_branch(drug)

        s = self.embed(seq)
        for c in self.convs:
            s = c(s)

        s = self.pool(s).squeeze(-1)

        return self.fusion(torch.cat([d, s], dim=1)).view(-1)


# ─── TRAIN FUNCTION ───────────────────────────────────────

def train_variant(model, name):
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_state = None
    best_auc = 0

    for epoch in range(ABLATION_EPOCHS):
        model.train()

        for batch in train_loader:
            if len(batch) == 4:
                drug, seq, cell, labels = batch
                cell = cell.to(DEVICE)
            else:
                drug, seq, labels = batch
                cell = None

            drug = drug.to(DEVICE)
            seq = seq.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(drug, seq, cell)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

        # ── Validation
        model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    drug, seq, cell, labels = batch
                    cell = cell.to(DEVICE)
                else:
                    drug, seq, labels = batch
                    cell = None

                drug = drug.to(DEVICE)
                seq = seq.to(DEVICE)

                logits = model(drug, seq, cell)
                probs = torch.sigmoid(logits)

                preds.extend(probs.cpu().numpy())
                trues.extend(labels.numpy())

        auc = roc_auc_score(trues, preds)

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

        print(f"{name} | Epoch {epoch+1} | Val AUC: {auc:.4f}")

    # ── Test
    model.load_state_dict(best_state)
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                drug, seq, cell, labels = batch
                cell = cell.to(DEVICE)
            else:
                drug, seq, labels = batch
                cell = None

            drug = drug.to(DEVICE)
            seq = seq.to(DEVICE)

            logits = model(drug, seq, cell)
            probs = torch.sigmoid(logits)

            preds.extend(probs.cpu().numpy())
            trues.extend(labels.numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    return {
        "Model": name,
        "AUC": round(roc_auc_score(trues, preds), 4),
        "AUPRC": round(average_precision_score(trues, preds), 4),
        "F1": round(f1_score(trues, (preds > 0.5)), 4)
    }


# ─── RUN ─────────────────────────────────────────────

variants = [
    ("Drug Only", Variant1_DrugOnly()),
    ("Sequence Only", Variant2_SeqOnly()),
    ("Dual No SENET/Attn", Variant3_DualNoAttnSENET()),
    ("Full Model", GenePromDL())
]

results = []

for name, model in variants:
    print("\n" + "="*50)
    print(f"Training {name}")
    print("="*50)

    res = train_variant(model, name)
    results.append(res)


# ─── RESULTS ─────────────────────────────────────────

df = pd.DataFrame(results)
print("\nFINAL RESULTS:\n")
print(df)

df.to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)

print("\nSaved to results/")