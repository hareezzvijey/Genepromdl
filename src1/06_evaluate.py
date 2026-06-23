# =============================================================================
# STEP 6 — EVALUATE AND ERROR ANALYSIS (FIXED VERSION)
# =============================================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module import get_dataloaders
from model_module import GenePromDL

print("=" * 60)
print("GenePromDL — Evaluation")
print("=" * 60)

# ── Load model ─────────────────────────────────────────────
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = GenePromDL().to(DEVICE)
model.load_state_dict(checkpoint["model_state"])   # ✅ FIXED KEY
model.eval()

print(f"\nLoaded model from epoch {checkpoint['epoch']}")
print(f"Checkpoint val AUROC: {checkpoint['val_auroc']:.4f}")

# ── Load data ──────────────────────────────────────────────
train_loader, val_loader, test_loader = get_dataloaders()

df = pd.read_csv(TABLE_PATH)
train_idx = np.load(os.path.join(DATA_DIR, "train_idx.npy"))
val_idx   = np.load(os.path.join(DATA_DIR, "val_idx.npy"))
test_idx  = np.load(os.path.join(DATA_DIR, "test_idx.npy"))

# ── Prediction function (FIXED) ────────────────────────────
def get_predictions(model, loader, device):
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in loader:

            # ✅ HANDLE BOTH CASES
            if len(batch) == 4:
                drug, seq, cell, labels = batch
                drug = drug.to(device)
                seq  = seq.to(device)
                cell = cell.to(device)

                logits = model(drug, seq, cell)

            else:
                drug, seq, labels = batch
                drug = drug.to(device)
                seq  = seq.to(device)

                logits = model(drug, seq)

            probs = torch.sigmoid(logits)   # ✅ IMPORTANT

            all_labels.extend(labels.numpy())
            all_preds.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# ── Evaluate all splits ────────────────────────────────────
splits = {"Train": train_loader, "Val": val_loader, "Test": test_loader}
all_results = {}

for name, loader in splits.items():
    print(f"\nEvaluating {name}...")
    y_true, y_pred = get_predictions(model, loader, DEVICE)
    all_results[name] = (y_true, y_pred)

# ── Metrics table ──────────────────────────────────────────
print("\n" + "=" * 70)
print("METRICS TABLE")
print("=" * 70)
print(f"{'Split':<8} {'AUROC':>7} {'AUPRC':>7} {'Accuracy':>9} {'F1':>7} {'Precision':>10} {'Recall':>8}")
print("-" * 70)

rows = []
for name, (y_true, y_pred) in all_results.items():
    y_bin  = (y_pred >= 0.5).astype(int)

    auroc  = roc_auc_score(y_true, y_pred)
    auprc  = average_precision_score(y_true, y_pred)
    acc    = accuracy_score(y_true, y_bin)
    f1     = f1_score(y_true, y_bin)
    prec   = precision_score(y_true, y_bin)
    rec    = recall_score(y_true, y_bin)

    print(f"{name:<8} {auroc:>7.4f} {auprc:>7.4f} {acc:>9.4f} {f1:>7.4f} {prec:>10.4f} {rec:>8.4f}")

    rows.append({
        "Split": name,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": acc,
        "F1": f1,
        "Precision": prec,
        "Recall": rec
    })

pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

# ── ROC + PR curves ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("GenePromDL — ROC and PR Curves", fontsize=13, fontweight="bold")

colors = {"Train": "blue", "Val": "green", "Test": "red"}

for name, (y_true, y_pred) in all_results.items():
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    axes[0].plot(fpr, tpr, color=colors[name], linewidth=2,
                 label=f"{name} (AUC={roc_auc_score(y_true,y_pred):.4f})")

    prec_c, rec_c, _ = precision_recall_curve(y_true, y_pred)
    axes[1].plot(rec_c, prec_c, color=colors[name], linewidth=2,
                 label=f"{name} (AP={average_precision_score(y_true,y_pred):.4f})")

axes[0].plot([0,1],[0,1],"k--",lw=1)
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("PR Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_pr_curves.png"), dpi=150)
plt.close()

print("\nROC/PR curves saved.")

# ── Confusion matrix (TEST) ────────────────────────────────
y_true_test, y_pred_test = all_results["Test"]
y_bin_test = (y_pred_test >= 0.5).astype(int)

cm = confusion_matrix(y_true_test, y_bin_test)

print(f"\nConfusion Matrix (Test):")
print(f"TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# ── Error analysis ─────────────────────────────────────────
print("\n" + "=" * 60)
print("ERROR ANALYSIS — Test Set")
print("=" * 60)

test_df = df.iloc[test_idx].copy().reset_index(drop=True)

test_df["y_true"] = y_true_test.astype(int)
test_df["y_pred"] = y_pred_test
test_df["error"]  = (test_df["y_true"] != y_bin_test).astype(int)

gene_err = test_df.groupby("gene")["error"].mean().sort_values(ascending=False)
drug_err = test_df.groupby("drug")["error"].mean().sort_values(ascending=False)

print("\nTop 10 worst-predicted genes:")
print(gene_err.head(10).to_string())

print("\nTop 10 worst-predicted drugs:")
print(drug_err.head(10).to_string())

gene_err.to_csv(os.path.join(RESULTS_DIR, "gene_error_rates.csv"))
drug_err.to_csv(os.path.join(RESULTS_DIR, "drug_error_rates.csv"))
test_df.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)

print("\n" + "=" * 60)
print("Evaluation complete. All outputs saved.")
print("=" * 60)