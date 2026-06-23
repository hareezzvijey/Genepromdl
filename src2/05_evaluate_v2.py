# =============================================================================
# STEP 5 — EVALUATE (V2 — Regression) + INPUT CONTRIBUTION TEST
# File: src/05_evaluate_v2.py
#
# Includes the shuffle test for ALL THREE inputs (not just drug) so you
# can verify each branch is actually contributing to the prediction —
# this is the same diagnostic that revealed the V1 problem, now run
# proactively rather than after the fact.
# =============================================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module_v2 import get_dataloaders_v2
from model_module_v2 import GenePromDLv2

print("=" * 60)
print("V2 — Evaluation (Drug Sensitivity Regression)")
print("=" * 60)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = GenePromDLv2().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"\nLoaded model from epoch {checkpoint['epoch']}")
print(f"Checkpoint val RMSE: {checkpoint['val_rmse']:.4f}  val R2: {checkpoint['val_r2']:.4f}")

train_loader, val_loader, test_loader = get_dataloaders_v2()
df = pd.read_csv(V2_TABLE_PATH)
train_idx = np.load(os.path.join(DATA_DIR, "train_idx_v2.npy"))
val_idx   = np.load(os.path.join(DATA_DIR, "val_idx_v2.npy"))
test_idx  = np.load(os.path.join(DATA_DIR, "test_idx_v2.npy"))


def get_predictions(model, loader, device):
    all_true, all_pred = [], []
    with torch.no_grad():
        for drug, seq, meth, target in loader:
            drug = drug.to(device); seq = seq.to(device); meth = meth.to(device)
            pred = model(drug, seq, meth)
            all_true.extend(target.numpy())
            all_pred.extend(pred.cpu().numpy())
    return np.array(all_true), np.array(all_pred)


# ── Metrics on all splits ─────────────────────────────────────────────────────
splits = {"Train": train_loader, "Val": val_loader, "Test": test_loader}
all_results = {}
print("\nEvaluating splits...")
for name, loader in splits.items():
    y_true, y_pred = get_predictions(model, loader, DEVICE)
    all_results[name] = (y_true, y_pred)

print("\n" + "=" * 70)
print("METRICS TABLE (Regression)")
print("=" * 70)
print(f"{'Split':<8} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'Pearson r':>10}")
print("-" * 70)
rows = []
for name, (y_true, y_pred) in all_results.items():
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    print(f"{name:<8} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f} {r:>10.4f}")
    rows.append({"Split": name, "RMSE": rmse, "MAE": mae, "R2": r2, "Pearson_r": r})

pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "metrics_v2.csv"), index=False)

# ── Scatter plot: predicted vs actual ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("GenePromDL V2 — Predicted vs Actual Z_SCORE", fontsize=13, fontweight="bold")
for ax, (name, (y_true, y_pred)) in zip(axes, all_results.items()):
    ax.scatter(y_true, y_pred, alpha=0.1, s=3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Actual Z_SCORE"); ax.set_ylabel("Predicted Z_SCORE")
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f"{name} (R²={r2:.4f})")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "scatter_pred_vs_actual.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nScatter plots saved.")

# ─────────────────────────────────────────────────────────────────────────────
# INPUT CONTRIBUTION TEST — run for ALL THREE inputs proactively
# This is the critical diagnostic. If ANY input shows ~0 contribution,
# that branch is not being used and should be reported as a finding.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("INPUT CONTRIBUTION TEST (shuffle each input independently)")
print("=" * 70)

drug, seq, meth, target = next(iter(test_loader))
drug, seq, meth = drug.to(DEVICE), seq.to(DEVICE), meth.to(DEVICE)

with torch.no_grad():
    pred_normal = model(drug, seq, meth)

    # Shuffle drug only
    drug_shuf = drug[torch.randperm(drug.size(0))]
    pred_drug_shuf = model(drug_shuf, seq, meth)
    drug_contrib = (pred_normal - pred_drug_shuf).abs().mean().item()

    # Shuffle sequence only
    seq_shuf = seq[torch.randperm(seq.size(0))]
    pred_seq_shuf = model(drug, seq_shuf, meth)
    seq_contrib = (pred_normal - pred_seq_shuf).abs().mean().item()

    # Shuffle methylation only
    meth_shuf = meth[torch.randperm(meth.size(0))]
    pred_meth_shuf = model(drug, seq, meth_shuf)
    meth_contrib = (pred_normal - pred_meth_shuf).abs().mean().item()

print(f"\n  Drug shuffle  -> mean prediction change: {drug_contrib:.4f}")
print(f"  Seq shuffle   -> mean prediction change: {seq_contrib:.4f}")
print(f"  Meth shuffle  -> mean prediction change: {meth_contrib:.4f}")

total = drug_contrib + seq_contrib + meth_contrib
if total > 0:
    print(f"\n  Relative contribution:")
    print(f"    Drug : {drug_contrib/total:.1%}")
    print(f"    Seq  : {seq_contrib/total:.1%}")
    print(f"    Meth : {meth_contrib/total:.1%}")

contrib_df = pd.DataFrame([
    {"Input": "Drug fingerprint", "Mean_pred_change": drug_contrib},
    {"Input": "DNA sequence",     "Mean_pred_change": seq_contrib},
    {"Input": "Methylation",      "Mean_pred_change": meth_contrib},
])
contrib_df.to_csv(os.path.join(RESULTS_DIR, "input_contribution_test.csv"), index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(contrib_df["Input"], contrib_df["Mean_pred_change"],
       color=["#1F4E79", "#2E75B6", "#85C0E8"])
ax.set_ylabel("Mean |prediction change| when shuffled")
ax.set_title("Input Contribution Test — V2")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "input_contribution_chart.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nContribution test saved: {os.path.join(RESULTS_DIR, 'input_contribution_test.csv')}")

print("\n" + "=" * 60)
print("Evaluation complete.")
print("Next: run src/06_ablation_v2.py")
print("=" * 60)