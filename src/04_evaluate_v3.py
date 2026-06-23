# =============================================================================
# STEP 4 — EVALUATE (V3) + 4-WAY INPUT CONTRIBUTION TEST
# File: src/04_evaluate_v3.py
#
# Same proactive shuffle test as V2, now extended to all FOUR inputs.
# This is the critical comparison: does adding tissue features
# (a) fix the R^2~0 problem, AND
# (b) avoid recreating the "ignore drug, memorize context" failure?
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
from dataset_module_v3 import get_dataloaders_v3
from model_module_v3 import GenePromDLv3

print("=" * 60)
print("V3 — Evaluation (Drug Sensitivity + Tissue Features)")
print("=" * 60)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = GenePromDLv3(tissue_dim=checkpoint.get("tissue_dim", TISSUE_DIM)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"\nLoaded model from epoch {checkpoint['epoch']}")
print(f"Checkpoint val RMSE: {checkpoint['val_rmse']:.4f}  val R2: {checkpoint['val_r2']:.4f}")

train_loader, val_loader, test_loader = get_dataloaders_v3()
df = pd.read_csv(V3_TABLE_PATH)


def get_predictions(model, loader, device):
    all_true, all_pred = [], []
    with torch.no_grad():
        for drug, seq, meth, tissue, target in loader:
            drug, seq, meth, tissue = drug.to(device), seq.to(device), meth.to(device), tissue.to(device)
            pred = model(drug, seq, meth, tissue)
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
print("METRICS TABLE (V3 — with Tissue Features)")
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
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "metrics_v3.csv"), index=False)

# ── Scatter plots ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("GenePromDL V3 — Predicted vs Actual Z_SCORE", fontsize=13, fontweight="bold")
for ax, (name, (y_true, y_pred)) in zip(axes, all_results.items()):
    ax.scatter(y_true, y_pred, alpha=0.1, s=3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"{name} (R²={r2_score(y_true,y_pred):.4f})")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "scatter_pred_vs_actual_v3.png"), dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4-WAY INPUT CONTRIBUTION TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4-WAY INPUT CONTRIBUTION TEST")
print("=" * 70)

drug, seq, meth, tissue, target = next(iter(test_loader))
drug, seq, meth, tissue = drug.to(DEVICE), seq.to(DEVICE), meth.to(DEVICE), tissue.to(DEVICE)

with torch.no_grad():
    pred_normal = model(drug, seq, meth, tissue)

    perm = torch.randperm(drug.size(0))
    drug_contrib   = (pred_normal - model(drug[perm], seq, meth, tissue)).abs().mean().item()
    seq_contrib    = (pred_normal - model(drug, seq[perm], meth, tissue)).abs().mean().item()
    meth_contrib   = (pred_normal - model(drug, seq, meth[perm], tissue)).abs().mean().item()
    tissue_contrib = (pred_normal - model(drug, seq, meth, tissue[perm])).abs().mean().item()

print(f"\n  Drug shuffle    -> mean prediction change: {drug_contrib:.4f}")
print(f"  Seq shuffle     -> mean prediction change: {seq_contrib:.4f}")
print(f"  Meth shuffle    -> mean prediction change: {meth_contrib:.4f}")
print(f"  Tissue shuffle  -> mean prediction change: {tissue_contrib:.4f}")

total = drug_contrib + seq_contrib + meth_contrib + tissue_contrib
if total > 0:
    print(f"\n  Relative contribution:")
    print(f"    Drug   : {drug_contrib/total:.1%}")
    print(f"    Seq    : {seq_contrib/total:.1%}")
    print(f"    Meth   : {meth_contrib/total:.1%}")
    print(f"    Tissue : {tissue_contrib/total:.1%}")

contrib_df = pd.DataFrame([
    {"Input": "Drug fingerprint", "Mean_pred_change": drug_contrib},
    {"Input": "DNA sequence",     "Mean_pred_change": seq_contrib},
    {"Input": "Methylation",      "Mean_pred_change": meth_contrib},
    {"Input": "Tissue/Cancer type","Mean_pred_change": tissue_contrib},
])
contrib_df.to_csv(os.path.join(RESULTS_DIR, "input_contribution_test_v3.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(contrib_df["Input"], contrib_df["Mean_pred_change"],
       color=["#1F4E79", "#2E75B6", "#85C0E8", "#F2A341"])
ax.set_ylabel("Mean |prediction change| when shuffled")
ax.set_title("Input Contribution Test — V3 (with Tissue Features)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "input_contribution_chart_v3.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nContribution test saved.")

# ── Compare against V2 if available ───────────────────────────────────────────
v2_metrics_path = os.path.join(os.path.dirname(RESULTS_DIR), "results_v2", "metrics_v2.csv")
if os.path.exists(v2_metrics_path):
    print("\n" + "=" * 70)
    print("COMPARISON: V2 (no tissue) vs V3 (with tissue)")
    print("=" * 70)
    v2_df = pd.read_csv(v2_metrics_path)
    v2_test = v2_df[v2_df["Split"] == "Test"].iloc[0]
    v3_test = pd.DataFrame(rows)
    v3_test = v3_test[v3_test["Split"] == "Test"].iloc[0]
    print(f"{'Metric':<12} {'V2 (no tissue)':>16} {'V3 (with tissue)':>18}")
    print("-" * 50)
    for m in ["RMSE", "MAE", "R2", "Pearson_r"]:
        print(f"{m:<12} {v2_test[m]:>16.4f} {v3_test[m]:>18.4f}")

print("\n" + "=" * 60)
print("Evaluation complete.")
print("Next: run src/05_ablation_v3.py")
print("=" * 60)