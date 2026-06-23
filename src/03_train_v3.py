# =============================================================================
# STEP 3 — TRAIN (V3 — Regression + Tissue Features)
# File: src/03_train_v3.py
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys
import time
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module_v3 import get_dataloaders_v3
from model_module_v3 import GenePromDLv3

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("V3 — Training (Drug Sensitivity + Tissue Features) on", DEVICE)
print("=" * 60)
print(f"Tissue dimension: {TISSUE_DIM}")

print("\nBuilding DataLoaders...")
train_loader, val_loader, _ = get_dataloaders_v3()
print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

print("\nBuilding model...")
model = GenePromDLv3(tissue_dim=TISSUE_DIM).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=LR_FACTOR,
                              patience=LR_PATIENCE, min_lr=MIN_LR)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_samples = 0.0, 0
    all_true, all_pred = [], []
    for drug, seq, meth, tissue, target in loader:
        drug, seq, meth, tissue, target = (
            drug.to(device), seq.to(device), meth.to(device),
            tissue.to(device), target.to(device))
        optimizer.zero_grad()
        pred = model(drug, seq, meth, tissue)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item() * len(target)
        total_samples += len(target)
        all_true.extend(target.detach().cpu().numpy())
        all_pred.extend(pred.detach().cpu().numpy())

    mse  = total_loss / total_samples
    rmse = np.sqrt(mse)
    r2   = r2_score(all_true, all_pred)
    return mse, rmse, r2


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for drug, seq, meth, tissue, target in loader:
            drug, seq, meth, tissue, target = (
                drug.to(device), seq.to(device), meth.to(device),
                tissue.to(device), target.to(device))
            pred = model(drug, seq, meth, tissue)
            loss = criterion(pred, target)
            total_loss    += loss.item() * len(target)
            total_samples += len(target)
            all_true.extend(target.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())

    mse  = total_loss / total_samples
    rmse = np.sqrt(mse)
    r2   = r2_score(all_true, all_pred)
    return mse, rmse, r2


history = {"train_mse": [], "train_rmse": [], "train_r2": [],
           "val_mse": [], "val_rmse": [], "val_r2": [], "lr": []}
best_val_rmse = float("inf")
patience_counter = 0

print(f"\nTraining — {EPOCHS} epochs max, patience={PATIENCE}")
print("-" * 90)
print(f"{'Epoch':>6} | {'Train MSE':>10} | {'Train RMSE':>10} | {'Train R2':>9} | "
      f"{'Val MSE':>9} | {'Val RMSE':>9} | {'Val R2':>8} | {'Time':>6}")
print("-" * 90)

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    train_mse, train_rmse, train_r2 = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_mse, val_rmse, val_r2       = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step(val_mse)
    elapsed = time.time() - start

    for k, v in zip(["train_mse","train_rmse","train_r2","val_mse","val_rmse","val_r2"],
                     [train_mse, train_rmse, train_r2, val_mse, val_rmse, val_r2]):
        history[k].append(v)
    history["lr"].append(optimizer.param_groups[0]["lr"])

    print(f"{epoch:>6} | {train_mse:>10.4f} | {train_rmse:>10.4f} | {train_r2:>9.4f} | "
          f"{val_mse:>9.4f} | {val_rmse:>9.4f} | {val_r2:>8.4f} | {elapsed:>5.1f}s")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        patience_counter = 0
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "val_rmse": val_rmse, "val_r2": val_r2,
                    "tissue_dim": TISSUE_DIM}, MODEL_PATH)
        print(f"         Saved best model (val_rmse={val_rmse:.4f}, val_r2={val_r2:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarlyStopping at epoch {epoch}.")
            break

print("-" * 90)

with open(os.path.join(RESULTS_DIR, "history_v3.pkl"), "wb") as f:
    pickle.dump(history, f)
pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, "training_log_v3.csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("GenePromDL V3 — Training Curves (with Tissue Features)", fontsize=14, fontweight="bold")
for ax, (tk, vk, title) in zip(axes, [
    ("train_mse", "val_mse", "MSE"), ("train_rmse", "val_rmse", "RMSE"), ("train_r2", "val_r2", "R²"),
]):
    ep = range(1, len(history[tk]) + 1)
    ax.plot(ep, history[tk], "b-o", markersize=3, label="Train")
    ax.plot(ep, history[vk], "r-o", markersize=3, label="Val")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves_v3.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nBest val RMSE: {best_val_rmse:.4f}")
print(f"Model saved  : {MODEL_PATH}")
print("Next: run src/04_evaluate_v3.py")
print("=" * 60)