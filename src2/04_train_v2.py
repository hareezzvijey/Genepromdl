# =============================================================================
# STEP 4 — TRAIN (V2 — Regression on Drug Sensitivity)
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
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module_v2 import get_dataloaders_v2
from model_module_v2 import GenePromDLv2

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 🔥 Faster GPU convs
torch.backends.cudnn.benchmark = True

print("=" * 60)
print("V2 — Training (Drug Sensitivity Regression)")
print("=" * 60)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
print("\nBuilding DataLoaders...")
train_loader, val_loader, _ = get_dataloaders_v2()
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
print("\nBuilding model...")
model = GenePromDLv2().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=MIN_LR
)

# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, epoch):
    model.train()

    total_loss, total_samples = 0.0, 0
    all_true, all_pred = [], []

    for batch_idx, (drug, seq, meth, target) in enumerate(loader):

        drug   = drug.to(DEVICE, non_blocking=True)
        seq    = seq.to(DEVICE, non_blocking=True)
        meth   = meth.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)

        # 🔍 GPU DEBUG (first epoch only)
        if epoch == 1 and batch_idx < 3:
            print(f"\n[DEBUG BATCH {batch_idx}]")
            print("  drug device :", drug.device)
            print("  seq device  :", seq.device)
            print("  meth device :", meth.device)
            print("  model device:", next(model.parameters()).device)
            if torch.cuda.is_available():
                print("  GPU memory  :", torch.cuda.memory_allocated() / 1e9, "GB")

        optimizer.zero_grad()

        pred = model(drug, seq, meth)
        loss = criterion(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(target)
        total_samples += len(target)

        all_true.append(target.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

        # 📊 Progress every 500 batches
        if batch_idx % 500 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    mse  = total_loss / total_samples
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(all_true, all_pred)
    r2   = r2_score(all_true, all_pred)

    try:
        corr, _ = pearsonr(all_true, all_pred)
    except:
        corr = 0.0

    return mse, rmse, mae, r2, corr


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()

    total_loss, total_samples = 0.0, 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for drug, seq, meth, target in loader:
            drug   = drug.to(DEVICE, non_blocking=True)
            seq    = seq.to(DEVICE, non_blocking=True)
            meth   = meth.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            pred = model(drug, seq, meth)
            loss = criterion(pred, target)

            total_loss += loss.item() * len(target)
            total_samples += len(target)

            all_true.append(target.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    mse  = total_loss / total_samples
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(all_true, all_pred)
    r2   = r2_score(all_true, all_pred)

    try:
        corr, _ = pearsonr(all_true, all_pred)
    except:
        corr = 0.0

    return mse, rmse, mae, r2, corr


# ─────────────────────────────────────────────
# TRAIN LOOP
# ─────────────────────────────────────────────
history = {
    "train_rmse": [], "val_rmse": [],
    "train_mae": [],  "val_mae": [],
    "train_r2": [],   "val_r2": [],
    "train_corr": [], "val_corr": [],
    "lr": []
}

best_val_rmse = float("inf")
patience_counter = 0

print("\nTraining...")
print("-" * 100)
print(f"{'Ep':>3} | {'TrRMSE':>8} | {'VlRMSE':>8} | {'TrR2':>6} | {'VlR2':>6} | {'TrCorr':>7} | {'VlCorr':>7} | Time")
print("-" * 100)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    tr_mse, tr_rmse, tr_mae, tr_r2, tr_corr = train_one_epoch(model, train_loader, epoch)
    vl_mse, vl_rmse, vl_mae, vl_r2, vl_corr = evaluate(model, val_loader)

    scheduler.step(vl_rmse)

    elapsed = time.time() - t0

    history["train_rmse"].append(tr_rmse)
    history["val_rmse"].append(vl_rmse)
    history["train_mae"].append(tr_mae)
    history["val_mae"].append(vl_mae)
    history["train_r2"].append(tr_r2)
    history["val_r2"].append(vl_r2)
    history["train_corr"].append(tr_corr)
    history["val_corr"].append(vl_corr)
    history["lr"].append(optimizer.param_groups[0]["lr"])

    print(f"{epoch:>3} | {tr_rmse:>8.4f} | {vl_rmse:>8.4f} | {tr_r2:>6.3f} | {vl_r2:>6.3f} | {tr_corr:>7.3f} | {vl_corr:>7.3f} | {elapsed:>4.0f}s")

    if vl_rmse < best_val_rmse:
        best_val_rmse = vl_rmse
        patience_counter = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_rmse": vl_rmse,
            "val_r2": vl_r2,
            "val_corr": vl_corr
        }, MODEL_PATH)

        print(f"     ✓ Saved best model (RMSE={vl_rmse:.4f})")

    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print("-" * 100)

# SAVE
with open(os.path.join(RESULTS_DIR, "history_v2.pkl"), "wb") as f:
    pickle.dump(history, f)

pd.DataFrame(history).to_csv(
    os.path.join(RESULTS_DIR, "training_log_v2.csv"), index=False
)

print(f"\nBest Val RMSE: {best_val_rmse:.4f}")
print("Model saved at:", MODEL_PATH)
print("=" * 60)