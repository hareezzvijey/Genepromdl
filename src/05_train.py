# =============================================================================
# STEP 5 — TRAIN (PyTorch + GPU)
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module import get_dataloaders
from model_module import GenePromDL

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("GenePromDL — Training on", DEVICE)
print("=" * 60)

if DEVICE.type == "cuda":
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("\nBuilding DataLoaders...")
train_loader, val_loader, _ = get_dataloaders()
print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")


# ─── BUILD MODEL ──────────────────────────────────────────────────────────────
print("\nBuilding model...")
model = GenePromDL().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 🔥 FIXED (removed verbose)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=MIN_LR
)

# 🔥 NEW: track LR changes
prev_lr = optimizer.param_groups[0]["lr"]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_labels, all_preds = [], []

    for i, (drug, seq, labels) in enumerate(loader):

        # 🔥 DEBUG PRINT (progress check)
        if i % 200 == 0:
            print(f"Batch {i}/{len(loader)} running...")

        # ─── MOVE TO GPU ─────────────────────────────
        drug   = drug.to(device)
        seq    = seq.to(device)
        labels = labels.to(device)

        # ─── FORWARD ────────────────────────────────
        optimizer.zero_grad()
        preds = model(drug, seq)

        # FORCE GPU EXECUTION (important for Windows)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # ─── LOSS ───────────────────────────────────
        loss = criterion(preds, labels)

        # ─── BACKWARD ───────────────────────────────
        loss.backward()

        # Gradient clipping (stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ─── METRICS ────────────────────────────────
        total_loss += loss.item() * len(labels)

        pred_labels = (preds >= 0.5).float()
        total_correct += (pred_labels == labels).sum().item()
        total_samples += len(labels)

        # Move to CPU for sklearn metrics
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

    # ─── FINAL METRICS ─────────────────────────────
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auroc = 0.5

    return avg_loss, accuracy, auroc


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for i, (drug, seq, labels) in enumerate(loader):

            if i % 200 == 0:
                print(f"[VAL] Batch {i}/{len(loader)}")

            drug   = drug.to(device)
            seq    = seq.to(device)
            labels = labels.to(device)

            preds = model(drug, seq)

            if device.type == "cuda":
                torch.cuda.synchronize()

            loss = criterion(preds, labels)

            total_loss += loss.item() * len(labels)

            pred_labels = (preds >= 0.5).float()
            total_correct += (pred_labels == labels).sum().item()
            total_samples += len(labels)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auroc = 0.5

    return avg_loss, accuracy, auroc

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
history = {
    "train_loss": [], "train_acc": [], "train_auroc": [],
    "val_loss":   [], "val_acc":   [], "val_auroc":   [],
    "lr": []
}

best_val_auroc = 0.0
patience_counter = 0

print(f"\nStarting training — {EPOCHS} epochs max, EarlyStopping patience={PATIENCE}")
print(f"Batch size: {BATCH_SIZE}  |  Learning rate: {LEARNING_RATE}")
print("-" * 80)
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Train AUC':>9} | "
      f"{'Val Loss':>9} | {'Val Acc':>8} | {'Val AUC':>8} | {'Time':>6} | {'LR':>8}")
print("-" * 80)

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    train_loss, train_acc, train_auroc = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_acc, val_auroc = evaluate(
        model, val_loader, criterion, DEVICE)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    # 🔥 VERBOSE LR CHANGE
    if current_lr != prev_lr:
        print(f"         ↓ LR reduced: {prev_lr:.6f} → {current_lr:.6f}")
        prev_lr = current_lr

    elapsed = time.time() - start

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["train_auroc"].append(train_auroc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_auroc"].append(val_auroc)
    history["lr"].append(current_lr)

    print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {train_auroc:>9.4f} | "
          f"{val_loss:>9.4f} | {val_acc:>8.4f} | {val_auroc:>8.4f} | {elapsed:>5.1f}s | {current_lr:.6f}")

    if val_auroc > best_val_auroc:
        best_val_auroc   = val_auroc
        patience_counter = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auroc": val_auroc,
            "val_loss": val_loss,
        }, MODEL_PATH)
        print(f"         ✓ Saved best model (val_auroc={val_auroc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarlyStopping triggered at epoch {epoch}.")
            break

print("-" * 80)