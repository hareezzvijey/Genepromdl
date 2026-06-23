# =============================================================================
# 05_train.py — v3 (Two targeted corrections)
# Changes from v2:
#   FIX 8: logits.view(-1) added in train_one_epoch() and evaluate()
#          BCEWithLogitsLoss requires logits and labels to have identical shape
#          model() can return (batch,1) but labels are (batch,) → shape error
#          view(-1) flattens (batch,1) → (batch,) safely
#   NOTE:  Root cause of AUC=0.5 is Attention+LayerNorm+Dropout combination
#          causing noisy, unstable updates at init — not depth alone.
#          USE_ATTENTION=False in model_module.py removes this instability.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module import get_dataloaders
from model_module import GenePromDL

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


if DEVICE.type == "cuda":
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ─── LABEL DISTRIBUTION CHECK ────────────────────────────────────────────────
# Printed at startup so you can confirm data is balanced before wasting GPU time
print("\n--- Label Distribution ---")
y_all = np.load(Y_PATH)
n_pos = int((y_all == 1).sum())
n_neg = int((y_all == 0).sum())
n_tot = len(y_all)
ratio = n_neg / max(n_pos, 1)

print(f"Total  : {n_tot:,}")
print(f"Pos(1) : {n_pos:,}  ({100*n_pos/n_tot:.1f}%)")
print(f"Neg(0) : {n_neg:,}  ({100*n_neg/n_tot:.1f}%)")
print(f"Ratio  : {ratio:.3f}")

if ratio > 3.0:
    print(f"\n  NOTE: ratio={ratio:.1f} > 3 — data IS imbalanced.")
    print(f"  Consider adding pos_weight={ratio:.1f} to BCEWithLogitsLoss.")
    print(f"  Set POS_WEIGHT in config.py and update criterion below.")
else:
    print(f"  Data is balanced (ratio≈{ratio:.2f}) — no pos_weight needed ✓")


# ─── DATA ────────────────────────────────────────────────────────────────────
print("\n--- DataLoaders ---")
train_loader, val_loader, test_loader = get_dataloaders()
print(f"Train batches : {len(train_loader)}")
print(f"Val   batches : {len(val_loader)}")


# ─── MODEL ───────────────────────────────────────────────────────────────────
print("\n--- Model ---")
model = GenePromDL().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params : {n_params:,}")

# FIX 1: BCEWithLogitsLoss — numerically stable, applies sigmoid internally
# No pos_weight since data is ~balanced
criterion = nn.BCEWithLogitsLoss()
print(f"Loss : BCEWithLogitsLoss  (no pos_weight — balanced data)")

# FIX: LR = 5e-4 (was 1e-4 — too slow for this architecture depth)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_FACTOR,
    patience=LR_PATIENCE, min_lr=MIN_LR
)
print(f"LR   : {LEARNING_RATE}  (was 1e-4)")
print(f"Drop : {DROPOUT}  (was 0.25)")
prev_lr = LEARNING_RATE


# ─── ONE EPOCH ───────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_labels, all_preds = [], []

    for i, (drug, seq, cell, labels) in enumerate(loader):

        if i % 200 == 0:
            print(f"Batch {i}/{len(loader)} running...")

        # GPU transfer (optimized)
        drug   = drug.to(device, non_blocking=True)
        seq    = seq.to(device, non_blocking=True)
        cell = cell.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward
        logits = model(drug, seq, cell)

        # FIX: enforce correct shape
        logits = logits.view(-1)

        # Loss
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Convert logits → probabilities
        probs = torch.sigmoid(logits)

        pred_labels = (probs >= 0.5).float()

        total_loss += loss.item() * len(labels)
        total_correct += (pred_labels == labels).sum().item()
        total_samples += len(labels)

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        auroc = 0.5

    return avg_loss, accuracy, auroc


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for drug, seq, cell, labels in loader:

            drug   = drug.to(device, non_blocking=True)
            seq    = seq.to(device, non_blocking=True)
            cell = cell.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(drug, seq, cell)
            logits = logits.view(-1)

            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            pred_labels = (probs >= 0.5).float()

            total_loss += loss.item() * len(labels)
            total_correct += (pred_labels == labels).sum().item()
            total_samples += len(labels)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        auroc = 0.5

    return avg_loss, accuracy, auroc


# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
history = {"tl":[], "ta":[], "tauc":[], "vl":[], "va":[], "vauc":[], "f1":[], "gn":[], "lr":[]}
best_val_auroc = 0.0
patience_counter = 0

print(f"\n{'='*75}")
print(f"{'Ep':>4} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrAUC':>6} | "
      f"{'VlLoss':>7} | {'VlAcc':>6} | {'VlAUC':>6} | Time | LR")
print(f"{'-'*75}")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    tl, ta, tauc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    vl, va, vauc = evaluate(model, val_loader, criterion, DEVICE)

    scheduler.step(vl)
    cur_lr = optimizer.param_groups[0]["lr"]
    if cur_lr != prev_lr:
        print(f"  ↓ LR: {prev_lr:.2e} → {cur_lr:.2e}")
        prev_lr = cur_lr

    elapsed = time.time() - t0
    for k, v in zip(["tl","ta","tauc","vl","va","vauc","f1","gn","lr"],
                    [tl, ta, tauc,  vl, va, vauc, cur_lr]):
        history[k].append(v)

    print(f"{epoch:>4} | {tl:>7.4f} | {ta:>6.4f} | {tauc:>6.4f} | "
          f"{vl:>7.4f} | {va:>6.4f} | {vauc:>6.4f} | "
          f"{elapsed:>4.0f} | {cur_lr:.1e}")

    # # Gradient health check — printed every epoch so you can see if learning starts
    # if gn < 0.001:
    #     print(f"  ⚠ Grad norm very small ({gn:.5f}) — vanishing gradient. Model not updating.")
    # elif gn > 5.0:
    #     print(f"  ⚠ Grad norm large ({gn:.2f}) — check for instability.")

    # Still stuck warning
    if epoch == 3 and vauc < 0.53:
        print(f"\n  ⚠ AUC still near 0.5 after 3 epochs.")
        print(f"  (2) Verify y.npy labels are correct (0s and 1s, not floats)")
        print(f"  (3) Confirm DRUG_DIM={DRUG_DIM} matches actual fingerprint shape")
        print(f"  (4) Try running model on 10 samples manually and print output\n")

    if vauc > best_val_auroc:
        best_val_auroc   = vauc
        patience_counter = 0
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_auroc": vauc, "val_loss": vl,
        }, MODEL_PATH)
        print(f"  ✓ Best saved — val_auroc={vauc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarlyStopping at epoch {epoch}.")
            break

print(f"\nBest val AUROC : {best_val_auroc:.4f}")
if best_val_auroc > 0.6:
    print("Phase 1 target reached. Set USE_ATTENTION=True in model_module.py and retrain.")
else:
    print("Phase 1 target not reached. Check gradient norms and label values above.")


# ─── SAVE CURVES ─────────────────────────────────────────────────────────────
ep = range(1, len(history["tl"]) + 1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].plot(ep, history["tl"], label="Train")
ax[0].plot(ep, history["vl"], label="Val")
ax[0].set_title("Loss"); ax[0].legend()

ax[1].plot(ep, history["tauc"], label="Train")
ax[1].plot(ep, history["vauc"], label="Val")
ax[1].axhline(0.5, color="grey", ls="--", label="Random")
ax[1].set_title("AUROC"); ax[1].legend()

ax[2].plot(ep, history["f1"],  label="F1")
ax[2].plot(ep, history["gn"],  label="Grad norm", alpha=0.7)
ax[2].set_title("F1 & Grad Norm"); ax[2].legend()

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "training_curves_v2.png")
plt.savefig(out, dpi=120)
print(f"Curves saved → {out}")