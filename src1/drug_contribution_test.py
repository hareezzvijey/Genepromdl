import torch
import numpy as np
import os
import sys

# ── imports ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from dataset_module import get_dataloaders
from model_module import GenePromDL

# ── load model ─────────────────────────────────────
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = GenePromDL().to(DEVICE)
model.load_state_dict(checkpoint["model_state"])   # ✅ FIX: correct key
model.eval()

# ── load data ──────────────────────────────────────
_, _, test_loader = get_dataloaders()

# ── get one batch ──────────────────────────────────
drug, seq, cell, labels = next(iter(test_loader))

drug = drug.to(DEVICE)
seq  = seq.to(DEVICE)
cell = cell.to(DEVICE)

# ── forward pass ───────────────────────────────────
with torch.no_grad():

    # normal prediction
    pred_normal = torch.sigmoid(model(drug, seq, cell))

    # shuffle ONLY drug
    perm = torch.randperm(drug.size(0))
    drug_shuffled = drug[perm]

    pred_shuffled = torch.sigmoid(model(drug_shuffled, seq, cell))

# ── compute difference ─────────────────────────────
diff = (pred_normal - pred_shuffled).abs().mean().item()

print("="*50)
print(f"Mean prediction change when drug is shuffled: {diff:.6f}")
print("="*50)

if diff < 0.01:
    print("❌ Drug branch contributes VERY LITTLE (model ignoring drug)")
elif diff < 0.05:
    print("⚠️ Drug contributes weak/moderate signal")
else:
    print("✅ Drug contributes strong signal")