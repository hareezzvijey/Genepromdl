# =============================================================================
# STEP 2 — MODEL ARCHITECTURE TEST (V3)
# File: src/02_model_v3.py
# =============================================================================

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from model_module_v3 import GenePromDLv3

print("=" * 60)
print("V3 — Model Architecture (Drug + Seq + Meth + Tissue)")
print("=" * 60)

print(f"\nTissue dimension loaded from Step 0: {TISSUE_DIM}")
assert TISSUE_DIM is not None, "Run 00_rebuild_v3.py first!"

model = GenePromDLv3(tissue_dim=TISSUE_DIM).to(DEVICE)
print(f"Device: {DEVICE}")

total = sum(p.numel() for p in model.parameters())
train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters    : {total:,}")
print(f"Trainable parameters: {train:,}")

dummy_drug   = torch.randint(0, 2, (8, DRUG_DIM)).float().to(DEVICE)
dummy_seq    = torch.randint(0, 2, (8, N_BASES, SEQ_LEN)).float().to(DEVICE)
dummy_meth   = torch.rand(8).to(DEVICE)
dummy_tissue = torch.randint(0, 2, (8, TISSUE_DIM)).float().to(DEVICE)

model.eval()
with torch.no_grad():
    out = model(dummy_drug, dummy_seq, dummy_meth, dummy_tissue)

print(f"\nDummy forward pass:")
print(f"  drug   : {dummy_drug.shape}")
print(f"  seq    : {dummy_seq.shape}")
print(f"  meth   : {dummy_meth.shape}")
print(f"  tissue : {dummy_tissue.shape}")
print(f"  output : {out.shape}  values: {out.cpu().numpy()}")

print("\n" + "=" * 60)
print("Model built. Next: run src/03_train_v3.py")
print("=" * 60)