# =============================================================================
# STEP 3 — MODEL ARCHITECTURE TEST (V2)
# File: src/03_model_v2.py
# =============================================================================

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from model_module_v2 import GenePromDLv2

print("=" * 60)
print("V2 — Model Architecture (Regression)")
print("=" * 60)

model = GenePromDLv2().to(DEVICE)
print(f"\nDevice: {DEVICE}")

total = sum(p.numel() for p in model.parameters())
train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters    : {total:,}")
print(f"Trainable parameters: {train:,}")

dummy_drug = torch.randint(0, 2, (8, DRUG_DIM)).float().to(DEVICE)
dummy_seq  = torch.randint(0, 2, (8, N_BASES, SEQ_LEN)).float().to(DEVICE)
dummy_meth = torch.rand(8).to(DEVICE)

model.eval()
with torch.no_grad():
    out = model(dummy_drug, dummy_seq, dummy_meth)

print(f"\nDummy forward pass:")
print(f"  drug input : {dummy_drug.shape}")
print(f"  seq  input : {dummy_seq.shape}")
print(f"  meth input : {dummy_meth.shape}")
print(f"  output     : {out.shape}  values: {out.cpu().numpy()}")
print(f"  (Output is unbounded — correct for regression, NOT 0-1)")

if DEVICE.type == "cuda":
    print(f"\nGPU memory allocated: {torch.cuda.memory_allocated(0)/1e6:.1f} MB")

print("\n" + "=" * 60)
print("Model built. Next: run src/04_train_v2.py")
print("=" * 60)