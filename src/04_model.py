# =============================================================================
# STEP 4 — MODEL ARCHITECTURE (PyTorch)
# File: src/04_model.py
# Dual-input CNN with Self-Attention + SENET
# Papers: DeepDSC + DeepMethylation + iDNA-ABT + StableDNAm + tCNNS
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


# ─── SENET BLOCK ──────────────────────────────────────────────────────────────
class SENetBlock(nn.Module):
    """
    Squeeze-and-Excitation block. From StableDNAm (2023).
    Learns WHICH convolutional filters matter most and reweights them.
    Input/Output: (batch, channels, length)
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.squeeze  = nn.AdaptiveAvgPool1d(1)          # global average
        self.excite   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels, length)
        w = self.squeeze(x)                              # (batch, channels, 1)
        w = self.excite(w)                               # (batch, channels)
        w = w.unsqueeze(-1)                              # (batch, channels, 1)
        return x * w                                     # scale each channel


# ─── SELF-ATTENTION BLOCK ─────────────────────────────────────────────────────
class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention over sequence positions. From iDNA-ABT.
    Learns WHICH positions in the 1200bp promoter are most important.
    Enables interpretability: visualise attention weights = see where model focuses.
    Input/Output: (batch, channels, length)
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False         # expects (seq, batch, embed)
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, channels, length)
        # MultiheadAttention expects (length, batch, channels)
        x_t = x.permute(2, 0, 1)                        # (length, batch, channels)
        attended, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm(x_t + self.dropout(attended))   # residual + LayerNorm
        return x_t.permute(1, 2, 0)                     # back to (batch, channels, length)


# ─── DRUG BRANCH ──────────────────────────────────────────────────────────────
class DrugBranch(nn.Module):
    """
    Fully connected branch for Morgan fingerprints.
    From DeepDSC: "Dense layers learn drug-specific chemical features."
    Input : (batch, 2048)
    Output: (batch, 256)
    """
    def __init__(self, drug_dim=2048, units=(512, 256), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(drug_dim, units[0]),
            nn.BatchNorm1d(units[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(units[0], units[1]),
            nn.BatchNorm1d(units[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)                               # (batch, 256)


# ─── SEQUENCE BRANCH ──────────────────────────────────────────────────────────
class SequenceBranch(nn.Module):
    """
    1D CNN branch for promoter DNA sequences.
    Input : (batch, 4, 1200)  — channels-first for PyTorch Conv1d
    Output: (batch, 256)

    Architecture:
    - Embedding Conv: projects 4 channels → embed_dim
    - 3x Dilated Conv1D: captures motifs at increasing scales
      dilation=1: 8bp window  (local CpG dinucleotides)
      dilation=2: 16bp window (CpG clusters)
      dilation=4: 32bp window (CpG islands, TATA box region)
    - Self-Attention: learns which positions matter  [iDNA-ABT]
    - SENET: learns which features matter            [StableDNAm]
    - GlobalMaxPool: takes strongest activation across all positions
    """
    def __init__(self, n_bases=4, embed_dim=16,
                 filters=(64, 128, 256), dilations=(1, 2, 4),
                 attn_heads=4, dropout=0.3):
        super().__init__()

        # Embedding: project 4 one-hot channels to embed_dim
        self.embed = nn.Sequential(
            nn.Conv1d(n_bases, embed_dim, kernel_size=1, padding=0),
            nn.LayerNorm([embed_dim, 1200]),   # normalise across positions
        )

        # Dilated Conv1D stack
        self.conv_layers = nn.ModuleList()
        in_ch = embed_dim
        for i, (f, d) in enumerate(zip(filters, dilations)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, f, kernel_size=8 if i==0 else 4,
                          dilation=d, padding="same"),
                nn.BatchNorm1d(f),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            in_ch = f

        final_filters = filters[-1]

        # Self-attention over positions
        self.attention = SelfAttentionBlock(
            embed_dim=final_filters, num_heads=attn_heads, dropout=0.1)

        # SENET: channel-wise reweighting
        self.senet = SENetBlock(channels=final_filters, reduction=4)

        # Global max pool: (batch, filters, length) → (batch, filters)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: (batch, 4, 1200)
        s = self.embed(x)                                # (batch, 16, 1200)

        for conv in self.conv_layers:
            s = conv(s)                                  # (batch, 256, 1200)

        s = self.attention(s)                            # (batch, 256, 1200)
        s = self.senet(s)                                # (batch, 256, 1200)
        s = self.pool(s).squeeze(-1)                     # (batch, 256)
        return s


# ─── FULL MODEL ───────────────────────────────────────────────────────────────
class GenePromDL(nn.Module):
    """
    Dual-input CNN for drug-induced promoter methylation prediction.

    Inputs:
        drug : (batch, 2048)   — Morgan fingerprint
        seq  : (batch, 4, 1200) — one-hot promoter sequence

    Output:
        (batch, 1) — probability of methylation (sigmoid)
    """
    def __init__(self):
        super().__init__()

        self.drug_branch = DrugBranch(
            drug_dim=DRUG_DIM,
            units=tuple(DENSE_UNITS),
            dropout=DROPOUT
        )
        self.seq_branch = SequenceBranch(
            n_bases=N_BASES,
            embed_dim=EMBED_DIM,
            filters=tuple(CONV_FILTERS),
            dilations=tuple(DILATION_RATES),
            attn_heads=ATTN_HEADS,
            dropout=DROPOUT
        )

        # Fusion: concatenated drug (256) + seq (256) = 512
        fusion_in = DENSE_UNITS[-1] + CONV_FILTERS[-1]  # 256 + 256 = 512
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, FUSION_UNITS[0]),
            nn.BatchNorm1d(FUSION_UNITS[0]),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_UNITS[0], FUSION_UNITS[1]),
            nn.BatchNorm1d(FUSION_UNITS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.output_layer = nn.Linear(FUSION_UNITS[1], 1)

    def forward(self, drug, seq):
        d = self.drug_branch(drug)                       # (batch, 256)
        s = self.seq_branch(seq)                         # (batch, 256)
        f = torch.cat([d, s], dim=1)                     # (batch, 512)
        f = self.fusion(f)                               # (batch, 64)
        out = torch.sigmoid(self.output_layer(f))        # (batch, 1)
        return out.squeeze(1)                            # (batch,)


# ─── STANDALONE TEST ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("GenePromDL — PyTorch Model Architecture")
    print("=" * 60)

    model = GenePromDL().to(DEVICE)
    print(f"\nDevice: {DEVICE}")

    # Count parameters
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {train:,}")

    # Test forward pass
    dummy_drug = torch.randint(0, 2, (8, 2048)).float().to(DEVICE)
    dummy_seq  = torch.randint(0, 2, (8, 4, 1200)).float().to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(dummy_drug, dummy_seq)

    print(f"\nDummy forward pass:")
    print(f"  drug input  : {dummy_drug.shape}")
    print(f"  seq  input  : {dummy_seq.shape}")
    print(f"  output      : {out.shape}  values: {out.cpu().numpy()}")
    print(f"  all in [0,1]: {'YES' if out.min() >= 0 and out.max() <= 1 else 'NO'}")

    if DEVICE.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated(0) / 1e6
        mem_res   = torch.cuda.memory_reserved(0) / 1e6
        print(f"\nGPU memory allocated : {mem_alloc:.1f} MB")
        print(f"GPU memory reserved  : {mem_res:.1f} MB")

    print("\n" + "=" * 60)
    print("Model built successfully.")
    print("Next: run src/05_train.py")
    print("=" * 60)