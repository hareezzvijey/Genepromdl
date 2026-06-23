# =============================================================================
# STEP 4 — MODEL ARCHITECTURE (PyTorch)
# =============================================================================

import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


# ─── CELL LINE BRANCH ─────────────────────────────────────────────
class CellLineBranch(nn.Module):
    def __init__(self, num_cells, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_cells, embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.fc(self.embedding(x))


# ─── SENET BLOCK ─────────────────────────────────────────────
class SENetBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.squeeze(x)
        w = self.excite(w)
        return x * w.unsqueeze(-1)


# ─── SEQUENCE BRANCH ─────────────────────────────────────────────
class SequenceBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv1d(N_BASES, EMBED_DIM, kernel_size=1),
            nn.LayerNorm([EMBED_DIM, SEQ_LEN]),
        )

        self.conv_layers = nn.ModuleList()
        in_ch = EMBED_DIM

        for i, (f, d) in enumerate(zip(CONV_FILTERS, DILATION_RATES)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, f,
                          kernel_size=8 if i == 0 else 4,
                          dilation=d, padding="same"),
                nn.BatchNorm1d(f),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            ))
            in_ch = f

        self.senet = SENetBlock(CONV_FILTERS[-1])
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        s = self.embed(x)
        for conv in self.conv_layers:
            s = conv(s)

        s = self.senet(s)
        return self.pool(s).squeeze(-1)


# ─── DRUG BRANCH ─────────────────────────────────────────────
class DrugBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DRUG_DIM, DENSE_UNITS[0]),
            nn.BatchNorm1d(DENSE_UNITS[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(DENSE_UNITS[0], DENSE_UNITS[1]),
            nn.BatchNorm1d(DENSE_UNITS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


# ─── FULL MODEL ─────────────────────────────────────────────
class GenePromDL(nn.Module):
    def __init__(self):
        super().__init__()

        self.drug_branch = DrugBranch()
        self.seq_branch = SequenceBranch()

        self.cell_branch = CellLineBranch(NUM_CELL_LINES)

        # UPDATED
        fusion_in = DENSE_UNITS[-1] + CONV_FILTERS[-1] + 128

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

    #  UPDATED FORWARD
    def forward(self, drug, seq, cell):
        d = self.drug_branch(drug)
        s = self.seq_branch(seq)
        c = self.cell_branch(cell)

        f = torch.cat([d, s, c], dim=1)
        f = self.fusion(f)

        return self.output_layer(f).view(-1)


# ─── TEST BLOCK ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("GenePromDL — Model Test")
    print("=" * 60)

    print(f"NUM_CELL_LINES: {NUM_CELL_LINES}")

    model = GenePromDL().to(DEVICE)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {total:,}")

    # Dummy inputs
    drug = torch.randint(0, 2, (4, DRUG_DIM)).float().to(DEVICE)
    seq  = torch.randint(0, 2, (4, N_BASES, SEQ_LEN)).float().to(DEVICE)
    cell = torch.randint(0, NUM_CELL_LINES, (4,)).long().to(DEVICE)

    with torch.no_grad():
        logits = model(drug, seq, cell)
        probs  = torch.sigmoid(logits)

    print("\nForward pass:")
    print(f"  logits shape: {logits.shape}")
    print(f"  probs  shape: {probs.shape}")
    print(f"  probs range: [{probs.min():.3f}, {probs.max():.3f}]")

    print("\nModel OK")