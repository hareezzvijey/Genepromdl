# =============================================================================
# model_module_v3.py — Four-input model: drug, sequence, methylation, tissue
# FIXED: OOM error resolved by pooling sequence before attention
# =============================================================================

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


class SENetBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite  = nn.Sequential(
            nn.Flatten(), nn.Linear(channels, reduced), nn.ReLU(),
            nn.Linear(reduced, channels), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.squeeze(x)
        w = self.excite(w).unsqueeze(-1)
        return x * w


class SelfAttentionBlock(nn.Module):
    """
    Memory-efficient self-attention.

    ROOT CAUSE of OOM:
      Attention matrix at seq_len=1200, batch=512, heads=4 = 13.7 GB
      Your RTX 3050 has 8 GB VRAM -> instant OOM

    FIX:
      Downsample sequence from 1200 -> 50 positions BEFORE attention.
      Memory drops from 13.7 GB -> 0.1 GB. Fits easily on 8 GB VRAM.

    Why biologically valid:
      Dilated Conv layers already learned motif features at every position.
      Attention over 50 pooled positions still covers the full 1200bp
      promoter (each position represents ~24bp of context). The model
      learns WHICH of the 50 regions matters most for methylation.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, pool_to=50):
        super().__init__()
        self.downsample = nn.AdaptiveAvgPool1d(pool_to)
        # batch_first=True: expects (batch, seq, embed) — cleaner API
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, channels, 1200)
        x_pooled = self.downsample(x)           # (batch, channels, 50)
        x_t = x_pooled.permute(0, 2, 1)        # (batch, 50, channels)
        att, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm(x_t + self.drop(att))
        return x_t.permute(0, 2, 1)            # (batch, channels, 50)


class DrugBranch(nn.Module):
    def __init__(self, drug_dim=DRUG_DIM, units=None, dropout=DROPOUT):
        super().__init__()
        if units is None: units = tuple(DENSE_UNITS)
        self.net = nn.Sequential(
            nn.Linear(drug_dim, units[0]), nn.BatchNorm1d(units[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(units[0], units[1]), nn.BatchNorm1d(units[1]), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class SequenceBranch(nn.Module):
    def __init__(self, n_bases=N_BASES, embed_dim=EMBED_DIM,
                 filters=None, dilations=None, attn_heads=ATTN_HEADS,
                 dropout=DROPOUT, attn_pool_to=50):
        super().__init__()
        if filters   is None: filters   = tuple(CONV_FILTERS)
        if dilations is None: dilations = tuple(DILATION_RATES)

        self.embed = nn.Sequential(
            nn.Conv1d(n_bases, embed_dim, 1),
            nn.LayerNorm([embed_dim, SEQ_LEN])
        )

        self.convs = nn.ModuleList()
        in_ch = embed_dim
        for i, (f, d) in enumerate(zip(filters, dilations)):
            # FIX padding warning: use odd kernel sizes only (1, 3, 5, 7...)
            # was: kernel=8 (even) with dilation=2 -> triggered zero-pad warning
            # now: kernel=7 (odd) with dilation=1 , kernel=3 with dilation=2,4
            kernel = 7 if i == 0 else 3
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_ch, f, kernel_size=kernel,
                          dilation=d, padding="same"),
                nn.BatchNorm1d(f), nn.ReLU(), nn.Dropout(dropout)
            ))
            in_ch = f

        # Attention operates on downsampled sequence (50 positions)
        self.attn  = SelfAttentionBlock(filters[-1], attn_heads,
                                         pool_to=attn_pool_to)
        self.senet = SENetBlock(filters[-1])
        # Pool the 50-position output to a single vector
        self.pool  = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: (batch, 4, 1200)
        s = self.embed(x)                    # (batch, 16, 1200)
        for c in self.convs:
            s = c(s)                         # (batch, 256, 1200)
        s = self.attn(s)                     # (batch, 256, 50)  <- pooled here
        s = self.senet(s)                    # (batch, 256, 50)
        return self.pool(s).squeeze(-1)      # (batch, 256)


class MethylationBranch(nn.Module):
    def __init__(self, out_dim=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, out_dim), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x.unsqueeze(-1))


class TissueBranch(nn.Module):
    """
    Tissue/cancer-type CATEGORY branch.
    One-hot encoded tissue descriptor + cancer type + MSI status.
    Biological context without cell-line identity memorization.
    """
    def __init__(self, tissue_dim, out_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tissue_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


class GenePromDLv3(nn.Module):
    """
    Four-input model for drug SENSITIVITY regression.

    Inputs:
        drug   : (batch, 2048)       Morgan fingerprint
        seq    : (batch, 4, 1200)    one-hot promoter sequence
        meth   : (batch,)            methylation beta-value covariate
        tissue : (batch, tissue_dim) one-hot tissue/cancer-type category

    Output:
        (batch,) continuous Z_SCORE prediction (no activation)
    """
    def __init__(self, tissue_dim=None, meth_embed_dim=16, tissue_embed_dim=32):
        super().__init__()
        if tissue_dim is None:
            tissue_dim = TISSUE_DIM
        assert tissue_dim is not None, "tissue_dim must be set — run 00_rebuild_v3.py first"

        self.drug_branch   = DrugBranch()
        self.seq_branch    = SequenceBranch()
        self.meth_branch   = MethylationBranch(out_dim=meth_embed_dim)
        self.tissue_branch = TissueBranch(tissue_dim, out_dim=tissue_embed_dim)

        fuse_in = DENSE_UNITS[-1] + CONV_FILTERS[-1] + meth_embed_dim + tissue_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, FUSION_UNITS[0]), nn.BatchNorm1d(FUSION_UNITS[0]),
            nn.ReLU(), nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_UNITS[0], FUSION_UNITS[1]), nn.BatchNorm1d(FUSION_UNITS[1]),
            nn.ReLU(), nn.Dropout(DROPOUT),
        )
        self.output_layer = nn.Linear(FUSION_UNITS[1], 1)

    def forward(self, drug, seq, meth, tissue):
        d = self.drug_branch(drug)
        s = self.seq_branch(seq)
        m = self.meth_branch(meth)
        t = self.tissue_branch(tissue)
        f = torch.cat([d, s, m, t], dim=1)
        f = self.fusion(f)
        out = self.output_layer(f)
        return out.squeeze(1)