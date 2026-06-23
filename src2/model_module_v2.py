# =============================================================================
# model_module_v2.py — Triple-input model for sensitivity REGRESSION
# Inputs: drug fingerprint, DNA sequence, methylation covariate (scalar)
# Output: continuous Z_SCORE (no sigmoid — linear output for regression)
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
            nn.Flatten(),
            nn.Linear(channels, reduced), nn.ReLU(),
            nn.Linear(reduced, channels), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.squeeze(x)
        w = self.excite(w).unsqueeze(-1)
        return x * w


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        xt = x.permute(2, 0, 1)
        att, attn_weights = self.attn(xt, xt, xt)
        xt = self.norm(xt + self.drop(att))
        return xt.permute(1, 2, 0)


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
                 filters=None, dilations=None, attn_heads=ATTN_HEADS, dropout=DROPOUT):
        super().__init__()
        if filters   is None: filters   = tuple(CONV_FILTERS)
        if dilations is None: dilations = tuple(DILATION_RATES)
        self.embed = nn.Sequential(nn.Conv1d(n_bases, embed_dim, 1), nn.LayerNorm([embed_dim, SEQ_LEN]))
        self.convs = nn.ModuleList()
        in_ch = embed_dim
        for i, (f, d) in enumerate(zip(filters, dilations)):
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_ch, f, 8 if i == 0 else 4, dilation=d, padding="same"),
                nn.BatchNorm1d(f), nn.ReLU(), nn.Dropout(dropout)
            ))
            in_ch = f
        self.attn  = SelfAttentionBlock(filters[-1], attn_heads)
        self.senet = SENetBlock(filters[-1])
        self.pool  = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        s = self.embed(x)
        for c in self.convs:
            s = c(s)
        # s = self.attn(s)
        s = self.senet(s)
        return self.pool(s).squeeze(-1)


class MethylationBranch(nn.Module):
    """
    Small branch for the methylation covariate (a single scalar 0-1 per sample).
    From literature pattern (TransCDR, TCR): methylation is one omics input
    among several, projected to a small embedding before fusion.
    """
    def __init__(self, out_dim=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, out_dim), nn.ReLU(),
        )
    def forward(self, x):
        # x: (batch,) -> (batch, 1)
        return self.net(x.unsqueeze(-1))


class GenePromDLv2(nn.Module):
    """
    Triple-input model for drug SENSITIVITY regression.

    Inputs:
        drug : (batch, 2048)    Morgan fingerprint
        seq  : (batch, 4, 1200) one-hot promoter sequence
        meth : (batch,)         methylation beta-value covariate

    Output:
        (batch,) continuous Z_SCORE prediction (no activation — linear)
    """
    def __init__(self, meth_embed_dim=16):
        super().__init__()
        self.drug_branch = DrugBranch()
        self.seq_branch  = SequenceBranch()
        self.meth_branch = MethylationBranch(out_dim=meth_embed_dim)

        fuse_in = DENSE_UNITS[-1] + CONV_FILTERS[-1] + meth_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, FUSION_UNITS[0]), nn.BatchNorm1d(FUSION_UNITS[0]),
            nn.ReLU(), nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_UNITS[0], FUSION_UNITS[1]), nn.BatchNorm1d(FUSION_UNITS[1]),
            nn.ReLU(), nn.Dropout(DROPOUT),
        )
        self.output_layer = nn.Linear(FUSION_UNITS[1], 1)

    def forward(self, drug, seq, meth):
        d = self.drug_branch(drug)
        s = self.seq_branch(seq)
        m = self.meth_branch(meth)
        f = torch.cat([d, s, m], dim=1)
        f = self.fusion(f)
        out = self.output_layer(f)            # NO sigmoid — regression
        return out.squeeze(1)