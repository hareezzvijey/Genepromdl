# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from sklearn.metrics import roc_auc_score
# import os
# import pickle

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("Device:", DEVICE)

# # ─── LOAD DATA ─────────────────────────────────────────────
# data_dir = r"D:\Hareezzvijey\genepromdl\data"

# X_drug = np.memmap(os.path.join(data_dir, "X_drug.dat"),
#                    dtype="uint8", mode="r", shape=(1119100, 2048))

# X_seq = np.memmap(os.path.join(data_dir, "X_seq.dat"),
#                   dtype="uint8", mode="r", shape=(1119100, 1200, 4))

# y = np.load(os.path.join(data_dir, "y.npy"))

# # ─── TAKE SMALL SUBSET (VERY IMPORTANT) ───────────────────
# N = 1000
# idx = np.random.choice(len(y), N, replace=False)

# X_drug_small = torch.tensor(X_drug[idx], dtype=torch.float32)
# X_seq_small  = torch.tensor(X_seq[idx], dtype=torch.float32).permute(0, 2, 1)
# y_small      = torch.tensor(y[idx], dtype=torch.float32)

# # ─── SIMPLE MODEL (NO ATTENTION, NO SENET) ─────────────────
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.seq = nn.Sequential(
#             nn.Conv1d(4, 32, 8, padding="same"),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, 4, padding="same"),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool1d(1)
#         )

#         self.drug = nn.Sequential(
#             nn.Linear(2048, 256),
#             nn.ReLU()
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(256 + 64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, d, s):
#         s = self.seq(s).squeeze(-1)
#         d = self.drug(d)
#         x = torch.cat([d, s], dim=1)
#         return self.fc(x).squeeze(1)  # logits

# model = SimpleModel().to(DEVICE)

# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=5e-4)

# # ─── TRAIN LOOP ───────────────────────────────────────────
# print("\nStarting OVERFIT TEST...\n")

# for epoch in range(1, 11):
#     model.train()

#     d = X_drug_small.to(DEVICE)
#     s = X_seq_small.to(DEVICE)
#     labels = y_small.to(DEVICE)

#     optimizer.zero_grad()

#     logits = model(d, s)
#     loss = criterion(logits, labels)

#     loss.backward()
#     optimizer.step()

#     probs = torch.sigmoid(logits).detach().cpu().numpy()
#     auc = roc_auc_score(y_small.numpy(), probs)

#     print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | AUC: {auc:.4f}")

# import numpy as np

# data_dir = r"D:\Hareezzvijey\genepromdl\data"

# X_drug = np.memmap(f"{data_dir}/X_drug.dat",
#                    dtype="uint8", mode="r", shape=(1119100, 2048))

# X_seq = np.memmap(f"{data_dir}/X_seq.dat",
#                   dtype="uint8", mode="r", shape=(1119100, 1200, 4))

# y = np.load(f"{data_dir}/y.npy")
# train_idx = np.load(f"{data_dir}/train_idx.npy")

# print("Checking 5 samples...\n")

# for i in range(5):
#     idx = train_idx[i]

#     print(f"Sample {i}")
#     print("Index:", idx)
#     print("Label:", y[idx])

#     print("Drug sum:", X_drug[idx].sum())
#     print("Seq sum :", X_seq[idx].sum())
#     print("-" * 40)

# import numpy as np

# data_dir = r"D:\Hareezzvijey\genepromdl\data"

# X_drug = np.memmap(f"{data_dir}/X_drug.dat",
#                    dtype="uint8",
#                    mode="r",
#                    shape=(1119100, 2048))

# # take random sample
# idx = np.random.choice(1119100, 200000, replace=False)
# sample = np.array(X_drug[idx])

# unique_drugs = np.unique(sample, axis=0)

# print("Sample size:", sample.shape[0])
# print("Unique drugs (sample):", unique_drugs.shape[0])

from collections import defaultdict
import numpy as np

data_dir = r"D:\Hareezzvijey\genepromdl\data"

X_drug = np.memmap(f"{data_dir}/X_drug.dat",
                   dtype="uint8", mode="r",
                   shape=(1119100, 2048))

X_seq = np.memmap(f"{data_dir}/X_seq.dat",
                  dtype="uint8", mode="r",
                  shape=(1119100, 1200, 4))

y = np.load(f"{data_dir}/y.npy")

pair_map = defaultdict(list)

for i in range(50000):  # sample
    key = (tuple(X_drug[i]), tuple(X_seq[i].flatten()))
    pair_map[key].append(y[i])

conflicts = sum(1 for v in pair_map.values() if len(set(v)) > 1)

print("Total pairs:", len(pair_map))
print("Conflicting pairs:", conflicts)