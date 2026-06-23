import torch
from torch.utils.data import Dataset

class GeneDataset(Dataset):
    def __init__(self, X_drug, X_seq, y, indices):
        self.X_drug = X_drug
        self.X_seq = X_seq
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        drug = torch.tensor(self.X_drug[i], dtype=torch.float32)
        seq  = torch.tensor(self.X_seq[i], dtype=torch.float32)
        label = torch.tensor(self.y[i], dtype=torch.float32)

        return drug, seq, label