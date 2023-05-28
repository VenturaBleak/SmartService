"""
Script: dataset.py
========================
This script defines a PyTorch custom Dataset class .

The CustomDataset class is a subclass of PyTorch's Dataset class and overrides the __init__, __len__, and __getitem__ methods. The class accepts input features and targets, which are then converted to PyTorch tensors. This class is used for creating datasets compatible with PyTorch's DataLoader, making it easier to iterate over batches of data during model training or evaluation.

Usage:
    dataset = CustomDataset(X, y)
"""

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y, dtype_X=torch.float32, dtype_y=torch.float32):
        self.X = torch.tensor(X, dtype=dtype_X)
        self.y = torch.tensor(y, dtype=dtype_y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y