"""
Script: utils.py
============================
Utility functions to assist in operations related to PyTorch model training and evaluation.

This script provides functions to:
- Test tensor shapes of data for input to the model.
- Save a PyTorch model checkpoint to disk.
- Load a PyTorch model checkpoint from disk.

Functions:
-----------
test_tensor_shapes(train_data_loader, test_data_loader, input_size):
    Verifies the shape of tensors for the train and test dataloaders, ensuring the input size matches the expected size.

save_checkpoint(checkpoint, model_name):
    Saves the current state of the model to disk as a checkpoint file.

load_checkpoint(model_name, model, optimizer=None):
    Loads a PyTorch model checkpoint from disk, restoring the model and optionally the optimizer state.
"""

import os
import torch

def test_tensor_shapes(train_data_loader, test_data_loader, input_size):
    for loader in [train_data_loader, test_data_loader]:
        for idx, (X, y) in enumerate(loader):
            assert X.shape[-1] == input_size, f"Expected input size: {input_size}, got: {X.shape[1]}"
            assert y.shape[1] == 1, f"Output tensor shape mismatch. Expected: 1, Found: {len(y.shape)}"
            if idx >= 2:
                break

def save_checkpoint(checkpoint, model_name):
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(checkpoint, model_path)
    return model_path

def load_checkpoint(model_name, model, optimizer=None):
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, "models")
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer