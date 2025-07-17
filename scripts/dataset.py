import os
import torch
from torch.utils.data import Dataset
import numpy as np

class LabNpyDataset(Dataset):
    def __init__(self, lab_dir):
        self.lab_dir = lab_dir
        self.files = [f for f in os.listdir(lab_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lab_path = os.path.join(self.lab_dir, self.files[idx])
        lab_img = np.load(lab_path).astype('float32')

        # Normaliseer L en ab kanalen
        L = lab_img[:, :, 0] / 100.0          # L: 0-100 → 0-1
        ab = lab_img[:, :, 1:] / 128.0        # ab: -128-127 → -1-1

        L = torch.tensor(L).unsqueeze(0)      # Shape: (1, H, W)
        ab = torch.tensor(ab).permute(2, 0, 1)  # Shape: (2, H, W)

        return L.float(), ab.float()
