# data/dataset.py
import torch
from torch.utils.data import Dataset
from torchvision.datasets import STL10
from torchvision import transforms
from PIL import Image

class GrayscaleColorizationDataset(Dataset):
    def __init__(self, root: str, split='train'):
        self.dataset = STL10(root=root, split=split, download=True)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((96, 96))
        self.to_gray = transforms.Grayscale()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img = self.resize(img)

        color = self.to_tensor(img)
        gray = self.to_tensor(self.to_gray(img))

        return gray, color
