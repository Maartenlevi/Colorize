# train/train_unet.py
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from data.dataset import GrayscaleColorizationDataset
from models.unet import UNet
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = GrayscaleColorizationDataset('./data/raw', split='train')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = UNet().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    total_loss = 0
    for gray, color in tqdm(loader):
        gray, color = gray.to(device), color.to(device)
        output = model(gray)
        loss = loss_fn(output, color)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), f'models/unet_epoch{epoch+1}.pth')
