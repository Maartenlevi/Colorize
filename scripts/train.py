import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LabNpyDataset
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LabNpyDataset(lab_dir="data/lab_images")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)

    opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(10):  # aantal epochs
        for i, (L, ab) in enumerate(dataloader):
            L = L.to(device)
            ab = ab.to(device)

            batch_size = L.size(0)
            real_labels = torch.full((batch_size, 1, 30, 30), real_label, device=device)
            fake_labels = torch.full((batch_size, 1, 30, 30), fake_label, device=device)

            # Train discriminator
            discriminator.zero_grad()
            real_input = torch.cat((L, ab), dim=1)
            output_real = discriminator(real_input)
            loss_D_real = criterion_GAN(output_real, real_labels)

            fake_ab = generator(L)
            fake_input = torch.cat((L, fake_ab.detach()), dim=1)
            output_fake = discriminator(fake_input)
            loss_D_fake = criterion_GAN(output_fake, fake_labels)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # Train generator
            generator.zero_grad()
            fake_ab = generator(L)
            fake_input = torch.cat((L, fake_ab), dim=1)
            output = discriminator(fake_input)
            loss_G_GAN = criterion_GAN(output, real_labels)
            loss_G_L1 = criterion_L1(fake_ab, ab) * 100.0
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            opt_G.step()

            if i % 50 == 0:
                print(f"Epoch {epoch} Batch {i} Loss D: {loss_D.item():.4f} Loss G: {loss_G.item():.4f}")

if __name__ == "__main__":
    train()
