# models/unet.py
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def down_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
            )

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
                nn.ReLU()
            )

        self.encoder1 = down_block(1, 64)
        self.encoder2 = down_block(64, 128)

        self.middle = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU())

        self.decoder1 = up_block(256, 128)
        self.decoder2 = up_block(128, 64)

        self.out_layer = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x_mid = self.middle(x2)
        x = self.decoder1(x_mid)
        x = self.decoder2(x)
        x = self.out_layer(x)
        return x
