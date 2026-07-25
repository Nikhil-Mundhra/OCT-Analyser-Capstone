import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class OIMHSUNet(nn.Module):
    """
    Model 4: U-Net for OIMHS Macular Hole & Intraretinal Cysts (5 Classes).
    0: Background, 1: Macular Hole, 2: Choroid, 3: Retina, 4: Intraretinal Cysts (IRC)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 5):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        d = self.up1(x4)
        d = torch.cat([d, x3], dim=1)
        d = self.conv1(d)

        d = self.up2(d)
        d = torch.cat([d, x2], dim=1)
        d = self.conv2(d)

        d = self.up3(d)
        d = torch.cat([d, x1], dim=1)
        d = self.conv3(d)

        return self.outc(d)
