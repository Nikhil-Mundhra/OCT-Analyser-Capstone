import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        if g1.shape[-2:] != x.shape[-2:]:
            g1 = F.interpolate(g1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpAttention(nn.Module):
    def __init__(self, in_channels_deeper, in_channels_skip, out_channels, bilinear=False):
        super().__init__()
        self.attention = AttentionBlock(F_g=in_channels_deeper, F_l=in_channels_skip, F_int=in_channels_skip // 2)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_skip + in_channels_deeper, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_deeper, in_channels_deeper // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels_skip + in_channels_deeper // 2, out_channels)

    def forward(self, x_deeper, x_skip):
        x_upsampled = self.up(x_deeper)
        diffY = x_skip.size()[2] - x_upsampled.size()[2]
        diffX = x_skip.size()[3] - x_upsampled.size()[3]
        x_upsampled = F.pad(x_upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x_skip_attended = self.attention(g=x_deeper, x=x_skip)
        x = torch.cat([x_skip_attended, x_upsampled], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class HRFAttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, base_filters=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_filters = base_filters

        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16)

        self.up4 = UpAttention(base_filters * 16, base_filters * 8, base_filters * 8, bilinear)
        self.up3 = UpAttention(base_filters * 8, base_filters * 4, base_filters * 4, bilinear)
        self.up2 = UpAttention(base_filters * 4, base_filters * 2, base_filters * 2, bilinear)
        self.up1 = UpAttention(base_filters * 2, base_filters, base_filters, bilinear)

        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        logits = self.outc(x)
        return logits

class HRFAttentionUNetWrapper:
    def __init__(self, weights_path: str = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = torch.device(device)
            
        self.model = HRFAttentionUNet(n_channels=3, n_classes=1).to(self.device)
        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['model_state_dict'])
                else:
                    self.model.load_state_dict(state_dict)
                print(f"[HRF-AUNet] Successfully loaded weights from {weights_path}")
            except Exception as e:
                print(f"[Warning] Could not load HRF weights: {e}")

    def apply_clahe(self, image_rgb: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    def preprocess_image(self, image_rgb: np.ndarray) -> torch.Tensor:
        clahe_img = self.apply_clahe(image_rgb)
        img_float = clahe_img.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_float - mean) / std
        
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()
        return tensor

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        self.model.eval()
        tensor = self.preprocess_image(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            return probs

    def predict_batch(self, batch_images_rgb: list) -> list:
        self.model.eval()
        tensors = [self.preprocess_image(img) for img in batch_images_rgb]
        batch_tensor = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            return [probs[i] for i in range(len(batch_images_rgb))]
