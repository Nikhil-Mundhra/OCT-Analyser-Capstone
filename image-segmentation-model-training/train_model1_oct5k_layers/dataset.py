import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import config

class OCT5KLayersDataset(Dataset):
    """
    High-Resolution (512x512) Dataset loader for Model 1: OCT5K Retinal Layer Segmentation (6 classes).
    0: Vitreous/Background, 1: ILM->OPL, 2: OPL->IS-OS, 3: IS-OS->IBRPE, 4: IBRPE->OBRPE, 5: Choroid/Outer
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "Images"
        self.masks_dir = self.root_dir / "Masks" / "Grading_1"
        self.transform = transform

        self.samples = []
        img_files = list(self.images_dir.rglob("*.png"))
        
        for img_path in img_files:
            rel_path = img_path.relative_to(self.images_dir)
            mask_path = self.masks_dir / rel_path
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        print(f"[OCT5KLayersDataset] Loaded {len(self.samples)} high-resolution matched image-mask pairs.", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if image.shape != (512, 512):
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        mask = np.clip(mask, 0, 5)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor
