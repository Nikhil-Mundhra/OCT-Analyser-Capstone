import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class OCT5KLayersDataset(Dataset):
    """
    Dataset loader for Model 1: OCT5K Retinal Layer Segmentation (5 classes).
    0: Vitreous/Background, 1: ILM->OPL, 2: OPL->IS-OS, 3: IS-OS->IBRPE, 4: IBRPE->OBRPE
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "Images"
        self.masks_dir = self.root_dir / "Masks"
        self.transform = transform

        self.image_paths = sorted(list(self.images_dir.glob("*.png")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Ensure values are within 0-4
        mask = np.clip(mask, 0, 4)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor
