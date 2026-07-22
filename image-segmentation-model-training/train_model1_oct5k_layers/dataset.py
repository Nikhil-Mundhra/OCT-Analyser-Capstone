import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

class OCT5KLayersDataset(Dataset):
    """
    In-Memory Ultra-Fast Dataset loader for Model 1: OCT5K Retinal Layer Segmentation (6 classes).
    0: Vitreous/Background, 1: ILM->OPL, 2: OPL->IS-OS, 3: IS-OS->IBRPE, 4: IBRPE->OBRPE, 5: Choroid/Outer
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "Images"
        self.masks_dir = self.root_dir / "Masks" / "Grading_1"
        self.transform = transform

        img_files = list(self.images_dir.rglob("*.png"))
        self.cached_images = []
        self.cached_masks = []
        
        print(f"[OCT5KLayersDataset] Pre-caching {len(img_files)} images into RAM for lightning-fast GPU training...")
        for img_path in img_files:
            rel_path = img_path.relative_to(self.images_dir)
            mask_path = self.masks_dir / rel_path
            if mask_path.exists():
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if image is not None and mask is not None:
                    self.cached_images.append(image)
                    self.cached_masks.append(np.clip(mask, 0, 5))

        print(f"[OCT5KLayersDataset] Successfully cached {len(self.cached_images)} matched image-mask pairs in RAM.")

    def __len__(self) -> int:
        return len(self.cached_images)

    def __getitem__(self, idx: int):
        image = self.cached_images[idx]
        mask = self.cached_masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor
