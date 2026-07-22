import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

class OCT5KDetectionDataset(Dataset):
    """
    Dataset loader for Model 5: OCT5K Object Detection (YOLO format annotations).
    9 Classes:
      0: Choroidalfolds, 1: Fluid, 2: Geographicatrophy, 3: Harddrusen,
      4: Hyperfluorescentspots, 5: PRlayerdisruption, 6: Reticulardrusen,
      7: Softdrusen, 8: SoftdrusenPED
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.labels_dir = self.root_dir / "labels"
        self.transform = transform

        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))
        print(f"[OCT5KDetectionDataset] Found {len(self.image_paths)} images in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        txt_path = self.labels_dir / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        h, w = image.shape
        boxes = []
        labels = []

        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0]) + 1  # 1-indexed for Faster R-CNN (0 is background)
                        xc, yc, bw, bh = map(float, parts[1:])

                        # Convert normalized YOLO (xc, yc, bw, bh) to absolute (xmin, ymin, xmax, ymax)
                        xmin = max(0, (xc - bw / 2.0) * w)
                        ymin = max(0, (yc - bh / 2.0) * h)
                        xmax = min(w, (xc + bw / 2.0) * w)
                        ymax = min(h, (yc + bh / 2.0) * h)

                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(cls_id)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0

        return image_tensor, target
