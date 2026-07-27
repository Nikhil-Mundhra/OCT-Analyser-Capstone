import cv2
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path

class OCT5KDetectionDataset(Dataset):
    """
    Dataset loader for Model 5: OCT5K Object Detection (9 classes of pathology bounding boxes).
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "Images"
        self.annotations_dir = self.root_dir / "Annotations"
        self.transform = transform

        self.image_paths = sorted(list(self.images_dir.glob("*.png")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        ann_path = self.annotations_dir / f"{img_path.stem}.json"

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        boxes = []
        labels = []

        if ann_path.exists():
            with open(ann_path, "r") as f:
                data = json.load(f)
                for item in data.get("annotations", []):
                    # bbox: [xmin, ymin, xmax, ymax]
                    boxes.append(item["bbox"])
                    labels.append(item["label_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0

        return image_tensor, target
