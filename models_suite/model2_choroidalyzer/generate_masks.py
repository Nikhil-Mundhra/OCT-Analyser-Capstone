import os
from pathlib import Path

import numpy as np
from PIL import Image

from choroidalyze import Choroidalyzer


def main():
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / "example_data" / "image1.png"
    output_dir = base_dir

    ch = Choroidalyzer()
    preds = ch.predict(str(image_path))

    np.savez(output_dir / "example_predictions.npz", region=preds[0].numpy(), vessel=preds[1].numpy(), fovea=preds[2].numpy())

    for name, arr in [("region", preds[0].numpy()), ("vessel", preds[1].numpy()), ("fovea", preds[2].numpy())]:
        img = (arr * 255).astype("uint8")
        Image.fromarray(img).save(output_dir / f"{name}_mask.png")

    print(f"Saved masks to {output_dir}")


if __name__ == "__main__":
    main()
