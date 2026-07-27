import os
import sys
import torch

paths = [
    "/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-classification-model-training/checkpoints/multi_head/fold0_last_model.pth",
    "/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-classification-model-training/hf_space/weights/multi_head_mps/fold0_last_model.pth",
    "/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-classification-model-training/hf_space/weights/multi_head_mps/fold0_best_model.pth"
]

for ckpt_path in paths:
    print("=" * 60)
    if os.path.exists(ckpt_path):
        st = os.stat(ckpt_path)
        print(f"File: {ckpt_path}")
        print(f"Modified: {st.st_mtime}")
        print(f"Size: {st.st_size / (1024**2):.2f} MB")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict):
                print("Keys in checkpoint:", list(ckpt.keys()))
                for k in ['epoch', 'phase', 'best_val_loss', 'best_val_macro_f1']:
                    if k in ckpt:
                        print(f"  {k}: {ckpt[k]}")
            else:
                print("Checkpoint is raw state dict")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print(f"Checkpoint not found at {ckpt_path}")
