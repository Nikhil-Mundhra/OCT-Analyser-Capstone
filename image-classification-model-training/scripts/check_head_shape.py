import torch

ckpt_path = "/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-classification-model-training/checkpoints/multi_head/fold0_last_model.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
sd = ckpt.get("model_state_dict", ckpt)
for k, v in sd.items():
    if "granular_pathology_head.3.weight" in k or "pathology" in k:
        print(k, v.shape)
