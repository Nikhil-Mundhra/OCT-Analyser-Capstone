import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multi_head_convnext import build_multi_head_model

model = build_multi_head_model(pretrained=False, warmup=False)

print("Model backbone type:", type(model.backbone))
print("Backbone named_children:")
for name, child in model.backbone.named_children():
    print(f"  {name}: {type(child)}")
