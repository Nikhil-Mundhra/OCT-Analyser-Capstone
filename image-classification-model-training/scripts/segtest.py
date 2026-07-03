import sys
import numpy as np
import torch

from pathlib import Path
sys.path.append(str(Path(".").resolve()))
from models.level1_gatekeeper import build_gatekeeper

l1_model = build_gatekeeper(pretrained=False)
tensor_224 = torch.randn(1, 3, 224, 224, requires_grad=True)

print("Forward pass...")
feats = l1_model.features(tensor_224)
print("Features created!")

print("Checking nonzero...")
try:
    c = torch.count_nonzero(feats)
    print(c)
except Exception as e:
    print(e)
