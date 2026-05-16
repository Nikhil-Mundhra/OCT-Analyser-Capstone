from .runtime import configure_runtime

configure_runtime()

import torch
from .model import get_3d_relaynet
from .losses import OrdinalAnatomicalLoss

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_3d_relaynet(num_layers=10).to(device)
criterion = OrdinalAnatomicalLoss(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_step(batch_data):
    # batch_data['image'] is the flattened tensor from Phase 1
    inputs = batch_data["image"].to(device)
    labels = batch_data["label"].to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()
