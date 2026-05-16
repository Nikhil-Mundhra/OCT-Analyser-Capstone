from .runtime import configure_runtime

configure_runtime()

import torch
import torch.nn as nn

class OrdinalAnatomicalLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        """
        predictions: (Batch, Classes, Z, Y, X) - Logits from ReLayNet
        targets: (Batch, 1, Z, Y, X) - Ground truth indices
        """
        # 1. Standard Dice/Cross-Entropy for basic overlap
        ce_loss = nn.CrossEntropyLoss()(predictions, targets.squeeze(1).long())
        
        # 2. Ordinal Penalty: Ensure the model respects the 1-10 sequence
        probs = torch.softmax(predictions, dim=1)
        class_indices = torch.arange(self.num_classes).to(predictions.device)
        class_indices = class_indices.view(1, -1, 1, 1, 1)
        
        # Calculate the "Expected Layer Index" for each pixel
        expected_index = torch.sum(probs * class_indices, dim=1, keepdim=True)
        
        # Penalize the Mean Squared Error between the expected index and the true index
        ordinal_penalty = torch.mean((expected_index - targets.float())**2)
        
        return ce_loss + (0.1 * ordinal_penalty)
