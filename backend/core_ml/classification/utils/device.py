import torch

def get_device():
    """
    Handles device routing across different environments.
    Strictly falls back to MPS (Metal Performance Shaders) for local unified memory,
    while remaining fully compatible with the CUDA requirement of @spaces.GPU
    when deployed to Hugging Face ZeroGPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda") # For ZeroGPU deployment / Colab
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # For local Apple Silicon prototyping
    else:
        return torch.device("cpu")
