import os
import torch
from backend.oct_analyzer.constants import get_compute_device

def get_device():
    """Delegates to the global compute device selector."""
    return get_compute_device()
