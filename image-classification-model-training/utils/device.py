import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    Auto-detect compute device.
    Priority: MPS (Apple Silicon local prototyping) > CUDA (Cloud GPU) > CPU.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def supports_bfloat16(device: torch.device) -> bool:
    """
    Runtime-check whether the given device supports bfloat16 autocasting without raising.

    This performs a lightweight sanity check by running a trivial operation inside a
    torch.autocast context with dtype=torch.bfloat16. If the context or the device
    does not support bfloat16, an exception will be raised and the function returns False.
    """
    if device.type == "cpu":
        # We intentionally avoid enabling autocast on CPU in this project.
        return False

    try:
        # Create a small tensor on the target device and run a trivial op under autocast
        t = torch.ones((2, 2), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            _ = t + t
        return True
    except Exception:
        return False


def get_raw_model(model: nn.Module) -> nn.Module:
    """
    Unwraps DataParallel container to expose raw underlying PyTorch module.
    Prevents AttributeError when calling custom model methods on DataParallel wrapped models.
    """
    return model.module if isinstance(model, nn.DataParallel) else model

class ComputeManager:
    """
    Unified Hardware Compute Manager.
    Encapsulates device routing, DataParallel model distribution, AMP autocasting,
    and periodic memory cache flushing across MacBook (MPS), Kaggle (CUDA), and CPU.
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_data_parallel: bool = False,
        cache_flush_interval: Optional[int] = None
    ) -> None:
        self.device = device or get_device()
        self.use_data_parallel = (
            use_data_parallel and 
            self.device.type == "cuda" and 
            torch.cuda.device_count() > 1
        )
        self.cache_flush_interval = cache_flush_interval

        logger.info(f"ComputeManager initialized on device: {self.device} (DataParallel={self.use_data_parallel})")

        # Log selected autocast preference (bfloat16 preferred when supported)
        try:
            if self.device.type in ("cuda", "mps"):
                if supports_bfloat16(self.device):
                    logger.info("Autocast preference: using torch.bfloat16 on device %s", self.device)
                else:
                    logger.info("Autocast preference: using torch.float16 on device %s", self.device)
            else:
                logger.info("Autocast disabled for CPU device: %s", self.device)
        except Exception as e:
            logger.debug("Failed to determine autocast preference: %s", e)

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Move model to device and optionally wrap with DataParallel if explicitly requested."""
        model = model.to(self.device)
        if self.use_data_parallel:
            logger.info(f"Wrapping model across {torch.cuda.device_count()} GPUs using DataParallel.")
            model = nn.DataParallel(model)
        return model

    def flush_cache(self, batch_idx: Optional[int] = None, force: bool = False) -> None:
        """
        Flushes GPU memory cache.
        If force=True, flushes immediately.
        If batch_idx is specified, flushes only when (batch_idx + 1) matches cache_flush_interval.
        """
        should_flush = force or (
            self.cache_flush_interval is not None and 
            batch_idx is not None and 
            (batch_idx + 1) % self.cache_flush_interval == 0
        )
        if not should_flush:
            return

        if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_autocast_context(self):
        """Returns the appropriate PyTorch autocast context for AMP.

        Preference order:
        - Use torch.bfloat16 autocast on MPS/CUDA if the runtime supports it (recommended on M2 Pro / Apple Silicon).
        - Fall back to torch.float16 autocast when bfloat16 is not supported but the device supports fp16.
        - Autocast is disabled on CPU by default in this project.
        """
        enabled = self.device.type in ("cuda", "mps")

        if not enabled:
            return torch.autocast(device_type="cpu", enabled=False)

        # MPS has native FP16 matrix engines (AMX) but emulates bfloat16 in software,
        # making bfloat16 SLOWER than float32 on Apple Silicon. Prefer float16 on MPS.
        # CUDA has native bfloat16 support, so prefer it there.
        if self.device.type == "cuda" and supports_bfloat16(self.device):
            amp_dtype = torch.bfloat16
        elif self.device.type == "mps":
            amp_dtype = torch.float16
        else:
            amp_dtype = torch.float16

        return torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=True)
