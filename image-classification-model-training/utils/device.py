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
    Runtime-check whether the given device has native hardware bfloat16 Tensor Core support.
    CUDA requires Compute Capability >= 8.0 (Ampere architecture or newer, e.g. A100, RTX 3000, L4).
    Turing GPUs (like NVIDIA T4, CC 7.5) do NOT have hardware BF16 Tensor Cores, so torch.float16
    must be used to trigger 130 TFLOPS FP16 Tensor Core acceleration.
    """
    if device.type == "cuda":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(device)
            return major >= 8
        return False

    if device.type == "cpu":
        return False

    try:
        t = torch.ones((2, 2), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            _ = t + t
        return True
    except Exception:
        return False


def get_raw_model(model: nn.Module) -> nn.Module:
    """
    Unwraps DataParallel or DistributedDataParallel container to expose raw underlying PyTorch module.
    Prevents AttributeError when calling custom model methods on Parallel wrapped models.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model

class ComputeManager:
    """
    Unified Hardware Compute Manager.
    Encapsulates device routing, DDP / DataParallel model distribution, AMP autocasting,
    and periodic memory cache flushing across MacBook (MPS), Kaggle (CUDA), and CPU.
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_data_parallel: bool = False,
        use_ddp: bool = False,
        cache_flush_interval: Optional[int] = None
    ) -> None:
        import os
        self.is_ddp = use_ddp or ("LOCAL_RANK" in os.environ)
        if self.is_ddp and torch.cuda.is_available():
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            self.is_main_process = (self.rank == 0)
            self.use_data_parallel = False
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.device = device or get_device()
            self.is_main_process = True
            self.use_data_parallel = (
                use_data_parallel and 
                self.device.type == "cuda" and 
                torch.cuda.device_count() > 1
            )
        self.cache_flush_interval = cache_flush_interval

        if self.is_main_process:
            logger.info(
                f"ComputeManager initialized | device={self.device} | DDP={self.is_ddp} (rank={self.rank}/{self.world_size}) | DataParallel={self.use_data_parallel}"
            )

        # Log selected autocast preference (bfloat16 preferred when supported)
        try:
            if self.is_main_process:
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
        """Move model to device and optionally wrap with DDP or DataParallel if requested."""
        model = model.to(self.device)
        if self.is_ddp and torch.cuda.is_available():
            if self.is_main_process:
                logger.info(f"Wrapping model across {self.world_size} GPUs using DistributedDataParallel (DDP).")
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                gradient_as_bucket_view=False
            )
        elif self.use_data_parallel:
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
