"""
utils/gpu_mutex.py

Inter-process GPU Mutex Lock using POSIX fcntl file locking.
Ensures that heavy PyTorch GPU workloads (training, validation, Grad-CAM evaluation)
execute strictly sequentially, preventing VRAM OOMs and GPU throttling on Apple Silicon (MPS) and CUDA.
"""

import os
import fcntl
import time
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_default_lock_path() -> Path:
    env_path = os.environ.get("OCT_GPU_MUTEX_LOCK")
    if env_path:
        return Path(env_path)
    try:
        home_lock = Path(os.path.expanduser("~/.oct_gpu_mutex.lock"))
        home_lock.parent.mkdir(parents=True, exist_ok=True)
        with open(home_lock, "a"):
            pass
        return home_lock
    except (OSError, PermissionError):
        return Path(tempfile.gettempdir()) / ".oct_gpu_mutex.lock"


LOCK_FILE_PATH = _get_default_lock_path()


class GPUMutex:
    """
    Context manager for acquiring an exclusive inter-process GPU lock.
    
    Usage:
        with GPUMutex(blocking=True, timeout=None):
            # PyTorch GPU execution (training or evaluation)
            ...
    """
    def __init__(self, lock_file: Path = None, blocking: bool = True, timeout: float = None):
        self.lock_file = lock_file or _get_default_lock_path()
        self.blocking = blocking
        self.timeout = timeout
        self.fd = None

    def __enter__(self):
        try:
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            self.fd = open(self.lock_file, "w")
        except (OSError, PermissionError):
            self.lock_file = Path(tempfile.gettempdir()) / ".oct_gpu_mutex.lock"
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            self.fd = open(self.lock_file, "w")
        
        start_time = time.time()
        flags = fcntl.LOCK_EX
        if not self.blocking:
            flags |= fcntl.LOCK_NB
            
        logger.info(f"Acquiring inter-process GPU Mutex lock ({self.lock_file})...")
        while True:
            try:
                fcntl.flock(self.fd, flags)
                self.fd.write(f"PID: {os.getpid()}\nTime: {time.ctime()}\n")
                self.fd.flush()
                logger.info(f"GPU Mutex lock acquired (PID: {os.getpid()}).")
                return self
            except (IOError, OSError):
                if not self.blocking:
                    logger.warning("Another GPU process is active. Non-blocking lock failed.")
                    raise RuntimeError("GPU Mutex contention: Another GPU process is currently executing.")
                
                if self.timeout and (time.time() - start_time) > self.timeout:
                    raise TimeoutError(f"Timed out after {self.timeout}s waiting for GPU Mutex lock.")
                
                logger.info("Waiting for active GPU task to finish and release lock...")
                time.sleep(3.0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()
                logger.info(f"GPU Mutex lock released (PID: {os.getpid()}).")
            except Exception as e:
                logger.warning(f"Error releasing GPU Mutex lock: {e}")
