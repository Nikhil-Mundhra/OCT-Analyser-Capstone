import os
import sys
import gc
import subprocess
import torch

# Hardwire OpenMP runtime safety
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def enforce_single_instance_and_clean_memory(script_name: str):
    """
    Hardwired Memory & Process Guard:
    1. Scans system processes and terminates any previous/orphaned python instances of script_name.
    2. Clears Python garbage collection (gc.collect()).
    3. Flushes PyTorch CUDA/MPS GPU memory cache.
    """
    current_pid = os.getpid()
    print(f"[MemoryGuard] Hardwired cleanup initiated for PID {current_pid} ({script_name})...", flush=True)

    # 1. Kill duplicate/orphaned runs of the same script using pkill
    try:
        cmd = f"pgrep -f {script_name}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout:
            pids = [int(p) for p in res.stdout.strip().split() if p.isdigit()]
            for pid in pids:
                if pid != current_pid:
                    print(f"[MemoryGuard] Killing orphaned process PID {pid}", flush=True)
                    subprocess.run(f"kill -9 {pid}", shell=True, capture_output=True)
    except Exception as e:
        print(f"[MemoryGuard] Note during process scan: {e}", flush=True)

    # 2. Hardwire Memory Cleanup
    clean_gpu_memory()
    print(f"[MemoryGuard] Memory cleanup complete. System safe for training.", flush=True)

def clean_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
