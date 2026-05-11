"""Runtime guards for local scientific Python stacks."""

import os
import platform


def configure_runtime():
    if platform.system() == "Darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
