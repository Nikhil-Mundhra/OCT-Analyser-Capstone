"""
scripts/tuning_server.py

Ergonomic root entrypoint for the OCT Preprocessing Tuning Server.
Delegates to the dedicated data.preprocessing.tuning package.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "image-classification-model-training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "image-classification-model-training"))

import data.preprocessing.tuning.server as _server

# Point module reference in sys.modules to the real server module
sys.modules[__name__] = _server

if __name__ == "__main__":
    _server.main()
