#!/usr/bin/env python3
"""
scripts/scout_unet_masked.py

Interactive Image Viewer and File Manager for Scouting Classified-unet-masked/.
Renders Split View of Real (Preprocessed) vs Mask (Attention U-Net Binary)
with automatic red-background alert treatment for suspicious scans.

Usage:
  python scripts/scout_unet_masked.py
  python scripts/scout_unet_masked.py --dir /path/to/Classified-unet-masked --port 8055
"""

import argparse
import os
from pathlib import Path
import sys
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "training" / "classification") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "training" / "classification"))

from data.preprocessing.tuning.scout import (
    get_scout_overview,
    set_unet_masked_dir,
)
import data.preprocessing.tuning.server as tuning_server


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch Image Viewer & File Manager for Classified-unet-masked/."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-unet-masked",
        help="Path to Classified-unet-masked directory.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8055,
        help="Port to serve the viewer on (default: 8055).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Disable automatic browser opening.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_dir = Path(args.dir).resolve()
    if not target_dir.exists():
        print(f"[WARNING] Specified dataset directory does not exist: {target_dir}")
        print("Scout viewer will start in stand-by mode.")

    set_unet_masked_dir(target_dir)

    overview = get_scout_overview(target_dir)

    print("=" * 76)
    print("  CLASSIFIED-UNET-MASKED SCOUT VIEWER & FILE MANAGER")
    print("=" * 76)
    print(f"  Target Dataset : {target_dir}")
    print(f"  Total Scans    : {overview.get('total_scans', 0):,}")
    print(f"  Clean Scans    : {overview.get('clean_count', 0):,}")
    print(f"  Suspicious     : {overview.get('suspicious_count', 0):,} (WARNING / CRITICAL)")
    print(f"  Folders Indexed: {overview.get('folder_count', 0)}")
    print("-" * 76)
    print(f"  Serving at     : http://{args.host}:{args.port}/#scout")
    print("  Controls       : Side-by-Side, Comparison Slider, Overlay Blend")
    print("  Keyboard       : Left / Right Arrow or J / K to navigate scans")
    print("  Alert Visual   : Suspicious scans highlight with RED background")
    print("=" * 76)

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, tuning_server.FineTuningRequestHandler)

    url = f"http://{args.host}:{args.port}/#scout"
    if not args.no_browser:
        def open_browser():
            time.sleep(0.6)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Scout Viewer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    main()
