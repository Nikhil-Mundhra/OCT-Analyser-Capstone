"""
scripts/analyze_dataset.py

Dataset EDA and class distribution analysis for the Multi-Head ConvNeXt Pipeline.

Usage:
    python3 scripts/analyze_dataset.py
    python3 scripts/analyze_dataset.py --config config/hierarchy.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Workaround for Homebrew Python + OpenMP conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import MultiHeadOCTDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def _header(title: str) -> None:
    logger.info("=" * 65)
    logger.info("  %s", title)
    logger.info("=" * 65)

def _subheader(title: str) -> None:
    logger.info("")
    logger.info("─── %s ───", title)

def analyze(config_path: str) -> None:
    _header("OCT Dataset EDA — Multi-Head Pipeline")

    # Build dataset which automatically generates the manifest
    ds = MultiHeadOCTDataset(config_path=config_path, transform=None)
    manifest = ds._manifest
    total = len(manifest)

    logger.info("Config file :  %s", Path(config_path).resolve())
    logger.info("Total images:  %d", total)

    if total == 0:
        logger.error("No images found. Check your OCT_DATA_ROOT or config mapping.")
        return

    # ── Head 1 — Binary (NORMAL vs ABNORMAL) ──────────────────────────────────
    _subheader("Head 1 — NORMAL (0) vs ABNORMAL (1)")
    l1_counts = manifest["l1_idx"].value_counts().sort_index()
    max_l1 = l1_counts.max()
    for cls_idx, cnt in l1_counts.items():
        name = "ABNORMAL" if cls_idx == 1 else "NORMAL"
        bar = "█" * int(30 * cnt / max_l1)
        logger.info("  [%d] %-12s │ %6d (%5.1f%%)  %s", cls_idx, name, cnt, 100 * cnt / total, bar)

    imbalance = l1_counts.max() / l1_counts.min() if l1_counts.min() > 0 else 0
    logger.info("  Imbalance ratio: %.1f : 1", imbalance)

    # ── Head 2 — Pathology (12-Class) ─────────────────────────────────────────
    _subheader("Head 2 — Pathology (ABNORMAL samples only)")
    abnormal = manifest[manifest["granular_idx"] != -1]
    n_abn = len(abnormal)
    
    if n_abn > 0:
        h2_counts = abnormal["granular_idx"].value_counts().sort_index()
        max_h2 = h2_counts.max()
        
        # Invert the granular dictionary to get names
        idx_to_name = {v: k for k, v in ds._granular_classes.items()}
        
        for cls_idx, cnt in h2_counts.items():
            name = idx_to_name.get(cls_idx, f"Unknown_{cls_idx}")
            bar = "█" * int(30 * cnt / max_h2)
            ratio = max_h2 / cnt if cnt > 0 else float('inf')
            flag = " ⚠ EXTREME" if ratio > 100 else (" ⚠ HIGH" if ratio > 20 else "")
            
            logger.info("  [%2d] %-15s │ %6d (%5.1f%%)  ratio=%-4.0f  %s%s", 
                        cls_idx, name, cnt, 100 * cnt / n_abn, ratio, bar, flag)
    else:
        logger.info("  No abnormal samples found.")

    logger.info("")
    _header("Analysis Complete")
    logger.info("Next step: Start training on Kaggle!")

def main() -> None:
    parser = argparse.ArgumentParser(description="OCT Dataset EDA")
    parser.add_argument(
        "--config",
        default="config/hierarchy.yaml",
        help="Path to hierarchy.yaml (default: config/hierarchy.yaml)",
    )
    args = parser.parse_args()
    analyze(args.config)

if __name__ == "__main__":
    main()
