"""
scripts/analyze_dataset.py

Dataset EDA and class distribution analysis.

Run this BEFORE training to verify:
  1. All 86,120 images are discovered correctly.
  2. Class mappings from hierarchy.yaml match the actual directory structure.
  3. Imbalance severity for each hierarchy level.
  4. Class weights that will be fed to FocalLoss.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/analyze_dataset.py
    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/analyze_dataset.py \\
        --config config/hierarchy.yaml
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Separator helpers
# ──────────────────────────────────────────────────────────────────────────────
_W = 65

def _header(title: str) -> None:
    logger.info("=" * _W)
    logger.info("  %s", title)
    logger.info("=" * _W)

def _subheader(title: str) -> None:
    logger.info("")
    logger.info("─── %s ───", title)


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze(config_path: str) -> None:
    from data.dataset import OCTHierarchicalDataset

    _header("OCT Dataset EDA — Hierarchical Classification Pipeline")

    # Build full manifest via level1 mode (all samples, no filtering)
    ds = OCTHierarchicalDataset(
        config_path=config_path,
        mode="level1",
        fold_indices=None,
        transform=None,
    )
    manifest = ds._full_manifest
    total    = len(manifest)

    data_root_display = manifest["image_path"].iloc[0] if not manifest.empty else "N/A"
    logger.info("Config file :  %s", Path(config_path).resolve())
    logger.info("Data root   :  %s", str(Path(data_root_display).parent.parent.parent) if not manifest.empty else "N/A")
    logger.info("Total images:  %d", total)

    # ── Level 1 — Binary ──────────────────────────────────────────────────────
    _subheader("Level 1 — NORMAL vs ABNORMAL")
    l1_counts = manifest["l1"].value_counts()
    max_l1    = l1_counts.max()
    for cls, cnt in l1_counts.sort_index().items():
        bar = "█" * int(30 * cnt / max_l1)
        logger.info("  %-12s │ %6d (%5.1f%%)  %s", cls, cnt, 100 * cnt / total, bar)
    imbalance = l1_counts.max() / l1_counts.min()
    logger.info("  Imbalance ratio: %.1f : 1", imbalance)

    # ── Level 2 — Disease Families ────────────────────────────────────────────
    _subheader("Level 2 — Disease Families (ABNORMAL samples only)")
    abnormal = manifest[manifest["l1"] == "ABNORMAL"]
    n_abn    = len(abnormal)
    l2_counts = abnormal["l2"].value_counts()
    max_l2    = l2_counts.max()
    for family, cnt in l2_counts.sort_values(ascending=False).items():
        bar = "█" * int(30 * cnt / max_l2)
        logger.info("  %-30s │ %6d (%5.1f%%)  %s", family, cnt, 100 * cnt / n_abn, bar)

    # ── Fine-grained classes ──────────────────────────────────────────────────
    _subheader("Fine-grained Class Distribution (All Samples)")
    fine_counts = manifest["fine_class"].value_counts()
    max_fine    = fine_counts.max()
    for cls, cnt in fine_counts.sort_values(ascending=False).items():
        bar     = "█" * int(30 * cnt / max_fine)
        ratio   = max_fine / cnt
        flag    = " ⚠ EXTREME" if ratio > 100 else (" ⚠ HIGH" if ratio > 20 else "")
        logger.info(
            "  %-20s │ %6d (%5.1f%%)  ratio=%-6.0f  %s%s",
            cls, cnt, 100 * cnt / total, ratio, bar, flag,
        )

    # ── L3 Specialist distributions ───────────────────────────────────────────
    _subheader("Level 3 Specialist Distributions")
    specialists = {
        "Macular":    "level3_macular",
        "Diabetic":   "level3_diabetic",
        "Vascular":   "level3_vascular",
        "Fluid":      "level3_fluid",
        "Structural": "level3_structural",
    }
    for spec_name, mode in specialists.items():
        spec_ds = OCTHierarchicalDataset(
            config_path=config_path,
            mode=mode,
            fold_indices=None,
            transform=None,
        )
        logger.info("")
        logger.info("  [%s] — %d samples, %d classes", spec_name, len(spec_ds), spec_ds.num_classes)
        labels     = spec_ds.get_labels()
        names      = spec_ds.class_names
        weights    = spec_ds.class_weights
        max_spec   = max(
            sum(labels == i) for i in range(spec_ds.num_classes)
        )
        for i, name in enumerate(names):
            cnt = int(sum(labels == i))
            bar = "█" * int(20 * cnt / max(max_spec, 1))
            logger.info(
                "    [%d] %-15s │ %5d  weight=%.3f  %s",
                i, name, cnt, weights[i].item(), bar,
            )

    # ── L1 Class weight summary ───────────────────────────────────────────────
    _subheader("L1 Class Weights (for FocalLoss alpha)")
    for name, w in zip(ds.class_names, ds.class_weights):
        logger.info("  %-12s → %.4f", name, w.item())

    # ── Imbalance warnings ────────────────────────────────────────────────────
    _subheader("Imbalance Warnings")
    any_warning = False
    for cls, cnt in fine_counts.items():
        ratio = max_fine / cnt
        if ratio > 500:
            logger.warning("  🔴 CRITICAL: %-18s  %d samples  (%.0fx minority)", cls, cnt, ratio)
            any_warning = True
        elif ratio > 50:
            logger.warning("  🟠 EXTREME:  %-18s  %d samples  (%.0fx minority)", cls, cnt, ratio)
            any_warning = True
        elif ratio > 10:
            logger.warning("  🟡 HIGH:     %-18s  %d samples  (%.0fx minority)", cls, cnt, ratio)
            any_warning = True
    if not any_warning:
        logger.info("  ✅  No severe imbalance detected.")

    logger.info("")
    _header("Analysis Complete")
    logger.info("Next step: KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/train_level1.py --smoke-test")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

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
