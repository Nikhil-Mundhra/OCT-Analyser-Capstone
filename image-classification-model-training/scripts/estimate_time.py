#!/usr/bin/env python3
"""
scripts/estimate_time.py

Parses the orchestration logs and estimates the remaining time for training all L3 specialists.
Runs on CPU only.
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Constants for dataset sizes (samples)
SAMPLES = {
    "Macular": 47107,
    "Diabetic": 11602,
    "Vascular": 1101,
    "Fluid": 102,
    "Structural": 231
}

# Average epochs per fold based on history
AVG_FINETUNE_EPOCHS = 35  # Early stopping avg
WARMUP_EPOCHS = 5

def parse_macular_stats(log_path: Path):
    """Parses the active Macular log to find exact epoch duration."""
    warmup_times = []
    finetune_times = []
    
    if not log_path.exists():
        return None, None
        
    with open(log_path, "r") as f:
        for line in f:
            # Match warm-up epoch line: Ep   0 [warmup|fold0] ... | 349.5s
            m_warm = re.search(r"Ep\s+\d+\s+\[warmup\|fold\d+\].*?\|\s+([\d\.]+)s", line)
            if m_warm:
                warmup_times.append(float(m_warm.group(1)))
                continue
                
            # Match finetune epoch line: Ep   5 [finetune|fold0] ... | 1004.3s
            m_fine = re.search(r"Ep\s+\d+\s+\[finetune\|fold\d+\].*?\|\s+([\d\.]+)s", line)
            if m_fine:
                finetune_times.append(float(m_fine.group(1)))
                
    avg_warm = sum(warmup_times) / len(warmup_times) if warmup_times else 350.0
    avg_fine = sum(finetune_times) / len(finetune_times) if finetune_times else 1000.0
    return avg_warm, avg_fine

def get_current_status(log_path: Path):
    """Finds what fold and epoch are currently active."""
    current_fold = 0
    current_epoch = 0
    phase = "warmup"
    
    if not log_path.exists():
        return 0, 0, "warmup"
        
    with open(log_path, "r") as f:
        lines = f.readlines()
        
    # Read backward to find the latest active state
    for line in reversed(lines):
        # Match e.g.: ###  FOLD 2 / 5  ###
        m_fold = re.search(r"###\s+FOLD\s+(\d+)\s+/\s+5\s+###", line)
        if m_fold and current_fold == 0:
            current_fold = int(m_fold.group(1)) - 1
            
        # Match e.g.: Ep   2 [warmup|fold1]
        m_ep = re.search(r"Ep\s+(\d+)\s+\[(warmup|finetune)\|fold(\d+)\]", line)
        if m_ep:
            ep = int(m_ep.group(1))
            ph = m_ep.group(2)
            fold = int(m_ep.group(3))
            return fold, ep, ph
            
    return current_fold, current_epoch, phase

def main():
    proj_dir = Path(__file__).resolve().parent.parent
    logs_dir = proj_dir / "logs" / "orchestration"
    
    # Get the latest Macular log file
    macular_logs = sorted(logs_dir.glob("train_l3_Macular_*.log"))
    if not macular_logs:
        print("❌ No Macular log files found in logs/orchestration/")
        sys.exit(1)
        
    active_log = macular_logs[-1]
    print(f"Reading active log: {active_log.name}")
    
    avg_warm_macular, avg_fine_macular = parse_macular_stats(active_log)
    print(f"Measured Macular Speed: Warmup = {avg_warm_macular:.1f}s/epoch | Finetune = {avg_fine_macular:.1f}s/epoch")
    
    # Calculate time scaling factor based on samples relative to Macular
    ref_samples = SAMPLES["Macular"]
    
    # Find current progress
    curr_fold, curr_epoch, curr_phase = get_current_status(active_log)
    print(f"Current Status: Fold {curr_fold + 1}/5, Epoch {curr_epoch} ({curr_phase})")
    
    total_remaining_seconds = 0.0
    
    # 1. Calculate remaining time for Macular (active specialist)
    # Folds remaining
    for f in range(curr_fold, 5):
        if f == curr_fold:
            # Active fold: calculate remaining epochs in this fold
            if curr_phase == "warmup":
                # Remaining warmup epochs in this fold
                rem_warm = max(0, WARMUP_EPOCHS - curr_epoch)
                total_remaining_seconds += rem_warm * avg_warm_macular
                total_remaining_seconds += AVG_FINETUNE_EPOCHS * avg_fine_macular
            else:
                # Remaining finetune epochs in this fold
                rem_fine = max(0, (WARMUP_EPOCHS + AVG_FINETUNE_EPOCHS) - curr_epoch)
                total_remaining_seconds += rem_fine * avg_fine_macular
        else:
            # Upcoming folds
            total_remaining_seconds += WARMUP_EPOCHS * avg_warm_macular
            total_remaining_seconds += AVG_FINETUNE_EPOCHS * avg_fine_macular
            
    macular_rem = total_remaining_seconds
    print(f"Estimated remaining for Macular: {str(timedelta(seconds=int(macular_rem)))}")

    # 2. Estimate time for other specialists based on dataset scale
    other_specs = ["Diabetic", "Vascular", "Fluid", "Structural"]
    print("\n--- Upcoming Specialists Estimates ---")
    for spec in other_specs:
        scale = SAMPLES[spec] / ref_samples
        # Scale epoch time linearly with dataset size
        spec_warm_time = avg_warm_macular * scale
        spec_fine_time = avg_fine_macular * scale
        
        # 5 folds, each has 5 warmup + 35 finetune epochs
        spec_total_time = 5 * (WARMUP_EPOCHS * spec_warm_time + AVG_FINETUNE_EPOCHS * spec_fine_time)
        total_remaining_seconds += spec_total_time
        print(f"  - {spec:<12}: {str(timedelta(seconds=int(spec_total_time))):<10} (Scale={scale:.4f}, size={SAMPLES[spec]} images)")
        
    print("\n==================================================")
    print(f"Total Estimated Time Remaining: {str(timedelta(seconds=int(total_remaining_seconds)))}")
    eta = datetime.now() + timedelta(seconds=int(total_remaining_seconds))
    print(f"Estimated Completion Time:      {eta.strftime('%Y-%m-%d %H:%M:%S')}")
    print("==================================================")

if __name__ == "__main__":
    main()
