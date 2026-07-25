#!/usr/bin/env python3
"""
scripts/estimate_time.py

Parses the multi-task training logs and estimates the remaining time for the full 35-hour training run.
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta

def main():
    if len(sys.argv) < 2:
        print("Usage: python estimate_time.py <path_to_train_mtl.log>")
        sys.exit(1)
        
    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        sys.exit(1)
        
    epoch_times = []
    current_epoch = 0
    total_epochs = 50
    
    with open(log_path, "r") as f:
        for line in f:
            # Match e.g.: --- Epoch 2/50 Summary (Time: 2541.32s) ---
            m_time = re.search(r"--- Epoch (\d+)/(\d+) Summary \(Time: ([\d\.]+)s\) ---", line)
            if m_time:
                current_epoch = int(m_time.group(1))
                total_epochs = int(m_time.group(2))
                epoch_times.append(float(m_time.group(3)))
                
    if not epoch_times:
        print("Waiting for the first epoch to complete to generate a time estimate...")
        sys.exit(0)
        
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = total_epochs - current_epoch
    
    total_remaining_seconds = remaining_epochs * avg_epoch_time
    
    print(f"Current Status: Epoch {current_epoch}/{total_epochs}")
    print(f"Average Epoch Time: {avg_epoch_time/60:.1f} minutes")
    print("\n==================================================")
    print(f"Total Estimated Time Remaining: {str(timedelta(seconds=int(total_remaining_seconds)))}")
    eta = datetime.now() + timedelta(seconds=int(total_remaining_seconds))
    print(f"Estimated Completion Time:      {eta.strftime('%Y-%m-%d %H:%M:%S')}")
    print("==================================================")

if __name__ == "__main__":
    main()
