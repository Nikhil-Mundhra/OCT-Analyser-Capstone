#!/usr/bin/env python3
import time
import re
import sys
import glob
import os
from pathlib import Path
from datetime import datetime, timedelta

def find_latest_agent_log():
    # Looks for the most recently modified task log
    base_dir = Path(os.environ.get("HOME")) / ".gemini" / "antigravity" / "brain"
    if not base_dir.exists():
        return None
        
    task_logs = glob.glob(str(base_dir / "**" / ".system_generated" / "tasks" / "*.log"), recursive=True)
    if not task_logs:
        return None
        
    return max(task_logs, key=os.path.getmtime)

def estimate_eta(log_file):
    print(f"Monitoring log file: {log_file}")
    print("Waiting for training batch updates to measure speed...")
    
    total_epochs = 150
    current_epoch = None
    num_batches = None
    
    # We will measure the time between two batch reports
    first_batch_info = None
    first_time = None
    
    with open(log_file, "r") as f:
        # seek to end
        f.seek(0, os.SEEK_END)
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
                
            # Expected format: "  Epoch 1 | Batch 20/19297 | Cls Loss: ..."
            m = re.search(r"Epoch (\d+) \| Batch (\d+)/(\d+) \|", line)
            if m:
                epoch = int(m.group(1))
                batch = int(m.group(2))
                total_b = int(m.group(3))
                
                if first_batch_info is None:
                    first_batch_info = (epoch, batch)
                    first_time = time.time()
                    print(f"Captured Batch {batch}/{total_b}. Waiting for next report...")
                else:
                    if batch > first_batch_info[1]:
                        second_time = time.time()
                        batches_elapsed = batch - first_batch_info[1]
                        time_elapsed = second_time - first_time
                        
                        # If time elapsed is near 0, it means we read two buffered lines simultaneously.
                        # We wait for the next genuine report instead of calculating an infinite speed.
                        if time_elapsed < 0.5:
                            first_batch_info = (epoch, batch)
                            first_time = second_time
                            continue
                            
                        time_per_batch = time_elapsed / batches_elapsed
                        time_per_epoch = time_per_batch * total_b
                        
                        remaining_batches_this_epoch = total_b - batch
                        remaining_epochs = total_epochs - epoch
                        
                        total_remaining_seconds = (remaining_batches_this_epoch * time_per_batch) + (remaining_epochs * time_per_epoch)
                        
                        print("\n" + "="*50)
                        print(f"Speed: {time_per_batch:.3f} seconds / batch")
                        print(f"Estimated Epoch Time: {timedelta(seconds=int(time_per_epoch))}")
                        print(f"Current Progress: Epoch {epoch}/{total_epochs} | Batch {batch}/{total_b}")
                        print("-" * 50)
                        print(f"Time Remaining: {timedelta(seconds=int(total_remaining_seconds))}")
                        eta = datetime.now() + timedelta(seconds=int(total_remaining_seconds))
                        print(f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
                        print("="*50)
                        return

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = find_latest_agent_log()
        
    if not log_path or not os.path.exists(log_path):
        print("Could not find a log file to monitor. Please provide one as an argument.")
        sys.exit(1)
        
    estimate_eta(log_path)
