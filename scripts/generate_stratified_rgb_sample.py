"""
scripts/generate_stratified_rgb_sample.py

Generates a stratified sample of visual RGB overlay masks (3-5 images per leaf subfolder)
across all subfolders in the Classified dataset for visual inspection.
"""

import os
import sys
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent.parent / "image-classification-model-training"))

from data.preprocessing import process_image, VALID_EXT


def _worker(args):
    src, dst = args
    return process_image(src, dst, overlay_rgb=True, frame=True, frame_size=384)


def build_stratified_sample_jobs(src_root: Path, dst_root: Path, samples_per_dir: int = 4):
    dir_to_files = defaultdict(list)
    for p in sorted(src_root.rglob('*')):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            if p.name.endswith('_mask.png'):
                continue
            dir_to_files[p.parent].append(p)

    jobs = []
    print(f"Found {len(dir_to_files)} leaf subdirectories in dataset.")
    for parent_dir, files in dir_to_files.items():
        # Select evenly spaced samples across each subfolder
        step = max(1, len(files) // samples_per_dir)
        sampled_files = files[::step][:samples_per_dir]
        for p in sampled_files:
            rel = p.relative_to(src_root)
            dst = dst_root / rel
            jobs.append((str(p), str(dst)))

    return jobs


def main():
    src_root = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed")
    dst_root = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed-RGB-Overlay-Sample")

    print(f"\n{'='*65}")
    print(f"  Generating Stratified RGB Overlay Mask Visualizations")
    print(f"{'='*65}")
    print(f"  Source : {src_root}")
    print(f"  Dest   : {dst_root}")
    print(f"{'='*65}\n")

    jobs = build_stratified_sample_jobs(src_root, dst_root, samples_per_dir=4)
    print(f"Selected {len(jobs)} stratified sample images across all subfolders.\n")

    ok = fail = 0
    with mp.Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
        for i, success in enumerate(pool.imap_unordered(_worker, jobs), start=1):
            if success:
                ok += 1
            else:
                fail += 1
            if i % 10 == 0 or i == len(jobs):
                pct = 100 * i / len(jobs)
                print(f"  Progress: {i:>3}/{len(jobs)} ({pct:5.1f}%)  ✓ {ok}  ✗ {fail}", flush=True)

    print(f"\n{'='*65}")
    print(f"  DONE | Processed: {ok} | Output Directory: {dst_root}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    main()
