"""
scripts/preprocess_dataset.py

One-time offline preprocessing pipeline CLI.
Imports processing engine from data.preprocessing and dispatches jobs across CPU processes.
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.preprocessing import process_image, VALID_EXT


def _worker(args):
    src, dst, quality, frame, frame_size, highlight_red, save_mask, overlay_rgb, clear_corners = args
    return process_image(
        src, dst, quality=quality, frame=frame,
        frame_size=frame_size, highlight_red=highlight_red,
        save_mask=save_mask, overlay_rgb=overlay_rgb, clear_corners=clear_corners
    )


def build_job_list(src_root: Path, dst_root: Path):
    jobs = []
    for p in sorted(src_root.rglob('*')):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            if p.name.endswith('_mask.png'):
                continue
            rel = p.relative_to(src_root)
            dst = dst_root / rel
            jobs.append((str(p), str(dst)))
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Offline OCT morphological tissue-mask, framing, & white bar preprocessing CLI")
    parser.add_argument('--src',                type=str, required=True,  help="Source: path to Classified/")
    parser.add_argument('--dst',                type=str, required=True,  help="Dest:   path to preprocessed dataset")
    parser.add_argument('--workers',            type=int, default=max(1, mp.cpu_count() - 2), help="Parallel worker processes")
    parser.add_argument('--quality',            type=int, default=95,     help="JPEG save quality")
    parser.add_argument('--limit',              type=int, default=None,   help="Cap total images (for smoke tests)")
    parser.add_argument('--highlight-red',      action='store_true',     help="Paint detected white bars in BRIGHT RED (evaluation mode)")
    parser.add_argument('--save-masks',         action='store_true',     help="Save binary tissue masks alongside output images (*_mask.png)")
    parser.add_argument('--overlay-rgb',        action='store_true',     help="Generate visual RGB overlay images")
    parser.add_argument('--no-corner-blacking', dest='clear_corners', action='store_false', help="Disable manual corner blacking, relying purely on Otsu tissue mask")
    parser.set_defaults(clear_corners=True)
    parser.add_argument('--frame',              dest='frame', action='store_true', help="Apply letterbox framing & resize")
    parser.add_argument('--no-frame',           dest='frame', action='store_false', help="Disable framing")
    parser.set_defaults(frame=True)
    parser.add_argument('--frame-size',         type=int, default=384, help="Framing target resolution (default: 384)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        print(f"ERROR: Source path does not exist: {src_root}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  OCT Preprocessing (Tissue Mask, White Bar Removal, Framing)")
    print(f"{'='*60}")
    print(f"  Source          : {src_root}")
    print(f"  Dest            : {dst_root}")
    print(f"  Workers         : {args.workers}")
    print(f"  Save Masks      : {args.save_masks}")
    print(f"  RGB Overlay     : {args.overlay_rgb}")
    print(f"  Corner Blacking : {args.clear_corners}")
    print(f"  Highlight Red   : {args.highlight_red}")
    print(f"  Framing         : {args.frame} (Target Resolution: {args.frame_size}x{args.frame_size})")
    print(f"{'='*60}\n")

    jobs = build_job_list(src_root, dst_root)
    if args.limit:
        jobs = jobs[:args.limit]
        print(f"  Limit         : {args.limit} images\n")

    total = len(jobs)
    print(f"Found {total:,} images to process.\n")
    if total == 0:
        print("No images found. Check --src path.")
        sys.exit(1)

    jobs_with_args = [
        (s, d, args.quality, args.frame, args.frame_size, args.highlight_red, args.save_masks, args.overlay_rgb, args.clear_corners)
        for s, d in jobs
    ]

    ok = fail = 0
    report_every = max(1, total // 20)

    with mp.Pool(processes=args.workers) as pool:
        for i, success in enumerate(pool.imap_unordered(_worker, jobs_with_args), start=1):
            if success: ok += 1
            else: fail += 1
            if i % report_every == 0 or i == total:
                pct = 100 * i / total
                print(f"  Progress: {i:>6,}/{total:,}  ({pct:5.1f}%)  ✓ {ok:,}  ✗ {fail:,}", flush=True)

    print(f"\n{'='*60}")
    print(f"  DONE | Processed: {ok:,} | Failed: {fail:,} | Output: {dst_root}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    main()
