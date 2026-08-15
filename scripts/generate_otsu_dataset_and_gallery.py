"""
scripts/generate_otsu_dataset_and_gallery.py

Generates a representative dataset with Otsu-initialized tissue masking (no corner blacking)
inside Classified-preprocessed-Otsu/ and builds an interactive HTML gallery (gallery.html)
to visually inspect tissue isolation across all disease classes.

Masking pipeline (via data.preprocessing.generate_tissue_mask):
  Stage 1 - Forgiving three-gate outlier rejection:
    Gate 1: Hampel MAD + bilateral neighbor test with intentionally wide thresholds
            (top: scale=3.0/thresh=12, bottom: scale=5.0/thresh=22).
    Gate 2: Width filter - any flagged run >= 3 consecutive columns is released
            unconditionally (real anatomy is wide; pure noise is narrow).
    Gate 3: Intensity validation - flagged points with a supporting reflectivity
            gradient in the raw image are preserved without interpolation.
  Stage 2 - Boundary-specific smoothing:
    Top boundary:    minimum_filter1d(size=9) to track ILM / foveal pit.
    Bottom boundary: Savitzky-Golay(window=31, poly=2) to smooth choroidal stair-steps.
"""

import os
import sys
import json
import base64
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent.parent / "training" / "classification"))

from data.preprocessing import process_image, VALID_EXT


def _worker(args):
    src, dst = args
    # Delegates to process_image -> generate_tissue_mask (three-gate forgiving pipeline).
    # clear_corners=False: no corner blacking; save_mask=True: writes _mask.png alongside.
    return process_image(
        src, dst, quality=95, frame=True, frame_size=384,
        save_mask=True, clear_corners=False
    )


def build_representative_jobs(src_root: Path, dst_root: Path, samples_per_dir: int = 5):
    dir_to_files = defaultdict(list)
    for p in sorted(src_root.rglob('*')):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            if p.name.endswith('_mask.png'):
                continue
            dir_to_files[p.parent].append(p)

    jobs = []
    print(f"Found {len(dir_to_files)} leaf subdirectories in dataset.")
    for parent_dir, files in dir_to_files.items():
        step = max(1, len(files) // samples_per_dir)
        sampled_files = files[::step][:samples_per_dir]
        for p in sampled_files:
            rel = p.relative_to(src_root)
            dst = dst_root / rel
            jobs.append((str(p), str(dst)))

    return jobs


def generate_html_gallery(src_root: Path, dst_root: Path, output_html: Path):
    """
    Scans dst_root for preprocessed Otsu images and matches them with raw images in src_root
    to build an interactive side-by-side comparison gallery.
    """
    records = []
    for p in sorted(dst_root.rglob('*')):
        if p.is_file() and p.suffix.lower() in VALID_EXT and not p.name.endswith('_mask.png'):
            rel_img = p.relative_to(dst_root)
            
            # Locate raw original image in src_root
            raw_img_path = src_root / rel_img
            if not raw_img_path.exists():
                raw_parent = src_root / rel_img.parent
                if raw_parent.exists():
                    matches = [m for m in raw_parent.glob(f"{p.stem}.*") if m.suffix.lower() in VALID_EXT]
                    if matches:
                        raw_img_path = matches[0]

            rel_raw = os.path.relpath(raw_img_path, dst_root) if raw_img_path.exists() else ""

            # Determine top-level disease category from relative path
            parts = rel_img.parts
            category = parts[0] if len(parts) > 1 else "General"
            disease = parts[1] if len(parts) > 2 else category

            records.append({
                "name": p.name,
                "category": category,
                "disease": disease,
                "img_rel": str(rel_img),
                "raw_rel": str(rel_raw)
            })

    categories = sorted(list(set(r["category"] for r in records)))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pure Otsu Tissue Masking vs Raw Classified Scans</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --subtext: #94a3b8;
            --primary: #38bdf8;
            --border: #334155;
            --accent: #22c55e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            padding: 2rem;
            line-height: 1.5;
        }}
        header {{
            max-width: 1400px;
            margin: 0 auto 2rem auto;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
        }}
        h1 {{ font-size: 2rem; font-weight: 700; color: var(--primary); margin-bottom: 0.5rem; }}
        p.subtitle {{ color: var(--subtext); font-size: 1rem; }}
        .badge {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.1);
            color: var(--primary);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.3);
            margin-top: 0.75rem;
        }}
        .controls {{
            max-width: 1400px;
            margin: 0 auto 2rem auto;
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}
        .btn {{
            background: var(--card-bg);
            color: var(--subtext);
            border: 1px solid var(--border);
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        .btn:hover, .btn.active {{
            background: var(--primary);
            color: #0f172a;
            border-color: var(--primary);
            font-weight: 600;
        }}
        .grid {{
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 1.5rem;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
            border-color: var(--primary);
        }}
        .card-header {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.85rem;
        }}
        .card-title {{ font-weight: 600; color: var(--text); truncate: ellipsis; overflow: hidden; white-space: nowrap; }}
        .card-meta {{ color: var(--subtext); font-size: 0.75rem; margin-top: 0.25rem; }}
        .image-pair {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px;
            background: #000;
        }}
        .img-box {{
            position: relative;
            aspect-ratio: 1/1;
            background: #000;
        }}
        .img-box img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }}
        .img-label {{
            position: absolute;
            bottom: 0.35rem;
            left: 0.35rem;
            background: rgba(15, 23, 42, 0.85);
            color: var(--text);
            font-size: 0.7rem;
            padding: 0.15rem 0.4rem;
            border-radius: 0.25rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .img-label.otsu {{
            border-color: var(--accent);
            color: var(--accent);
        }}
    </style>
</head>
<body>
    <header>
        <h1>Pure Otsu Tissue Masking vs Raw Original Classified Scans</h1>
        <p class="subtitle">Side-by-Side Comparison: Pure Otsu Tissue Isolation (No Corner Blacking) vs Raw Original B-Scan Input.</p>
        <div class="badge">Representative Dataset Sample: {len(records)} Scans</div>
    </header>

    <div class="controls">
        <button class="btn active" onclick="filterCategory('all')">All ({len(records)})</button>
"""
    for cat in categories:
        count = sum(1 for r in records if r["category"] == cat)
        html_content += f'        <button class="btn" onclick="filterCategory(\'{cat}\')">{cat} ({count})</button>\n'

    html_content += """    </div>

    <div class="grid" id="galleryGrid">
"""
    for idx, r in enumerate(records):
        html_content += f"""
        <div class="card" data-category="{r['category']}">
            <div class="card-header">
                <div class="card-title">{r['name']}</div>
                <div class="card-meta">{r['category']} &rarr; {r['disease']}</div>
            </div>
            <div class="image-pair">
                <div class="img-box">
                    <img src="{r['img_rel']}" alt="Pure Otsu Tissue Image" loading="lazy">
                    <div class="img-label otsu">Otsu Isolated Tissue</div>
                </div>
                <div class="img-box">
                    <img src="{r['raw_rel']}" alt="Raw Classified Scan" loading="lazy">
                    <div class="img-label">Raw Classified Scan</div>
                </div>
            </div>
        </div>
"""

    html_content += """
    </div>

    <script>
        function filterCategory(cat) {
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            
            document.querySelectorAll('.card').forEach(card => {
                if (cat === 'all' || card.getAttribute('data-category') === cat) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated HTML Gallery at: {output_html}")


def main():
    src_root = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified")
    dst_root = Path("/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed-Otsu")

    print(f"\n{'='*70}")
    print(f"  Generating Pure Otsu Preprocessed Dataset (No Corner Blacking)")
    print(f"{'='*70}")
    print(f"  Source : {src_root}")
    print(f"  Dest   : {dst_root}")
    print(f"{'='*70}\n")

    jobs = build_representative_jobs(src_root, dst_root, samples_per_dir=6)
    print(f"Selected {len(jobs)} representative sample images across all subfolders.\n")

    ok = fail = 0
    with mp.Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
        for i, success in enumerate(pool.imap_unordered(_worker, jobs), start=1):
            if success: ok += 1
            else: fail += 1
            if i % 15 == 0 or i == len(jobs):
                pct = 100 * i / len(jobs)
                print(f"  Progress: {i:>3}/{len(jobs)} ({pct:5.1f}%)  ✓ {ok}  ✗ {fail}", flush=True)

    print(f"\n{'='*70}")
    print(f"  DONE | Processed: {ok} | Output Directory: {dst_root}")
    print(f"{'='*70}\n")

    gallery_path = dst_root / "gallery.html"
    generate_html_gallery(src_root, dst_root, gallery_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    main()
