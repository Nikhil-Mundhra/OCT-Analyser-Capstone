"""
tests/test_scout_viewer.py

Comprehensive test suite for the Classified-unet-masked Image Viewer & File Manager.
Tests backend dataset indexing, tree construction, anomaly filtering, secure file serving,
in-process HTTP API endpoints, and frontend HTML DOM contracts.
"""

import io
import json
import os
from pathlib import Path
import tempfile
import urllib.parse
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "training" / "classification") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "training" / "classification"))

from data.preprocessing.tuning.scout import (
    get_scout_detail,
    get_scout_overview,
    get_scout_tree,
    get_unet_masked_dir,
    invalidate_scout_cache,
    query_scout_images,
    resolve_scout_file,
    set_unet_masked_dir,
)
from data.preprocessing.tuning.diagnostics import dispatch_in_process_request


def create_mock_unet_masked_dir(base_path: Path):
    """Creates a mock Classified-unet-masked directory with clean and suspicious scans."""
    dataset_dir = base_path / "Classified-unet-masked"
    images_dir = dataset_dir / "Images" / "DME" / "Sub1"
    masks_dir = dataset_dir / "Masks" / "DME" / "Sub1"
    quarantine_dir = dataset_dir / "quarantine_review"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean scan (384x384 continuous band)
    clean_img = np.full((384, 384, 3), 120, dtype=np.uint8)
    clean_mask = np.zeros((384, 384), dtype=np.uint8)
    clean_mask[140:240, :] = 255
    cv2.imwrite(str(images_dir / "clean_scan.png"), clean_img)
    cv2.imwrite(str(masks_dir / "clean_scan.png"), clean_mask)

    # 2. Suspicious scan (Floaters / multiple disjoint components)
    susp_img = np.full((384, 384, 3), 100, dtype=np.uint8)
    susp_mask = np.zeros((384, 384), dtype=np.uint8)
    susp_mask[150:200, 50:150] = 255
    susp_mask[20:30, 20:30] = 255  # floater 1
    susp_mask[320:340, 300:320] = 255  # floater 2
    cv2.imwrite(str(images_dir / "suspicious_scan.png"), susp_img)
    cv2.imwrite(str(masks_dir / "suspicious_scan.png"), susp_mask)

    # Write manifest.json
    manifest_data = {
        "version": "1.0",
        "dataset": "Classified-unet-masked",
        "total_samples": 2,
        "samples": {
            "DME/Sub1/clean_scan.png": {
                "image_path": "Images/DME/Sub1/clean_scan.png",
                "mask_path": "Masks/DME/Sub1/clean_scan.png",
                "severity": "CLEAN",
                "is_suspicious": False,
                "anomaly_flags": [],
                "metrics": {
                    "area_ratio": 0.26,
                    "thickness_mean": 100.0,
                    "foreground_components": 1,
                    "max_dy_top": 0.0,
                    "max_dy_bot": 0.0,
                    "pitch_black_violations": 0,
                },
            },
            "DME/Sub1/suspicious_scan.png": {
                "image_path": "Images/DME/Sub1/suspicious_scan.png",
                "mask_path": "Masks/DME/Sub1/suspicious_scan.png",
                "severity": "WARNING",
                "is_suspicious": True,
                "anomaly_flags": ["SUSPICIOUS_FLOATERS_OR_FRAGMENTATION"],
                "metrics": {
                    "area_ratio": 0.08,
                    "thickness_mean": 50.0,
                    "foreground_components": 3,
                    "max_dy_top": 5.0,
                    "max_dy_bot": 5.0,
                    "pitch_black_violations": 0,
                },
            },
        },
    }
    with open(dataset_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    summary_data = {
        "total_processed": 2,
        "clean_count": 1,
        "warning_count": 1,
        "critical_count": 0,
        "flags_tally": {"SUSPICIOUS_FLOATERS_OR_FRAGMENTATION": 1},
    }
    with open(dataset_dir / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f)

    set_unet_masked_dir(dataset_dir)
    return dataset_dir


def test_scout_overview_and_indexing(mock_dir):
    """Verifies that scout overview correctly indexes clean vs suspicious scans."""
    overview = get_scout_overview(mock_dir)
    assert overview["total_scans"] == 2, f"Expected 2, got {overview['total_scans']}"
    assert overview["clean_count"] == 1
    assert overview["warning_count"] == 1
    assert overview["suspicious_count"] == 1
    assert "SUSPICIOUS_FLOATERS_OR_FRAGMENTATION" in overview["flags_tally"]
    assert overview["flags_tally"]["SUSPICIOUS_FLOATERS_OR_FRAGMENTATION"] == 1


def test_scout_folder_tree(mock_dir):
    """Verifies that the folder hierarchy aggregates scan counts correctly."""
    tree = get_scout_tree(mock_dir)
    assert len(tree) == 1
    assert tree[0]["folder"] == "DME/Sub1"
    assert tree[0]["total"] == 2
    assert tree[0]["clean"] == 1
    assert tree[0]["suspicious"] == 1


def test_query_scout_images_filtering(mock_dir):
    """Verifies image querying with filtering by clean, suspicious, and search text."""
    # 1. All images
    res_all = query_scout_images(mock_dir, filter_type="all")
    assert res_all["total"] == 2

    # 2. Suspicious only
    res_susp = query_scout_images(mock_dir, filter_type="suspicious")
    assert res_susp["total"] == 1
    assert res_susp["images"][0]["filename"] == "suspicious_scan.png"
    assert res_susp["images"][0]["is_suspicious"] is True
    assert "SUSPICIOUS_FLOATERS_OR_FRAGMENTATION" in res_susp["images"][0]["anomaly_flags"]

    # 3. Clean only
    res_clean = query_scout_images(mock_dir, filter_type="clean")
    assert res_clean["total"] == 1
    assert res_clean["images"][0]["filename"] == "clean_scan.png"
    assert res_clean["images"][0]["is_suspicious"] is False

    # 4. Search query
    res_search = query_scout_images(mock_dir, query="clean")
    assert res_search["total"] == 1
    assert res_search["images"][0]["filename"] == "clean_scan.png"


def test_resolve_scout_file_security(mock_dir):
    """Verifies secure file resolution and blocks directory traversal."""
    clean_p = resolve_scout_file(mock_dir, "image", "DME/Sub1/clean_scan.png")
    assert clean_p is not None and clean_p.exists()

    mask_p = resolve_scout_file(mock_dir, "mask", "DME/Sub1/clean_scan.png")
    assert mask_p is not None and mask_p.exists()

    # Directory traversal attempts must return None
    assert resolve_scout_file(mock_dir, "image", "../../../etc/passwd") is None
    assert resolve_scout_file(mock_dir, "mask", "../../secrets.txt") is None


def test_scout_http_endpoints_in_process(mock_dir):
    """Tests scout HTTP endpoints using in-process dispatch."""
    # 1. Overview endpoint
    status, _, body = dispatch_in_process_request("/api/scout/overview", "GET")
    assert status == 200
    data = json.loads(body.decode("utf-8"))
    assert data["total_scans"] == 2
    assert data["suspicious_count"] == 1

    # 2. Folder tree endpoint
    status, _, body = dispatch_in_process_request("/api/scout/tree", "GET")
    assert status == 200
    tree = json.loads(body.decode("utf-8"))
    assert len(tree) == 1
    assert tree[0]["folder"] == "DME/Sub1"

    # 3. Query images endpoint with suspicious filter
    status, _, body = dispatch_in_process_request("/api/scout/images?filter=suspicious", "GET")
    assert status == 200
    res = json.loads(body.decode("utf-8"))
    assert res["total"] == 1
    assert res["images"][0]["is_suspicious"] is True

    # 4. File serving endpoint for image
    img_rel = "DME/Sub1/clean_scan.png"
    status, _, body = dispatch_in_process_request(f"/api/scout/file?type=image&path={urllib.parse.quote(img_rel)}", "GET")
    assert status == 200
    assert len(body) > 0

    # 5. File serving endpoint for mask
    status, _, body = dispatch_in_process_request(f"/api/scout/file?type=mask&path={urllib.parse.quote(img_rel)}", "GET")
    assert status == 200
    assert len(body) > 0

    # 6. Detail endpoint
    status, _, body = dispatch_in_process_request(f"/api/scout/detail?path={urllib.parse.quote(img_rel)}", "GET")
    assert status == 200
    detail = json.loads(body.decode("utf-8"))
    assert detail["status"] == "success"
    assert detail["severity"] == "CLEAN"

    # 7. Surgical background ablation endpoint
    ablate_payload = json.dumps({
        "rel_path": "DME/Sub1/suspicious_scan.png",
        "x": 25,
        "y": 25,
        "radius_x": 60,
    }).encode("utf-8")
    status, _, body = dispatch_in_process_request("/api/scout/ablate_bg", "POST", body=ablate_payload)
    assert status == 200
    ablate_res = json.loads(body.decode("utf-8"))
    assert ablate_res["status"] == "success"
    assert ablate_res["can_undo"] is True

    # 8. Undo ablation endpoint
    undo_payload = json.dumps({
        "rel_path": "DME/Sub1/suspicious_scan.png",
    }).encode("utf-8")
    status, _, body = dispatch_in_process_request("/api/scout/undo_ablate", "POST", body=undo_payload)
    assert status == 200
    undo_res = json.loads(body.decode("utf-8"))
    assert undo_res["status"] == "success"
    assert undo_res["can_undo"] is False


def test_dashboard_html_contains_scout_viewer_contract():
    """Verifies that index.html contains the scout navigation tab, containers, and assets."""
    html_path = PROJECT_ROOT / "training" / "classification" / "data" / "preprocessing" / "tuning" / "dashboard" / "index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    # Contract checks:
    assert 'id="nav-tab-scout"' in content, "Navigation tab for scout missing"
    assert 'id="scout-explorer-view"' in content, "Main scout container missing"
    assert 'id="scout-stage-viewport"' in content, "Stage viewport missing"
    assert 'id="scout-anomaly-banner"' in content, "Anomaly warning banner missing"
    assert 'id="scout-real-img-side"' in content, "Real image side element missing"
    assert 'id="scout-mask-img-side"' in content, "Mask image side element missing"
    assert 'id="scout-split-slider"' in content, "Split slider container missing"
    assert 'id="scout-overlay-view"' in content, "Overlay container missing"
    assert 'id="scout-tool-ablate"' in content, "Surgical BG Eraser button missing"
    assert 'id="scout-btn-undo"' in content, "Undo Eraser button missing"
    assert 'scout.css' in content, "scout.css link missing"
    assert 'scout.js' in content, "scout.js script link missing"


def run_all():
    print("=" * 70)
    print("RUNNING CLASSIFIED-UNET-MASKED SCOUT VIEWER TEST SUITE")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        mock_dir = create_mock_unet_masked_dir(tmp_dir)

        tests = [
            ("test_scout_overview_and_indexing", lambda: test_scout_overview_and_indexing(mock_dir)),
            ("test_scout_folder_tree", lambda: test_scout_folder_tree(mock_dir)),
            ("test_query_scout_images_filtering", lambda: test_query_scout_images_filtering(mock_dir)),
            ("test_resolve_scout_file_security", lambda: test_resolve_scout_file_security(mock_dir)),
            ("test_scout_http_endpoints_in_process", lambda: test_scout_http_endpoints_in_process(mock_dir)),
            ("test_dashboard_html_contains_scout_viewer_contract", test_dashboard_html_contains_scout_viewer_contract),
        ]

        passed = 0
        failed = 0
        for name, fn in tests:
            try:
                fn()
                print(f"  [PASS] {name}")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                failed += 1

        print("-" * 70)
        print(f"Summary: {passed} passed, {failed} failed")
        print("=" * 70)
        if failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    run_all()
