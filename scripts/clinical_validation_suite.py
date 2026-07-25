"""
scripts/clinical_validation_suite.py

Automated evaluation pipeline for the OCT Analyzer API.
Validates the end-to-end performance of the system by uploading a test set
of images and computing precision, recall, F1, and AUROC for the clinical reports.
"""

import os
import time
import requests
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def wait_for_scan(api_url: str, scan_id: str, api_key: str, timeout: int = 60) -> dict:
    headers = {"X-API-Key": api_key}
    start = time.time()
    while time.time() - start < timeout:
        res = requests.get(f"{api_url}/api/scans/{scan_id}", headers=headers)
        if res.status_code == 200:
            data = res.json()
            if data["status"] == "completed":
                return data
            elif data["status"] == "failed":
                logging.error(f"Scan {scan_id} failed: {data.get('detail')}")
                return None
        time.sleep(2)
    logging.error(f"Timeout waiting for scan {scan_id}")
    return None

def run_evaluation(api_url: str, test_dir: str, api_key: str):
    """
    Evaluates the system using images in a structured test directory:
    test_dir/
      NORMAL/
        img1.png
        ...
      ABNORMAL/
        img2.png
        ...
    """
    test_path = Path(test_dir)
    if not test_path.exists():
        logging.error(f"Test directory {test_dir} not found.")
        return

    y_true = []
    y_pred = []
    y_scores = []
    
    headers = {"X-API-Key": api_key}

    for label_dir in test_path.iterdir():
        if not label_dir.is_dir():
            continue
            
        true_label = label_dir.name.upper()
        if true_label not in ["NORMAL", "ABNORMAL"]:
            continue
            
        for img_path in label_dir.iterdir():
            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".dcm"]:
                continue
                
            logging.info(f"Processing {img_path.name} (True: {true_label})...")
            
            with open(img_path, "rb") as f:
                files = {"file": f}
                res = requests.post(f"{api_url}/api/scans", files=files, headers=headers)
                
            if res.status_code != 200:
                logging.error(f"Failed to upload {img_path.name}: {res.text}")
                continue
                
            scan_id = res.json()["scan_id"]
            result = wait_for_scan(api_url, scan_id, api_key)
            
            if result:
                diagnosis = result.get("diagnosis", "NORMAL").upper()
                # Assuming diagnosis != NORMAL implies ABNORMAL
                pred_label = "NORMAL" if diagnosis == "NORMAL" else "ABNORMAL"
                confidence = result.get("confidence", 0.0)
                
                y_true.append(1 if true_label == "ABNORMAL" else 0)
                y_pred.append(1 if pred_label == "ABNORMAL" else 0)
                
                # If predicted abnormal, score is confidence. If normal, score is 1 - confidence
                score = confidence if pred_label == "ABNORMAL" else (1.0 - confidence)
                y_scores.append(score)

    if not y_true:
        logging.warning("No test samples processed.")
        return

    logging.info("\n--- CLINICAL VALIDATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "ABNORMAL"]))
    
    try:
        auc = roc_auc_score(y_true, y_scores)
        logging.info(f"AUROC: {auc:.4f}")
    except ValueError:
        logging.warning("Could not compute AUROC (only one class present in test set?).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Validation Suite")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL of the OCT API")
    parser.add_argument("--test-dir", required=True, help="Directory containing NORMAL/ and ABNORMAL/ subfolders")
    parser.add_argument("--api-key", default="sk_oct_analyzer_demo_key", help="API Key for authentication")
    
    args = parser.parse_args()
    run_evaluation(args.api_url, args.test_dir, args.api_key)
