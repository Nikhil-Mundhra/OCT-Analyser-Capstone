#!/usr/bin/env python3
"""
scripts/run_all_tests.py

Unified, noise-free test runner and telemetry aggregator for OCT-Analyser.
Executes the full test battery across all subsystems:
  1. Backend & Ingestion API (web-app/backend/tests)
  2. Production Models Suite (tests/test_models_suite.py)
  3. Preprocessing Tuning & Model Registry (tests/)
  4. Disease Classification Training (training/classification/tests)
  5. Retinal Segmentation Training (training/segmentation/tests)

Usage:
  python3 scripts/run_all_tests.py
  python3 scripts/run_all_tests.py -k test_models
  python3 scripts/run_all_tests.py --verbose
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configure environment for quiet execution
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# 1. Add site-packages from local virtual environments
p1 = list((WORKSPACE_ROOT / ".venv" / "lib").glob("python*/site-packages"))
if p1:
    sys.path.insert(0, str(p1[0]))

p2 = list((WORKSPACE_ROOT / "training" / "classification" / "venv" / "lib").glob("python*/site-packages"))
if p2:
    sys.path.insert(0, str(p2[0]))

# 2. Add project source modules to sys.path
for subpath in [
    WORKSPACE_ROOT,
    WORKSPACE_ROOT / "web-app",
    WORKSPACE_ROOT / "web-app" / "backend",
    WORKSPACE_ROOT / "training" / "classification",
    WORKSPACE_ROOT / "training" / "segmentation",
    WORKSPACE_ROOT / "models_suite",
    WORKSPACE_ROOT / "scripts"
]:
    if str(subpath) not in sys.path:
        sys.path.insert(0, str(subpath))

try:
    import pytest
except ImportError:
    print("Error: 'pytest' is required. Run via 'uv run --with pytest' or activate a venv.", file=sys.stderr)
    sys.exit(1)

try:
    import torch
    if torch.cuda.is_available():
        DEVICE_NAME = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE_NAME = "MPS (Apple Silicon GPU)"
    else:
        DEVICE_NAME = "CPU"
    TORCH_VER = torch.__version__
except Exception:
    DEVICE_NAME = "Unknown"
    TORCH_VER = "N/A"


class TelemetryCollector:
    """Pytest plugin collecting execution metrics per subsystem without terminal noise."""
    
    def __init__(self):
        self.records = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "duration": 0.0})
        self.failures = []
        self.start_time = None
        self.total_duration = 0.0

    def _get_subsystem(self, nodeid: str) -> str:
        if "web-app/backend/tests" in nodeid:
            return "Backend Ingestion & API"
        elif "test_models_suite.py" in nodeid:
            return "Production Models Suite (Models 1-5)"
        elif "tests/test_tuning" in nodeid or "tests/test_checkpoint" in nodeid:
            return "Tuning Engine & Model Registry"
        elif "training/classification/tests" in nodeid:
            return "Disease Classification Training"
        elif "training/segmentation/tests" in nodeid:
            return "Retinal Segmentation Training"
        elif "tests/" in nodeid:
            return "Root Integration Tests"
        return "General"

    def pytest_sessionstart(self, session):
        self.start_time = time.time()

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.outcome == "skipped"):
            subsystem = self._get_subsystem(report.nodeid)
            stats = self.records[subsystem]
            stats["total"] += 1
            stats["duration"] += report.duration

            if report.passed:
                stats["passed"] += 1
            elif report.failed:
                stats["failed"] += 1
                self.failures.append((report.nodeid, report.longreprtext or str(report.longrepr)))
            elif report.skipped:
                stats["skipped"] += 1

    def pytest_sessionfinish(self, session, exitstatus):
        self.total_duration = time.time() - self.start_time


def print_telemetry(collector: TelemetryCollector) -> int:
    """Renders a clean, structured telemetry summary table in the terminal."""
    total_tests = sum(s["total"] for s in collector.records.values())
    total_passed = sum(s["passed"] for s in collector.records.values())
    total_failed = sum(s["failed"] for s in collector.records.values())
    total_skipped = sum(s["skipped"] for s in collector.records.values())
    pass_rate = (total_passed / total_tests * 100.0) if total_tests > 0 else 0.0

    line_width = 82
    print("\n" + "=" * line_width)
    print("                      OCT-ANALYSER TEST SUITE TELEMETRY")
    print("=" * line_width)
    print(f"  Environment: Python {sys.version.split()[0]} | PyTorch {TORCH_VER} | Device: {DEVICE_NAME}")
    print(f"  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Root: {WORKSPACE_ROOT.name}")
    print("-" * line_width)
    print(f"{'Subsystem':<40} {'Tests':>6} {'Passed':>7} {'Failed':>7} {'Skipped':>8} {'Duration':>9}")
    print("-" * line_width)

    # Display in logical order
    order = [
        "Backend Ingestion & API",
        "Production Models Suite (Models 1-5)",
        "Tuning Engine & Model Registry",
        "Disease Classification Training",
        "Retinal Segmentation Training",
        "Root Integration Tests",
        "General"
    ]

    for category in order:
        if category in collector.records:
            stats = collector.records[category]
            dur_str = f"{stats['duration']:.2f}s"
            print(f"{category:<40} {stats['total']:>6} {stats['passed']:>7} {stats['failed']:>7} {stats['skipped']:>8} {dur_str:>9}")

    print("-" * line_width)
    total_dur_str = f"{collector.total_duration:.2f}s"
    print(f"{'TOTAL':<40} {total_tests:>6} {total_passed:>7} {total_failed:>7} {total_skipped:>8} {total_dur_str:>9}")
    print("=" * line_width)

    if total_failed > 0:
        print(f"OVERALL STATUS: FAILED ({pass_rate:.1f}% Pass Rate, {total_failed} Failures)")
        print("=" * line_width)
        print("\n--- FAILURE DETAILS ---")
        for nodeid, err in collector.failures:
            print(f"\n[FAIL] {nodeid}")
            # Show first 8 lines of error trace
            err_lines = err.strip().split("\n")
            print("  " + "\n  ".join(err_lines[:8]))
            if len(err_lines) > 8:
                print(f"  ... ({len(err_lines) - 8} lines omitted)")
        print("\n" + "=" * line_width)
        return 1
    else:
        print(f"OVERALL STATUS: PASSED ({pass_rate:.1f}% Success Rate - All {total_tests} Tests Passing)")
        print("=" * line_width + "\n")
        return 0


def main():
    parser = argparse.ArgumentParser(description="OCT-Analyser Unified Test Runner & Telemetry Aggregator")
    parser.add_argument("-k", "--keyword", type=str, help="Filter tests by keyword expression")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full pytest progress output")
    parser.add_argument("--subsystem", type=str, choices=["backend", "models", "tuning", "classification", "segmentation"],
                        help="Run tests only for a specific subsystem")
    args, unknown = parser.parse_known_args()

    pytest_args = [
        "-c", str(WORKSPACE_ROOT / "pytest.ini"),
        "--disable-warnings",
        "-q" if not args.verbose else "-v",
    ]

    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    # Select target paths
    if args.subsystem == "backend":
        targets = [str(WORKSPACE_ROOT / "web-app" / "backend" / "tests")]
    elif args.subsystem == "models":
        targets = [str(WORKSPACE_ROOT / "tests" / "test_models_suite.py")]
    elif args.subsystem == "tuning":
        targets = [str(WORKSPACE_ROOT / "tests" / "test_tuning_dashboard.py"), str(WORKSPACE_ROOT / "tests" / "test_checkpoint_versioning.py")]
    elif args.subsystem == "classification":
        targets = [str(WORKSPACE_ROOT / "training" / "classification" / "tests")]
    elif args.subsystem == "segmentation":
        targets = [str(WORKSPACE_ROOT / "training" / "segmentation" / "tests")]
    else:
        targets = [
            str(WORKSPACE_ROOT / "tests"),
            str(WORKSPACE_ROOT / "web-app" / "backend" / "tests"),
            str(WORKSPACE_ROOT / "training" / "classification" / "tests"),
            str(WORKSPACE_ROOT / "training" / "segmentation" / "tests")
        ]

    pytest_args.extend(targets)
    pytest_args.extend(unknown)

    collector = TelemetryCollector()

    print("Running OCT-Analyser test suite (quiet mode)...", flush=True)
    exit_code = pytest.main(pytest_args, plugins=[collector])
    
    summary_code = print_telemetry(collector)
    sys.exit(summary_code if summary_code != 0 else exit_code)


if __name__ == "__main__":
    main()
