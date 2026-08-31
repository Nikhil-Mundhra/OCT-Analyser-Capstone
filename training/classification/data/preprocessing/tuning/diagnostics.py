"""
data/preprocessing/tuning/diagnostics.py

Filesystem read/write verification, health self-checks, and in-process endpoint testing.
"""

import io
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional
import urllib.error
import urllib.parse
import urllib.request

from data.preprocessing.params import initialize_default_params_file, load_all_params
from data.preprocessing.tuning.processor import (
    OUTPUT_DIR,
    SOURCE_DIR,
    find_folder_path,
    get_available_subfolders,
    get_output_dir,
    get_source_dir,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = PACKAGE_DIR / "dashboard"


def check_filesystem_access(
    source_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> dict[str, Any]:
    """
    Performs comprehensive filesystem permissions and dataset availability diagnostics.
    Tests directory existence, read permissions, write permissions, and sample counts.
    """
    active_source = Path(source_dir or get_source_dir())
    active_output = Path(output_dir or get_output_dir())

    source_exists = False
    source_readable = False
    source_permission_error = None
    subfolders: list[str] = []
    total_images_sampled = 0

    try:
        source_exists = active_source.exists() and active_source.is_dir()
        if source_exists:
            try:
                with os.scandir(active_source) as it:
                    _ = next(it, None)
                source_readable = True
                subfolders = get_available_subfolders(active_source)
                for sf in subfolders[:5]:
                    folder_p = find_folder_path(sf, active_source)
                    if folder_p:
                        imgs = list(folder_p.glob("*.jp*g")) + list(folder_p.glob("*.png"))
                        total_images_sampled += len(imgs)
            except (PermissionError, OSError) as pe:
                source_readable = False
                source_permission_error = f"Permission Denied: {pe}"
    except (PermissionError, OSError) as pe:
        source_permission_error = f"Permission Denied: {pe}"

    output_exists = False
    output_writable = False
    output_permission_error = None

    try:
        active_output.mkdir(parents=True, exist_ok=True)
        output_exists = active_output.exists()
        probe_file = active_output / ".write_probe.tmp"
        try:
            probe_file.write_text("probe_ok", encoding="utf-8")
            if probe_file.exists():
                probe_file.unlink()
            output_writable = True
        except (PermissionError, OSError) as we:
            output_writable = False
            output_permission_error = f"Write Denied: {we}"
    except (PermissionError, OSError) as we:
        output_permission_error = f"Create Directory Denied: {we}"

    config_writable = False
    config_permission_error = None
    param_count = 0
    try:
        initialize_default_params_file()
        params = load_all_params()
        config_writable = isinstance(params, dict)
        param_count = len(params) if isinstance(params, dict) else 0
    except (PermissionError, OSError) as ce:
        config_permission_error = f"Config Error: {ce}"

    if source_permission_error or (source_exists and not source_readable):
        fs_status = "PERMISSION_DENIED"
        remediation = "Server process lacks OS or sandbox permissions to read SOURCE_DIR. Run with filesystem access."
    elif not source_exists:
        fs_status = "SOURCE_NOT_FOUND"
        remediation = f"Source directory '{active_source}' does not exist. Specify path via --source-dir or SOURCE_DIR env var."
    elif len(subfolders) == 0:
        fs_status = "NO_CLASSES_FOUND"
        remediation = f"Directory '{active_source}' exists but 0 classes with .jpg/.png images were detected."
    elif not output_writable:
        fs_status = "OUTPUT_NOT_WRITABLE"
        remediation = f"Output directory '{active_output}' is not writable: {output_permission_error}"
    else:
        fs_status = "HEALTHY"
        remediation = None

    return {
        "status": fs_status,
        "source_dir": str(active_source),
        "source_exists": source_exists,
        "source_readable": source_readable,
        "source_permission_error": source_permission_error,
        "folder_count": len(subfolders),
        "subfolders": subfolders,
        "images_sampled": total_images_sampled,
        "output_dir": str(active_output),
        "output_exists": output_exists,
        "output_writable": output_writable,
        "output_permission_error": output_permission_error,
        "config_writable": config_writable,
        "config_permission_error": config_permission_error,
        "param_count": param_count,
        "remediation": remediation
    }


def perform_preflight_checks(
    source_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    dashboard_dir: Optional[Path] = None
) -> dict[str, Any]:
    """
    Performs pre-flight environment, directory, and asset validation checks.
    """
    d_dir = dashboard_dir or DASHBOARD_DIR
    html_file = d_dir / "index.html"
    css_file = d_dir / "css" / "dashboard.css"
    js_file = d_dir / "js" / "app.js"

    dashboard_assets_ok = html_file.exists() and css_file.exists() and js_file.exists()
    fs = check_filesystem_access(source_dir=source_dir, output_dir=output_dir)

    return {
        "dashboard_assets_ok": dashboard_assets_ok,
        "html_exists": html_file.exists(),
        "css_exists": css_file.exists(),
        "js_exists": js_file.exists(),
        "filesystem": fs,
        "source_dir": fs["source_dir"],
        "source_exists": fs["source_exists"],
        "source_readable": fs["source_readable"],
        "output_dir": fs["output_dir"],
        "output_exists": fs["output_exists"],
        "output_writable": fs["output_writable"],
        "folder_count": fs["folder_count"],
        "subfolders": fs["subfolders"],
        "params_loaded": fs["config_writable"],
        "param_count": fs["param_count"]
    }


class InProcessSocket:
    """Mock socket for zero-dependency in-process HTTP dispatch and testing."""
    def __init__(self, request_bytes: bytes):
        self.rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "r" in mode:
            return self.rfile
        return self.wfile

    def sendall(self, data: bytes):
        self.wfile.write(data)


def dispatch_in_process_request(path: str, method: str = "GET", body: bytes = b"") -> tuple[int, dict, bytes]:
    """
    Executes a complete HTTP request/response transaction in-process via FineTuningRequestHandler.
    """
    from data.preprocessing.tuning.server import FineTuningRequestHandler

    content_len_hdr = f"Content-Length: {len(body)}\r\n" if body or method == "POST" else ""
    req_bytes = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n{content_len_hdr}\r\n".encode("utf-8") + body
    mock_sock = InProcessSocket(req_bytes)

    try:
        FineTuningRequestHandler(mock_sock, ("127.0.0.1", 8000), None)
    except Exception as e:
        sys.stderr.write(f"[DEBUG] Exception in in-process request: {e}\n")

    raw_response = mock_sock.wfile.getvalue()
    if not raw_response:
        return 500, {}, b""

    parts = raw_response.split(b"\r\n\r\n", 1)
    header_lines = parts[0].decode("utf-8", errors="ignore").split("\r\n")
    resp_body = parts[1] if len(parts) > 1 else b""

    status_code = 0
    if header_lines and len(header_lines[0].split()) >= 2:
        try:
            status_code = int(header_lines[0].split()[1])
        except ValueError:
            status_code = 0

    return status_code, {}, resp_body


def verify_server_endpoints(base_url: Optional[str] = None, timeout: float = 3.0) -> dict[str, Any]:
    """
    Directly tests HTTP server endpoints.
    Uses in-process handler dispatch if base_url is None or if network sockets are restricted.
    """
    endpoints = [
        {"name": "Health Check", "path": "/api/health", "method": "GET", "expected_code": 200},
        {"name": "Folder List & Params", "path": "/api/folders", "method": "GET", "expected_code": 200},
        {"name": "Dashboard Index", "path": "/", "method": "GET", "expected_code": 200},
        {"name": "Dashboard CSS", "path": "/css/dashboard.css", "method": "GET", "expected_code": 200},
        {"name": "Dashboard JS", "path": "/js/app.js", "method": "GET", "expected_code": 200},
    ]

    results = []
    all_passed = True

    for ep in endpoints:
        start_t = time.time()
        status_code = 0
        resp_len = 0
        err_msg = None

        if base_url:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            url = f"{base_url.rstrip('/')}{ep['path']}"
            try:
                req = urllib.request.Request(url, method=ep["method"])
                with opener.open(req, timeout=timeout) as resp:
                    status_code = resp.getcode()
                    content = resp.read()
                    resp_len = len(content)
            except urllib.error.HTTPError as e:
                status_code = e.code
                err_msg = str(e)
            except Exception:
                status_code, _, content = dispatch_in_process_request(ep["path"], ep["method"])
                resp_len = len(content)
        else:
            status_code, _, content = dispatch_in_process_request(ep["path"], ep["method"])
            resp_len = len(content)

        elapsed_ms = (time.time() - start_t) * 1000
        passed = (status_code == ep["expected_code"])
        if not passed:
            all_passed = False

        results.append({
            "endpoint": ep["name"],
            "path": ep["path"],
            "status_code": status_code,
            "passed": passed,
            "latency_ms": round(elapsed_ms, 2),
            "bytes": resp_len,
            "error": err_msg
        })

    return {
        "all_passed": all_passed,
        "base_url": base_url or "in-process",
        "total": len(endpoints),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "results": results
    }


def run_standalone_self_tests(port: int = 8000) -> int:
    """
    Executes a complete self-contained test suite against the server, handler, and filesystem permissions.
    Outputs clear pass/fail status for each endpoint, filesystem access, and preflight check.
    """
    print("=" * 70)
    print("  Running OCT Tuning Server Built-In Diagnostic & Endpoint Checks")
    print("=" * 70)

    pf = perform_preflight_checks()
    fs = pf["filesystem"]
    print(f"  [1/3] Filesystem & Dataset Access Checks:")

    src_readable_tag = "PASS" if fs["source_readable"] and fs["folder_count"] > 0 else ("WARN" if fs["source_exists"] else "FAIL")
    print(f"    Source Path       : {'PASS' if fs['source_exists'] else 'FAIL'} ({fs['source_dir']})")
    print(f"    Source Readable   : {src_readable_tag} ({fs['folder_count']} classes detected, {fs['images_sampled']}+ samples)")
    if fs["source_permission_error"]:
        print(f"    Permission Error  : {fs['source_permission_error']}")
    print(f"    Output Writable   : {'PASS' if fs['output_writable'] else 'FAIL'} ({fs['output_dir']})")
    if fs["output_permission_error"]:
        print(f"    Write Error       : {fs['output_permission_error']}")
    print(f"    Config Access     : {'PASS' if fs['config_writable'] else 'FAIL'} ({fs['param_count']} classes configured)")
    print(f"    Filesystem Status : {fs['status']}")
    if fs["remediation"]:
        print(f"    Remediation Note  : {fs['remediation']}")

    print(f"\n  [2/3] Dashboard UI Asset Checks:")
    print(f"    HTML Index        : {'PASS' if pf['html_exists'] else 'FAIL'}")
    print(f"    CSS Stylesheets   : {'PASS' if pf['css_exists'] else 'FAIL'}")
    print(f"    JS Application    : {'PASS' if pf['js_exists'] else 'FAIL'}")

    print(f"\n  [3/3] Endpoint Health Checks:")
    check_res = verify_server_endpoints()

    for r in check_res["results"]:
        status_tag = "PASS" if r["passed"] else "FAIL"
        print(f"    {status_tag:4} | {r['path']:<25} | HTTP {r['status_code']:<3} | {r['latency_ms']}ms")

    print("-" * 70)
    if check_res["all_passed"] and pf["dashboard_assets_ok"]:
        print(f"  RESULT: ALL CHECKS PASSED ({check_res['passed']}/{check_res['total']} endpoints verified)")
        print("=" * 70)
        return 0
    else:
        print(f"  RESULT: FAILED ({check_res['failed']} endpoint failures)")
        print("=" * 70)
        return 1
