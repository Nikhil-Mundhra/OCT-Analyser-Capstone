"""
data/preprocessing/tuning/server.py

Interactive Folder-Specific Preprocessing Parameter Tuning Local HTTP Server.
Modular orchestration layer serving dashboard assets, JSON APIs, and diagnostic health checks.
"""

import argparse
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Optional
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Add root project dirs to sys.path if not already present
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "image-classification-model-training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "image-classification-model-training"))

from data.preprocessing.params import (
    DEFAULT_PARAMS,
    get_folder_params,
    initialize_default_params_file,
    load_all_params,
    save_all_params,
)

# Re-export boundary algorithms for backward compatibility
from data.preprocessing.tuning.boundaries import (
    _estimate_adaptive_thresholds,
    _extract_raw_boundary_contours,
    _interpolate_and_filter_boundaries,
    compute_sfcm_choroid_boundary,
    detect_rpe_band,
    detect_choroidal_caverns,
    generate_tissue_mask_custom,
    get_sfcm_cache_key,
    letterbox_pad_and_resize,
    project_and_downsample_vectors,
    suppress_boundary_spikes,
)

# Re-export image & dataset processing logic and caches
from data.preprocessing.tuning.processor import (
    FOLDER_SAMPLES_CACHE,
    MASKED_DATASET_DIR,
    OUTPUT_DIR,
    SFCM_CACHE,
    SOURCE_DIR,
    curate_folder_batch,
    find_folder_path,
    find_image_path,
    get_available_subfolders,
    get_curated_manifest,
    get_masked_dataset_dir,
    get_output_dir,
    get_source_dir,
    process_and_save_image,
    remove_curated_mask_sample,
    reprocess_folder_sample,
    reprocess_single_image,
    save_curated_mask_sample,
)

# Re-export diagnostics and health checks
from data.preprocessing.tuning.diagnostics import (
    InProcessSocket,
    check_filesystem_access,
    dispatch_in_process_request,
    perform_preflight_checks,
    run_standalone_self_tests,
    verify_server_endpoints,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = PACKAGE_DIR / "dashboard"


class FineTuningRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving dashboard UI, static assets, and tuning API endpoints with robust error boundaries."""

    def log_message(self, format, *args):
        sys.stderr.write(f"[HTTP] {self.address_string()} - {format % args}\n")

    def _send_json_response(self, data: dict, status: int = 200):
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to send JSON response: {e}\n")

    def _send_file_response(self, file_path: Path, content_type: str):
        try:
            if not file_path.exists() or not file_path.is_file():
                self._send_json_response({"status": "error", "message": f"File not found: {file_path.name}"}, status=404)
                return
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self._send_json_response({"status": "error", "message": f"Failed to read file: {str(e)}"}, status=500)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html", "/tuning_dashboard.html"):
                html_p = DASHBOARD_DIR / "index.html"
                if html_p.exists():
                    self._send_file_response(html_p, "text/html; charset=utf-8")
                else:
                    self._send_json_response({"status": "error", "message": "Dashboard index.html not found"}, status=404)
                return

            elif path.startswith("/css/"):
                rel = path[len("/css/"):].lstrip("/")
                css_file = (DASHBOARD_DIR / "css" / rel).resolve()
                if str(css_file).startswith(str((DASHBOARD_DIR / "css").resolve())) and css_file.exists() and css_file.is_file():
                    self._send_file_response(css_file, "text/css; charset=utf-8")
                else:
                    self._send_json_response({"status": "error", "message": f"CSS file not found: {rel}"}, status=404)
                return

            elif path.startswith("/js/"):
                rel = path[len("/js/"):].lstrip("/")
                js_file = (DASHBOARD_DIR / "js" / rel).resolve()
                if str(js_file).startswith(str((DASHBOARD_DIR / "js").resolve())) and js_file.exists() and js_file.is_file():
                    self._send_file_response(js_file, "application/javascript; charset=utf-8")
                else:
                    self._send_json_response({"status": "error", "message": f"JS file not found: {rel}"}, status=404)
                return

            elif path == "/api/health":
                active_source = get_source_dir()
                active_output = get_output_dir()
                fs = check_filesystem_access(active_source, active_output)
                self._send_json_response({
                    "status": "healthy" if fs["status"] == "HEALTHY" else "degraded",
                    "filesystem": fs,
                    "version": "1.0.0",
                    "source_dir": fs["source_dir"],
                    "source_exists": fs["source_exists"],
                    "source_readable": fs["source_readable"],
                    "output_dir": fs["output_dir"],
                    "output_exists": fs["output_exists"],
                    "output_writable": fs["output_writable"],
                    "folder_count": fs["folder_count"],
                    "dashboard_ready": (DASHBOARD_DIR / "index.html").exists()
                })
                return

            elif path == "/api/folders":
                active_source = get_source_dir()
                subfolders = get_available_subfolders(active_source)
                saved_params = load_all_params()

                res = {
                    "folders": subfolders,
                    "saved_params": saved_params,
                    "default_params": DEFAULT_PARAMS
                }
                self._send_json_response(res)
                return

            elif path == "/api/curated_manifest":
                manifest = get_curated_manifest()
                self._send_json_response(manifest)
                return

            elif path.startswith("/masked/"):
                active_masked = get_masked_dataset_dir()
                rel = path[len("/masked/"):].lstrip("/")
                target_file = (active_masked / rel).resolve()
                if str(target_file).startswith(str(active_masked.resolve())) and target_file.exists() and target_file.is_file():
                    content_type = "image/png" if rel.lower().endswith(".png") else "image/jpeg"
                    self._send_file_response(target_file, content_type)
                else:
                    self._send_json_response({"status": "error", "message": "Curated dataset file not found"}, status=404)
                return

            elif path.startswith("/preprocessed/"):
                active_output = get_output_dir()
                rel = path[len("/preprocessed/"):].lstrip("/")
                target_file = (active_output / rel).resolve()
                if str(target_file).startswith(str(active_output.resolve())) and target_file.exists() and target_file.is_file():
                    self._send_file_response(target_file, "image/jpeg")
                else:
                    self._send_json_response({"status": "error", "message": "Preprocessed image not found"}, status=404)
                return

            if path.startswith("/api/"):
                self._send_json_response({"status": "error", "message": f"Unknown API endpoint: {path}"}, status=404)
                return

            self._send_json_response({"status": "error", "message": f"Path not found: {path}"}, status=404)
        except Exception as exc:
            self._send_json_response({"status": "error", "message": f"Internal Server Error: {str(exc)}"}, status=500)

    def do_POST(self):
        try:
            content_length_hdr = self.headers.get("Content-Length")
            if not content_length_hdr:
                self._send_json_response({"status": "error", "message": "Missing Content-Length header"}, status=400)
                return

            try:
                length = int(content_length_hdr)
            except ValueError:
                self._send_json_response({"status": "error", "message": "Invalid Content-Length"}, status=400)
                return

            if length <= 0 or length > 10 * 1024 * 1024:
                self._send_json_response({"status": "error", "message": "Invalid payload size"}, status=400)
                return

            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                self._send_json_response({"status": "error", "message": "Invalid JSON"}, status=400)
                return

            if self.path == "/api/reprocess":
                folder_name = data.get("folder")
                if not folder_name:
                    self._send_json_response({"status": "error", "message": "Missing 'folder' field"}, status=400)
                    return
                params = data.get("params", DEFAULT_PARAMS)
                random_sample = data.get("random_sample", False)

                all_params = load_all_params()
                all_params[folder_name] = params
                save_all_params(all_params)

                samples = reprocess_folder_sample(folder_name, params, random_sample=random_sample)

                res = {
                    "status": "success",
                    "folder": folder_name,
                    "samples": samples
                }
                self._send_json_response(res)
                return

            elif self.path == "/api/reprocess_single":
                folder_name = data.get("folder")
                filename = data.get("filename")
                if not folder_name or not filename:
                    self._send_json_response({"status": "error", "message": "Missing 'folder' or 'filename' field"}, status=400)
                    return
                params = data.get("params", DEFAULT_PARAMS)

                sample = reprocess_single_image(folder_name, filename, params)
                if sample:
                    self._send_json_response({"status": "success", "sample": sample})
                else:
                    self._send_json_response({"status": "error", "message": "Image not found"}, status=404)
                return

            elif self.path == "/api/curate_sample":
                folder_name = data.get("folder")
                filename = data.get("filename")
                if not folder_name or not filename:
                    self._send_json_response({"status": "error", "message": "Missing 'folder' or 'filename' field"}, status=400)
                    return
                params = data.get("params", DEFAULT_PARAMS)
                try:
                    res = save_curated_mask_sample(folder_name, filename, params)
                    self._send_json_response(res)
                except Exception as err:
                    self._send_json_response({"status": "error", "message": str(err)}, status=500)
                return

            elif self.path == "/api/uncurate_sample":
                folder_name = data.get("folder")
                filename = data.get("filename")
                if not folder_name or not filename:
                    self._send_json_response({"status": "error", "message": "Missing 'folder' or 'filename' field"}, status=400)
                    return
                try:
                    res = remove_curated_mask_sample(folder_name, filename)
                    self._send_json_response(res)
                except Exception as err:
                    self._send_json_response({"status": "error", "message": str(err)}, status=500)
                return

            elif self.path == "/api/curate_batch":
                folder_name = data.get("folder")
                filenames = data.get("filenames", [])
                if not folder_name or not filenames:
                    self._send_json_response({"status": "error", "message": "Missing 'folder' or 'filenames' field"}, status=400)
                    return
                params = data.get("params", DEFAULT_PARAMS)
                try:
                    res = curate_folder_batch(folder_name, filenames, params)
                    self._send_json_response(res)
                except Exception as err:
                    self._send_json_response({"status": "error", "message": str(err)}, status=500)
                return

            self._send_json_response({"status": "error", "message": f"Unknown API endpoint: {self.path}"}, status=404)
        except Exception as exc:
            self._send_json_response({"status": "error", "message": f"Internal Server Error: {str(exc)}"}, status=500)


class ReusableHTTPServer(HTTPServer):
    """HTTPServer with socket reuse enabled and daemon worker handling."""
    allow_reuse_address = True
    daemon_threads = True


def run_server(port: int = 8000, host: str = "", auto_port: bool = True, run_self_check: bool = True) -> int:
    """
    Runs the Preprocessing Tuning Server with preflight validation,
    port conflict resolution, automated endpoint checks, and clean shutdown handling.
    """
    preflight = perform_preflight_checks()
    fs = preflight["filesystem"]

    current_port = port
    max_attempts = 10 if auto_port else 1
    httpd = None

    for attempt in range(max_attempts):
        try:
            server_address = (host, current_port)
            httpd = ReusableHTTPServer(server_address, FineTuningRequestHandler)
            break
        except OSError as e:
            if auto_port and attempt < max_attempts - 1:
                print(f"[WARN] Port {current_port} is in use, attempting port {current_port + 1}...")
                current_port += 1
            else:
                print(f"[ERROR] Could not bind to port {current_port}: {e}")
                print("[TIP] You can specify a different port with: python3 scripts/tuning_server.py --port <port>")
                return 1

    display_host = "localhost" if host in ("", "0.0.0.0") else host
    server_url = f"http://{display_host}:{current_port}"

    print("=" * 70)
    print("  OCT Preprocessing Folder Fine-Tuning Server")
    print("=" * 70)
    print(f"  URL:         {server_url}")
    print(f"  Dashboard:   {'READY' if preflight['dashboard_assets_ok'] else 'ASSETS MISSING'}")
    print(f"  Source Dir:  {fs['source_dir']} ({fs['folder_count']} classes, {'READABLE' if fs['source_readable'] else 'ACCESS BLOCKED'})")
    print(f"  Output Dir:  {fs['output_dir']} ({'WRITABLE' if fs['output_writable'] else 'WRITE BLOCKED'})")
    if fs["status"] != "HEALTHY" and fs["remediation"]:
        print("-" * 70)
        print(f"  [FS NOTICE: {fs['status']}]")
        print(f"  {fs['remediation']}")
    print("-" * 70)
    print("  API Endpoints:")
    print("    GET  /api/health            -> System health, permissions & preflight status")
    print("    GET  /api/folders           -> Folder parameters and class list")
    print("    POST /api/reprocess         -> Reprocess folder sample batch")
    print("    POST /api/reprocess_single  -> Reprocess individual image")
    print("-" * 70)

    if run_self_check:
        def _async_self_check():
            time.sleep(0.3)
            check_res = verify_server_endpoints(f"http://127.0.0.1:{current_port}")
            if check_res["all_passed"]:
                print(f"[HEALTH-CHECK] PASS - All {check_res['passed']}/{check_res['total']} internal endpoint checks succeeded.")
            else:
                failed_items = [r["path"] for r in check_res["results"] if not r["passed"]]
                print(f"[HEALTH-CHECK] WARN - Some endpoint checks failed: {failed_items}")

        threading.Thread(target=_async_self_check, daemon=True).start()

    print(f"  Server listening on {server_url} (Press Ctrl+C to stop)")
    print("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down tuning server gracefully...")
    finally:
        if httpd:
            httpd.server_close()
        print("[INFO] Server stopped.")
    return 0


def print_cli_metrics_table(res: dict):
    """Prints a formatted diagnostic table of anatomical layer metrics and saved file paths."""
    m = res["metrics"]
    files = res["saved_files"]
    print("\n" + "=" * 70)
    print(f"  OCT Scan Preprocessing Analysis: {res['filename']}")
    print("=" * 70)
    print(f"  Folder:      {res['folder']}")
    print(f"  Dimensions:  {res['dimensions']['width']}x{res['dimensions']['height']} px")
    print(f"  Source Path: {res['filepath']}")
    print("-" * 70)
    print("  Anatomical Boundary & Thickness Metrics:")
    print(f"    ILM Surface (y_top):    mean = {m['ilm_y']['mean']:>5.1f} px  [min: {m['ilm_y']['min']:>5.1f}, max: {m['ilm_y']['max']:>5.1f}]")
    print(f"    RPE Band (y_rpe):       mean = {m['rpe_y']['mean']:>5.1f} px  [min: {m['rpe_y']['min']:>5.1f}, max: {m['rpe_y']['max']:>5.1f}]")
    print(f"    Choroid Floor (y_sfcm): mean = {m['choroid_y']['mean']:>5.1f} px  [min: {m['choroid_y']['min']:>5.1f}, max: {m['choroid_y']['max']:>5.1f}]")
    print(f"    Retinal Thickness:      mean = {m['retinal_thickness_px']['mean']:>5.1f} px  [min: {m['retinal_thickness_px']['min']:>5.1f}, max: {m['retinal_thickness_px']['max']:>5.1f}]")
    print(f"    Choroid Thickness:      mean = {m['choroid_thickness_px']['mean']:>5.1f} px  [min: {m['choroid_thickness_px']['min']:>5.1f}, max: {m['choroid_thickness_px']['max']:>5.1f}]")
    print(f"    Choroidal Holes:        {m['holes_count']} detected")
    print(f"    Choroidal Caverns:      {m['caverns_count']} detected")
    print("-" * 70)
    print("  Output Files Generated:")
    print(f"    Processed (384x384):  {files['processed']}")
    print(f"    Raw Scan (384x384):   {files['raw']}")
    if files.get("overlay"):
        print(f"    Diagnostic Overlay:   {files['overlay']}")
    if files.get("mask"):
        print(f"    Tissue Mask:          {files['mask']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="OCT Preprocessing Tuning Server & Direct Image Processing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive web server:
  python3 -m data.preprocessing.tuning
  python3 scripts/tuning_server.py --port 8000

  # Process a single image directly and output overlay:
  python3 -m data.preprocessing.tuning --image "/path/to/Subject_01_slice_030.png"

  # Process an image by filename within a dataset class:
  python3 -m data.preprocessing.tuning --folder Chiu_BOE_2014-DME --image Subject_01_slice_030.png

  # Process with custom JSON parameters and save mask:
  python3 -m data.preprocessing.tuning -i "/path/to/scan.png" -c '{"auto_mode": true, "margin_top": 20}' --save-mask -o ./output

  # Process a 4-sample batch from a dataset folder:
  python3 -m data.preprocessing.tuning --folder Chiu_BOE_2014-DME --sample 4 -o ./output
"""
    )
    # Server configuration options
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="", help="Host interface to bind to (default: all)")
    parser.add_argument("--check", "--self-test", dest="self_test", action="store_true", help="Run diagnostic health checks and exit")
    parser.add_argument("--no-check", dest="no_check", action="store_true", help="Disable automatic startup self-check")
    parser.add_argument("--no-auto-port", dest="no_auto_port", action="store_true", help="Disable automatic port bumping on conflict")
    parser.add_argument("--source-dir", type=str, default=None, help="Override path to Classified dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Override path to preprocessed output directory")
    parser.add_argument("--masked-dir", type=str, default=None, help="Override path to curated Classified-masked output directory")

    # Direct Processing CLI options
    parser.add_argument("-i", "--image", type=str, default=None, help="Path to a single image file or filename to process directly")
    parser.add_argument("-f", "--folder", type=str, default=None, help="Dataset subfolder name (e.g. Chiu_BOE_2014-DME)")
    parser.add_argument("-c", "--config", type=str, default=None, help="JSON configuration string or path to JSON config file")
    parser.add_argument("-o", "--out", "--out-dir", dest="out_dir", type=str, default=None, help="Output directory to save processed files")
    parser.add_argument("--save-overlay", action="store_true", default=True, help="Save diagnostic RGB boundary overlay image (default: True)")
    parser.add_argument("--no-overlay", dest="save_overlay", action="store_false", help="Disable saving overlay image")
    parser.add_argument("--save-mask", action="store_true", default=False, help="Save binary tissue mask image")
    parser.add_argument("--sample", type=int, default=None, help="Process N sample images from specified --folder")
    parser.add_argument("--json", dest="output_json", action="store_true", help="Output result as JSON to stdout")

    args = parser.parse_args()

    from data.preprocessing.tuning import processor as proc
    if args.source_dir:
        proc.SOURCE_DIR = Path(args.source_dir)
    if args.output_dir:
        proc.OUTPUT_DIR = Path(args.output_dir)
    if args.masked_dir:
        proc.MASKED_DATASET_DIR = Path(args.masked_dir)

    if args.self_test:
        sys.exit(run_standalone_self_tests(port=args.port))

    # Parse JSON config override if provided
    params_override = None
    if args.config:
        cfg_str = args.config.strip()
        if Path(cfg_str).exists() and Path(cfg_str).is_file():
            with open(cfg_str, "r", encoding="utf-8") as f:
                params_override = json.load(f)
        else:
            try:
                params_override = json.loads(cfg_str)
            except json.JSONDecodeError as err:
                sys.stderr.write(f"[ERROR] Failed to parse --config JSON: {err}\n")
                sys.exit(1)

    # 1. Direct Single Image Processing
    if args.image:
        try:
            res = proc.process_single_image_cli(
                image_path_or_filename=args.image,
                folder_name=args.folder,
                params=params_override,
                out_dir=args.out_dir,
                save_overlay=args.save_overlay,
                save_mask=args.save_mask,
            )
            if args.output_json:
                print(json.dumps(res, indent=2))
            else:
                print_cli_metrics_table(res)
            sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to process image: {e}\n")
            sys.exit(1)

    # 2. Direct Folder Batch Processing
    if args.folder and args.sample is not None:
        try:
            results = proc.process_folder_cli(
                folder_name=args.folder,
                params=params_override,
                sample_count=args.sample,
                out_dir=args.out_dir,
                save_overlay=args.save_overlay,
                save_mask=args.save_mask,
            )
            if args.output_json:
                print(json.dumps(results, indent=2))
            else:
                print(f"\nProcessed {len(results)} samples from folder '{args.folder}':")
                for res in results:
                    print_cli_metrics_table(res)
            sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to process folder: {e}\n")
            sys.exit(1)

    # 3. Interactive Web Server
    sys.exit(run_server(
        port=args.port,
        host=args.host,
        auto_port=not args.no_auto_port,
        run_self_check=not args.no_check
    ))


if __name__ == "__main__":
    main()
