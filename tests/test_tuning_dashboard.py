import io
import json
import re
from pathlib import Path
from html.parser import HTMLParser
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

class ElementIdExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.inputs = {}
        self.selects = {}
        self.title = ""
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        if 'id' in attr_dict:
            self.ids.add(attr_dict['id'])
        
        if tag == 'title':
            self.in_title = True
        elif tag == 'input' and 'id' in attr_dict:
            self.inputs[attr_dict['id']] = attr_dict
        elif tag == 'select' and 'id' in attr_dict:
            self.selects[attr_dict['id']] = attr_dict

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data

@pytest.fixture
def dashboard_parser():
    html_path = PROJECT_ROOT / 'training' / 'classification' / 'data' / 'preprocessing' / 'tuning' / 'dashboard' / 'index.html'
    if not html_path.exists():
        html_path = PROJECT_ROOT / 'scripts' / 'tuning_dashboard.html'
    assert html_path.exists(), f"Dashboard HTML file not found at {html_path}"
    content = html_path.read_text(encoding='utf-8')
    
    parser = ElementIdExtractor()
    parser.feed(content)
    return parser, content

def test_html_structure_and_title(dashboard_parser):
    parser, content = dashboard_parser
    assert "OCT Preprocessing Folder Fine-Tuning Dashboard" in parser.title

def test_sidebar_parameter_controls_exist(dashboard_parser):
    parser, _ = dashboard_parser
    
    expected_ids = [
        'folder-select',
        'btn-mode-slider',
        'btn-mode-json',
        'panel-sliders',
        'panel-json',
        'json-editor',
        'json-error',
        'btn-apply-json',
        'btn-save',
        'btn-reset',
        'gallery-grid',
        'btn-toggle-vectors',
        'btn-refresh-samples',
        'status-msg'
    ]
    for element_id in expected_ids:
        assert element_id in parser.ids, f"Required UI element #{element_id} missing in tuning_dashboard.html"

def test_slider_inputs_exist(dashboard_parser):
    parser, _ = dashboard_parser
    
    slider_ids = [
        'param-top_noise_mult',
        'param-bot_noise_mult',
        'param-shadow_bridge_top_pct',
        'param-shadow_bridge_bot_pct',
        'param-gaussian_sigma',
        'param-top_spike_suppress_px',
        'param-top_spike_window_px',
        'param-top_dip_suppress_px',
        'param-top_dip_window_px',
        'param-margin_top',
        'param-margin_bottom'
    ]
    for slider_id in slider_ids:
        assert slider_id in parser.inputs, f"Slider #{slider_id} missing in dashboard"
        assert parser.inputs[slider_id].get('type') == 'range'

def test_compass_controls_exist(dashboard_parser):
    parser, content = dashboard_parser
    
    assert 'param-compass_ui_enabled' in parser.inputs
    assert parser.inputs['param-compass_ui_enabled'].get('type') == 'checkbox'
    
    assert 'param-compass_location' in parser.selects
    assert 'value="auto"' in content
    assert 'value="bottom_left"' in content
    assert 'value="bottom_right"' in content

def test_js_script_methods_present():
    js_dir = PROJECT_ROOT / 'training' / 'classification' / 'data' / 'preprocessing' / 'tuning' / 'dashboard' / 'js'
    all_js_content = ""
    if js_dir.exists():
        for js_file in js_dir.glob("*.js"):
            all_js_content += js_file.read_text(encoding='utf-8') + "\n"
    else:
        legacy_html = PROJECT_ROOT / 'scripts' / 'tuning_dashboard.html'
        all_js_content = legacy_html.read_text(encoding='utf-8')
    
    required_js_symbols = [
        'init()',
        'loadFolderParams',
        'getParamsFromUI',
        'triggerReprocess',
        'onHandleDragStart',
        'onHandleDrag',
        'handleTopVectorDrag',
        'handleBottomVectorDrag',
        'buildSvgPathD',
        'buildSvgHandlesHtml',
        'showStatusMessage',
        'renderGallery',
        'initParamGroups'
    ]
    for symbol in required_js_symbols:
        assert symbol in all_js_content, f"JavaScript symbol {symbol} missing in modular JS scripts"


def test_boundary_spike_suppression():
    from tuning_server import suppress_boundary_spikes
    
    y = np.full(100, 50.0)
    # Add an upward spike (small y value)
    y[50] = 10.0
    
    y_fixed = suppress_boundary_spikes(y, spike_px=10.0, window=20, direction='up')
    assert y_fixed[50] > 10.0
    assert pytest.approx(y_fixed[50], abs=2.0) == 50.0

def test_tissue_mask_custom_generation():
    from tuning_server import generate_tissue_mask_custom, DEFAULT_PARAMS
    
    mock_gray = np.zeros((200, 200), dtype=np.uint8)
    mock_gray[40:160, 40:160] = 100
    
    params = DEFAULT_PARAMS.copy()
    mask, top_vec, bot_vec = generate_tissue_mask_custom(mock_gray, params)
    
    assert mask.shape == (200, 200)
    assert len(top_vec) > 0
    assert len(bot_vec) > 0

def test_compass_bbox_bottom_vector_capping():
    from tuning_server import generate_tissue_mask_custom, DEFAULT_PARAMS
    
    H, W = 200, 200
    mock_gray = np.zeros((H, W), dtype=np.uint8)
    mock_gray[30:170, 20:180] = 100
    
    # Place a compass box at bottom left (cols 0-40, rows 140-200)
    compass_bbox = (0, 140, 40, 200)
    params = DEFAULT_PARAMS.copy()
    params['margin_bottom'] = 30  # Intentional large margin
    
    mask, top_vec, bot_vec = generate_tissue_mask_custom(mock_gray, params, compass_bbox=compass_bbox)
    
    # In original coordinates before scaling (or via bot_vec), check that bottom boundary in cols 0-40 is <= by0 (140)
    # bot_vec returned is y_bottom_outer in original coordinates
    for x in range(0, 41):
        assert bot_vec[x] <= 140.0, f"Bottom vector at x={x} is {bot_vec[x]}, which is below compass box top (140)"

def test_reprocess_folder_sample_caching(tmp_path, monkeypatch):
    import tuning_server
    
    # Set SOURCE_DIR and OUTPUT_DIR to temp directory
    src_dir = tmp_path / "Classified"
    folder = src_dir / "TestFolder"
    folder.mkdir(parents=True)
    
    # Create dummy images
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(folder / "sample1.jpg"), img)
    cv2.imwrite(str(folder / "sample2.jpg"), img)
    
    out_dir = tmp_path / "Output"
    monkeypatch.setattr(tuning_server, "SOURCE_DIR", src_dir)
    monkeypatch.setattr(tuning_server, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(tuning_server, "FOLDER_SAMPLES_CACHE", {})
    
    res = tuning_server.reprocess_folder_sample("TestFolder", tuning_server.DEFAULT_PARAMS)
    assert len(res) == 2
    assert "TestFolder" in tuning_server.FOLDER_SAMPLES_CACHE

def test_sfcm_choroid_boundary_generation():
    from tuning_server import generate_tissue_mask_custom, compute_sfcm_choroid_boundary, DEFAULT_PARAMS
    
    H, W = 150, 150
    mock_gray = np.zeros((H, W), dtype=np.uint8)
    mock_gray[20:120, 20:130] = 110
    
    y_top_outer = np.full(W, 20.0)
    params = DEFAULT_PARAMS.copy()
    params['use_sfcm'] = True
    
    y_rpe, sfcm_bot = compute_sfcm_choroid_boundary(mock_gray, y_top_outer, params)
    assert len(sfcm_bot) == W
    assert np.all(sfcm_bot > y_top_outer)
    
    mask, top_vec, bot_vec = generate_tissue_mask_custom(mock_gray, params)
    assert mask.shape == (H, W)
    assert len(bot_vec) == W


def test_reprocess_single_image_success_and_not_found(tmp_path, monkeypatch):
    import tuning_server
    import cv2

    src_dir = tmp_path / "Classified"
    folder = src_dir / "TestFolder"
    folder.mkdir(parents=True)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(folder / "sample_single.jpg"), img)

    out_dir = tmp_path / "Output"
    monkeypatch.setattr(tuning_server, "SOURCE_DIR", src_dir)
    monkeypatch.setattr(tuning_server, "OUTPUT_DIR", out_dir)

    # Success case
    res = tuning_server.reprocess_single_image("TestFolder", "sample_single.jpg", tuning_server.DEFAULT_PARAMS)
    assert res is not None
    assert res["filename"] == "sample_single.jpg"
    assert "top_vector" in res
    assert "bottom_vector" in res
    assert len(res["top_vector"]) == 64
    assert len(res["bottom_vector"]) == 64
    assert (out_dir / "TestFolder" / "sample_single_proc.jpg").exists()

    # Missing file case
    missing_res = tuning_server.reprocess_single_image("TestFolder", "non_existent.jpg", tuning_server.DEFAULT_PARAMS)
    assert missing_res is None

    # Missing folder case
    missing_folder_res = tuning_server.reprocess_single_image("NoSuchFolder", "sample_single.jpg", tuning_server.DEFAULT_PARAMS)
    assert missing_folder_res is None


def test_letterbox_pad_and_resize():
    from tuning_server import letterbox_pad_and_resize

    # Test non-square image
    img = np.ones((100, 200, 3), dtype=np.uint8) * 128
    resized, scale, pad_t, pad_l, orig_h, orig_w = letterbox_pad_and_resize(img, target_dim=384)

    assert resized.shape == (384, 384, 3)
    assert orig_h == 100
    assert orig_w == 200
    assert scale == 384.0 / 200.0
    assert pad_l == 0
    assert pad_t == (200 - 100) // 2


def test_project_and_downsample_vectors():
    from tuning_server import project_and_downsample_vectors

    orig_w = 100
    y_top = np.full(orig_w, 20.0)
    y_bot = np.full(orig_w, 80.0)
    y_rpe = np.full(orig_w, 50.0)
    y_sfcm = np.full(orig_w, 70.0)

    top_pts, bot_pts, rpe_pts, sfcm_pts = project_and_downsample_vectors(
        orig_w=orig_w,
        y_top_outer=y_top,
        y_bottom_outer=y_bot,
        y_rpe=y_rpe,
        y_bottom_sfcm=y_sfcm,
        pad_t=10,
        pad_l=0,
        scale=2.0,
        num_points=16
    )

    assert len(top_pts) == 16
    assert len(bot_pts) == 16
    assert len(rpe_pts) == 16
    assert len(sfcm_pts) == 16
    assert top_pts[0][1] == (20.0 + 10) * 2.0
    assert bot_pts[0][1] == (80.0 + 10) * 2.0


def test_path_and_folder_helpers(tmp_path, monkeypatch):
    import tuning_server

    src_dir = tmp_path / "Classified"
    src_dir.mkdir()
    f1 = src_dir / "FolderA"
    f1.mkdir()
    f2 = src_dir / "FolderB"
    f2.mkdir()

    # Create images
    (f1 / "img1.png").touch()
    (f2 / "img2.jpg").touch()

    monkeypatch.setattr(tuning_server, "SOURCE_DIR", src_dir)

    subfolders = tuning_server.get_available_subfolders(src_dir)
    assert subfolders == ["FolderA", "FolderB"]

    found_f1 = tuning_server.find_folder_path("FolderA")
    assert found_f1 == f1
    assert tuning_server.find_folder_path("NonExistent") is None

    img_p = tuning_server.find_image_path(f1, "img1.png")
    assert img_p == (f1 / "img1.png")
    assert tuning_server.find_image_path(f1, "non_existent.png") is None


def test_estimate_adaptive_thresholds():
    from tuning_server import _estimate_adaptive_thresholds

    mock_gray = np.full((100, 100), 20, dtype=np.uint8)
    mock_gray[:15, :15] = 10
    mock_gray[30:70, 30:70] = 120

    t_top, t_bot, otsu_val = _estimate_adaptive_thresholds(mock_gray, top_mult=1.5, bot_mult=3.0)
    assert t_top > 0
    assert t_bot > 0
    assert otsu_val > 0
    assert t_top <= 220.0
    assert t_bot <= 220.0


def test_http_request_handler_endpoints(tmp_path, monkeypatch):
    import io
    import tuning_server

    src_dir = tmp_path / "Classified"
    folder = src_dir / "Folder1"
    folder.mkdir(parents=True)
    cv2_img = np.zeros((100, 100, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite(str(folder / "img.jpg"), cv2_img)

    out_dir = tmp_path / "Output"
    monkeypatch.setattr(tuning_server, "SOURCE_DIR", src_dir)
    monkeypatch.setattr(tuning_server, "OUTPUT_DIR", out_dir)

    class MockSocket:
        def __init__(self, request_bytes):
            self.rfile = io.BytesIO(request_bytes)
            self.wfile = io.BytesIO()

        def makefile(self, mode, *args, **kwargs):
            if "r" in mode:
                return self.rfile
            return self.wfile

        def sendall(self, data):
            self.wfile.write(data)


    # 1. Test GET /api/folders
    mock_sock = MockSocket(b"GET /api/folders HTTP/1.1\r\nHost: localhost\r\n\r\n")
    handler = tuning_server.FineTuningRequestHandler(mock_sock, ("127.0.0.1", 8000), None)
    response = mock_sock.wfile.getvalue().decode("utf-8")
    assert "200 OK" in response
    assert "application/json" in response
    assert "Folder1" in response

    # 2. Test POST /api/reprocess with invalid JSON
    mock_sock = MockSocket(b"POST /api/reprocess HTTP/1.1\r\nHost: localhost\r\nContent-Length: 7\r\n\r\nnotjson")
    handler = tuning_server.FineTuningRequestHandler(mock_sock, ("127.0.0.1", 8000), None)
    response = mock_sock.wfile.getvalue().decode("utf-8")
    assert "400" in response or "Invalid JSON" in response

    # 3. Test POST /api/reprocess with valid JSON
    req_body = json.dumps({"folder": "Folder1", "params": tuning_server.DEFAULT_PARAMS, "random_sample": False}).encode("utf-8")
    req = f"POST /api/reprocess HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(req_body)}\r\n\r\n".encode("utf-8") + req_body
    mock_sock = MockSocket(req)
    handler = tuning_server.FineTuningRequestHandler(mock_sock, ("127.0.0.1", 8000), None)
    response = mock_sock.wfile.getvalue().decode("utf-8")
    assert "200 OK" in response
    assert '"status": "success"' in response

    # 4. Test POST /api/reprocess_single with valid JSON
    req_body = json.dumps({"folder": "Folder1", "filename": "img.jpg", "params": tuning_server.DEFAULT_PARAMS}).encode("utf-8")
    req = f"POST /api/reprocess_single HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(req_body)}\r\n\r\n".encode("utf-8") + req_body
    mock_sock = MockSocket(req)
    handler = tuning_server.FineTuningRequestHandler(mock_sock, ("127.0.0.1", 8000), None)
    response = mock_sock.wfile.getvalue().decode("utf-8")
    assert "200 OK" in response
    assert '"status": "success"' in response

    # 5. Test POST /api/reprocess_single missing file
    req_body = json.dumps({"folder": "Folder1", "filename": "missing.jpg", "params": tuning_server.DEFAULT_PARAMS}).encode("utf-8")
    req = f"POST /api/reprocess_single HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(req_body)}\r\n\r\n".encode("utf-8") + req_body
    mock_sock = MockSocket(req)
    handler = tuning_server.FineTuningRequestHandler(mock_sock, ("127.0.0.1", 8000), None)
    response = mock_sock.wfile.getvalue().decode("utf-8")
    assert "404" in response


def test_parameter_tooltips_and_info_buttons_exist(dashboard_parser):
    _, content = dashboard_parser
    assert 'class="info-btn"' in content or 'class="info-btn ' in content
    assert 'class="tooltip-text"' in content
    assert 'SFCM Choroid Margin (px)' in content
    assert 'Gaussian Smoothness' in content
    assert 'Noise Cutoff' in content


def test_static_asset_serving(monkeypatch):
    import tuning_server
    
    class MockSocket:
        def __init__(self, request_bytes):
            self._rfile = io.BytesIO(request_bytes)
            self.wfile = io.BytesIO()
        def makefile(self, mode, *args, **kwargs):
            if 'r' in mode:
                return self._rfile
            return self.wfile
        def sendall(self, data):
            self.wfile.write(data)

    # Test GET /css/dashboard.css
    mock_sock = MockSocket(b"GET /css/dashboard.css HTTP/1.1\r\nHost: localhost\r\n\r\n")
    handler = tuning_server.FineTuningRequestHandler(mock_sock, ("127.0.0.1", 8000), None)
    response = mock_sock.wfile.getvalue().decode("utf-8")
    assert "200 OK" in response
    assert "text/css" in response
    assert "--accent-cyan" in response

    # Test GET /js/app.js
    mock_sock = MockSocket(b"GET /js/app.js HTTP/1.1\r\nHost: localhost\r\n\r\n")
    handler = tuning_server.FineTuningRequestHandler(mock_sock, ("127.0.0.1", 8000), None)
    response = mock_sock.wfile.getvalue().decode("utf-8")
    assert "200 OK" in response
    assert "application/javascript" in response
    assert "init" in response










