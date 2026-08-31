"""
data/preprocessing/params.py

Centralized JSON Parameter Configuration Manager (folder_params.json).
Provides single source of truth for folder-specific preprocessing parameters across
all dataset pipelines, server API endpoints, and processing scripts.
Supports compass_location selection ('auto', 'bottom_left', 'bottom_right').
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PARAMS_FILE = PROJECT_ROOT / "data" / "folder_params.json"
SOURCE_DIR = Path('/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified')

DEFAULT_PARAMS = {
    "top_noise_mult": 1.5,
    "bot_noise_mult": 3.0,
    "shadow_bridge_pct": 20,
    "gaussian_sigma": 15,
    "margin_top": 15,
    "margin_bottom": 15,
    "compass_ui_enabled": False,
    "compass_location": "auto",
    "use_sfcm": False,
    "sfcm_margin_bottom": 15,
    "sfcm_gaussian_sigma": 15,
    "sfcm_n_clusters": 3,
    "sfcm_fuzziness_m": 2.0,
    "rpe_smooth_weight": 0.20,
    "rpe_depth_weight": 0.40,
    "rpe_gradient_weight": 0.30,
    "rpe_bottom_env_size": 15,
    "sfcm_slack_bottom_px": 20,
    "detect_caverns": False,
    "cavern_transmission_threshold": 1.35,
    "holes_enabled": True,
    "hole_min_area": 25,
    "hole_max_area": 15000,
    "hole_contrast_offset": 8,
    "hole_local_window": 15,
    "hole_max_aspect_ratio": 2.8,
    "use_dp_ilm": True,
    "ilm_gradient_weight": 0.70,
    "ilm_smooth_weight": 0.25,
    "auto_mode": True
}


def is_spectralis_folder(folder_name: str) -> bool:
    s = folder_name.lower()
    return any(k in s for k in ('chu', 'mh_', 'mh38', 'mh84', 'mh69', 'macular-hole'))


def initialize_default_params_file() -> dict:
    """
    Scans dataset directory and populates folder_params.json with default entries
    for all 20 subfolders. Adds compass_location='auto' if missing.
    """
    subfolders = []
    if SOURCE_DIR.exists():
        for p in SOURCE_DIR.rglob('*'):
            if p.is_dir() and any(f.suffix.lower() in {'.jpg', '.jpeg', '.png'} for f in p.glob('*')):
                subfolders.append(p.name)
    subfolders = sorted(list(set(subfolders)))

    params_dict = {}
    if PARAMS_FILE.exists():
        try:
            with open(PARAMS_FILE, 'r') as f:
                params_dict = json.load(f)
        except Exception:
            params_dict = {}

    updated = False
    for folder, cfg in list(params_dict.items()):
        if "compass_location" not in cfg:
            cfg["compass_location"] = "auto"
            updated = True
        if "margin" in cfg and ("margin_top" not in cfg or "margin_bottom" not in cfg):
            cfg["margin_top"] = cfg.get("margin", 15)
            cfg["margin_bottom"] = cfg.get("margin", 15)
            del cfg["margin"]
            updated = True

    for folder in subfolders:
        if folder not in params_dict:
            folder_cfg = DEFAULT_PARAMS.copy()
            if is_spectralis_folder(folder):
                folder_cfg["compass_ui_enabled"] = True
            params_dict[folder] = folder_cfg
            updated = True

    if updated or not PARAMS_FILE.exists():
        save_all_params(params_dict)

    return params_dict


def load_all_params() -> dict:
    if not PARAMS_FILE.exists():
        return initialize_default_params_file()
    try:
        with open(PARAMS_FILE, 'r') as f:
            data = json.load(f)
            updated = False
            for k, v in data.items():
                if "compass_location" not in v:
                    v["compass_location"] = "auto"
                    updated = True
                if "margin" in v and ("margin_top" not in v or "margin_bottom" not in v):
                    v["margin_top"] = v.get("margin", 15)
                    v["margin_bottom"] = v.get("margin", 15)
                    del v["margin"]
                    updated = True
            if updated:
                save_all_params(data)
            return data
    except Exception:
        return initialize_default_params_file()


def save_all_params(params_dict: dict):
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_FILE, 'w') as f:
        json.dump(params_dict, f, indent=2)


def get_folder_params(folder_name: str) -> dict:
    all_p = load_all_params()
    if folder_name in all_p:
        return all_p[folder_name]
    
    cfg = DEFAULT_PARAMS.copy()
    if is_spectralis_folder(folder_name):
        cfg["compass_ui_enabled"] = True
    return cfg
