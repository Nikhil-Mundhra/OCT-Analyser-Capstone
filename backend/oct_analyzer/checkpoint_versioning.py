"""
Centralized Checkpoint Versioning Engine for OCT-Analyser.
Standardizes version directory creation (v1, v2, ...), metadata tracking, and deployment registration.
"""

import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def get_git_info() -> Dict[str, str]:
    """Extract current git hash, commit message, and branch safely."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        git_msg = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%B"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        git_hash, git_msg, git_branch = "Unknown", "Unknown", "Unknown"
    return {"hash": git_hash, "msg": git_msg, "branch": git_branch}


def get_hardware_info() -> Dict[str, str]:
    """Extract accelerator device and system info."""
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device_name = "Apple Silicon MPS (Metal Performance Shaders)"

    return {
        "device": device_name,
        "torch_version": torch.__version__,
        "os_arch": f"{platform.system()} {platform.release()} ({platform.machine()})"
    }


def resolve_and_create_version_dir(
    base_dir: Path,
    requested_version: str = "auto",
    args_dict: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    logger_inst: Optional[Any] = None
) -> Tuple[Path, str]:
    """
    Resolves next available semantic version directory (v1, v2, ...) or uses requested version,
    and initializes version_metadata.md with system and execution specs.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if requested_version == "auto" or not requested_version:
        existing_versions = []
        for child in base_dir.iterdir():
            if child.is_dir() and child.name.startswith("v"):
                try:
                    ver_num = int(child.name[1:])
                    existing_versions.append(ver_num)
                except ValueError:
                    pass
        next_ver_num = (max(existing_versions) + 1) if existing_versions else 1
        version_tag = f"v{next_ver_num}"
    else:
        version_tag = requested_version if requested_version.startswith("v") else f"v{requested_version}"

    version_dir = base_dir / version_tag
    version_dir.mkdir(parents=True, exist_ok=True)

    meta_path = version_dir / "version_metadata.md"
    if not meta_path.exists():
        git_info = get_git_info()
        hw_info = get_hardware_info()
        cmd_line = "python3 " + " ".join(sys.argv)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        args_md = ""
        if args_dict:
            args_md = "\n".join([f"- **`{k}`**: `{v}`" for k, v in args_dict.items()])

        extra_md = ""
        if extra_metadata:
            extra_md = "\n".join([f"- **{k}**: `{v}`" for k, v in extra_metadata.items()])

        md_content = f"""# Model Weight Version Metadata — `{version_tag}`

## Run Identification
- **Version Tag**: `{version_tag}`
- **Date & Time**: `{now_str}`
- **Git Branch**: `{git_info['branch']}`
- **Git Commit Hash**: `{git_info['hash']}`
- **Commit Message**: `{git_info['msg']}`

## Hardware & System Specs
- **Compute Device**: `{hw_info['device']}`
- **PyTorch Version**: `{hw_info['torch_version']}`
- **Host OS / Arch**: `{hw_info['os_arch']}`

## Execution Command Line
```bash
{cmd_line}
```

## Training Arguments & Hyperparameters
{args_md if args_md else 'None specified'}

## Model Performance & Convergence Metrics
{extra_md if extra_md else 'Pending training completion'}
"""
        with open(meta_path, "w") as f_meta:
            f_meta.write(md_content)

        if logger_inst:
            logger_inst.info(
                f"=== Checkpoint Versioning Engine: Initialized {version_dir} with version_metadata.md ==="
            )

    return version_dir, version_tag


def update_version_metadata_metrics(version_dir: Path, metrics: Dict[str, Any]) -> None:
    """Appends or updates verified metrics section in version_metadata.md upon training completion."""
    meta_path = Path(version_dir) / "version_metadata.md"
    if not meta_path.exists():
        return

    content = meta_path.read_text()
    metrics_md = "\n".join([f"- **{k}**: `{v}`" for k, v in metrics.items()])
    
    if "## Model Performance & Convergence Metrics" in content:
        parts = content.split("## Model Performance & Convergence Metrics")
        new_content = parts[0] + "## Model Performance & Convergence Metrics\n" + metrics_md + "\n"
    else:
        new_content = content + "\n\n## Model Performance & Convergence Metrics\n" + metrics_md + "\n"

    meta_path.write_text(new_content)
