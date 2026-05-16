from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .runtime import configure_runtime

configure_runtime()

import torch
import torch.nn.functional as F


IPNV2_MODEL_PATH = Path(__file__).resolve().parents[2] / "IPNV2_pytorch" / "model.py"
IPNV2_CHECKPOINT_ENV = "IPNV2_CHECKPOINT"
IPNV2_TARGET_SHAPE = (160, 100, 100)


@dataclass(frozen=True)
class IPNV2Result:
    available: bool
    mode: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    warning: str
    probability_map: np.ndarray
    mask: np.ndarray
    reference_image: np.ndarray


def run_ipnv2_smoke_inference(
    volume: np.ndarray,
    checkpoint_path: str | Path | None = None,
    model_factory: Callable[..., torch.nn.Module] | None = None,
    channels: int | None = None,
    plane_perceptron_channels: int | None = None,
    target_shape: tuple[int, int, int] = IPNV2_TARGET_SHAPE,
) -> IPNV2Result:
    """
    Runs IPN-V2 against a normalized OCT/OCTA volume for local test-drive use.

    When no checkpoint is supplied, the model is intentionally untrained and the
    returned mode makes that explicit. This validates integration plumbing only.
    """
    checkpoint = _resolve_checkpoint(checkpoint_path)
    mode = "checkpoint" if checkpoint is not None else "untrained_smoke"
    model_channels = channels if channels is not None else (64 if checkpoint is not None else 8)
    plane_channels = plane_perceptron_channels if plane_perceptron_channels is not None else model_channels

    tensor, reference = _volume_to_ipnv2_tensor(volume, target_shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_factory, model_channels, plane_channels, target_shape)

    if checkpoint is not None:
        _load_checkpoint(model, checkpoint, device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        logits, _features = model(tensor.to(device))
        probability = torch.softmax(logits, dim=1)[0, 1, 0].detach().cpu().numpy().astype(np.float32)

    warning = (
        "IPN-V2 is running with random weights; output validates plumbing only."
        if mode == "untrained_smoke"
        else ""
    )
    mask = (probability >= 0.5).astype(np.uint8)

    return IPNV2Result(
        available=True,
        mode=mode,
        input_shape=tuple(int(value) for value in tensor.shape),
        output_shape=tuple(int(value) for value in logits.shape),
        warning=warning,
        probability_map=probability,
        mask=mask,
        reference_image=reference,
    )


def ipnv2_metadata(result: IPNV2Result, previews: dict[str, str] | None = None) -> dict:
    return {
        "available": result.available,
        "mode": result.mode,
        "input_shape": list(result.input_shape),
        "output_shape": list(result.output_shape),
        "warning": result.warning,
        "previews": previews or {},
    }


def failed_ipnv2_metadata(error: Exception) -> dict:
    return {
        "available": False,
        "mode": "unavailable",
        "input_shape": [],
        "output_shape": [],
        "warning": f"IPN-V2 unavailable: {error}",
        "previews": {},
    }


def _resolve_checkpoint(checkpoint_path: str | Path | None) -> Path | None:
    raw_path = checkpoint_path or os.environ.get(IPNV2_CHECKPOINT_ENV)
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.exists() else None


def _build_model(
    model_factory: Callable[..., torch.nn.Module] | None,
    channels: int,
    plane_perceptron_channels: int,
    target_shape: tuple[int, int, int],
) -> torch.nn.Module:
    if model_factory is not None:
        return model_factory(
            in_channels=1,
            channels=channels,
            plane_perceptron_channels=plane_perceptron_channels,
            n_classes=2,
            block_size=list(target_shape),
            plane_perceptron="UNet_3Plus",
        )

    module = _load_ipnv2_model_module()
    return module.IPN_V2(
        in_channels=1,
        channels=channels,
        plane_perceptron_channels=plane_perceptron_channels,
        n_classes=2,
        block_size=list(target_shape),
        plane_perceptron="UNet_3Plus",
    )


def _load_ipnv2_model_module():
    if not IPNV2_MODEL_PATH.exists():
        raise FileNotFoundError(IPNV2_MODEL_PATH)
    spec = importlib.util.spec_from_file_location("ipnv2_model", IPNV2_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load IPN-V2 model from {IPNV2_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean_state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    model.load_state_dict(clean_state)


def _volume_to_ipnv2_tensor(volume: np.ndarray, target_shape: tuple[int, int, int]) -> tuple[torch.Tensor, np.ndarray]:
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"IPN-V2 expects a 3D volume, got shape {array.shape}")
    array = np.nan_to_num(array, copy=False)
    low = float(np.percentile(array, 1.0))
    high = float(np.percentile(array, 99.0))
    if high <= low:
        normalized = np.zeros_like(array, dtype=np.float32)
    else:
        normalized = np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)

    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    reference = resized[0, 0].mean(dim=0).detach().cpu().numpy().astype(np.float32)
    return resized.float(), reference
