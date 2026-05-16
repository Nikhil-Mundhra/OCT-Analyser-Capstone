import builtins
import importlib
import os
import sys
import types
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import torch


def test_runtime_sets_macos_environment(monkeypatch):
    import backend.oct_analyzer.runtime as runtime

    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    runtime.configure_runtime()

    assert runtime.os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert runtime.os.environ["OMP_NUM_THREADS"] == "1"


def test_runtime_ignores_non_macos(monkeypatch):
    import backend.oct_analyzer.runtime as runtime

    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    runtime.configure_runtime()

    assert "KMP_DUPLICATE_LIB_OK" not in runtime.os.environ
    assert "OMP_NUM_THREADS" not in runtime.os.environ


def test_fallback_preprocessing_normalizes_volume():
    from backend.oct_analyzer.pre_processing import _fallback_preprocessing

    tensor = _fallback_preprocessing(np.arange(27, dtype=np.float32).reshape(3, 3, 3))

    assert tensor.shape == (1, 3, 3, 3)
    assert tensor.dtype == torch.float32
    assert torch.isclose(tensor.min(), torch.tensor(0.0))
    assert torch.isclose(tensor.max(), torch.tensor(1.0))


def test_fallback_preprocessing_handles_flat_and_invalid_volumes():
    from backend.oct_analyzer.pre_processing import _fallback_preprocessing

    flat = _fallback_preprocessing(np.ones((2, 2, 2), dtype=np.float32))
    assert torch.equal(flat, torch.zeros((1, 2, 2, 2)))

    with pytest.raises(ValueError, match="Expected a 3D OCT volume"):
        _fallback_preprocessing(np.ones((2, 2), dtype=np.float32))


def test_get_preprocessing_pipeline_uses_fallback_when_monai_missing(monkeypatch):
    import backend.oct_analyzer.pre_processing as pre_processing

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "monai.transforms":
            raise ModuleNotFoundError("No module named 'monai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert pre_processing.get_preprocessing_pipeline() is pre_processing._fallback_preprocessing


def test_get_preprocessing_pipeline_uses_monai_when_available(monkeypatch):
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

    class EnsureChannelFirst:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ScaleIntensityRangePercentiles:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ToTensor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    transforms = types.ModuleType("monai.transforms")
    transforms.Compose = Compose
    transforms.EnsureChannelFirst = EnsureChannelFirst
    transforms.ScaleIntensityRangePercentiles = ScaleIntensityRangePercentiles
    transforms.ToTensor = ToTensor
    monai = types.ModuleType("monai")
    monai.transforms = transforms

    monkeypatch.setitem(sys.modules, "monai", monai)
    monkeypatch.setitem(sys.modules, "monai.transforms", transforms)

    from backend.oct_analyzer.pre_processing import get_preprocessing_pipeline

    pipeline = get_preprocessing_pipeline()

    assert isinstance(pipeline, Compose)
    assert len(pipeline.transforms) == 3


def test_load_oct_volume_rejects_missing_and_unsupported_files(tmp_path):
    from backend.oct_analyzer.data_loader import load_oct_volume

    with pytest.raises(FileNotFoundError):
        load_oct_volume(tmp_path / "missing.vol")

    unsupported = tmp_path / "scan.txt"
    unsupported.write_text("not an OCT file")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_oct_volume(unsupported)


def test_load_oct_volume_reads_vol_with_mocked_eyepy(tmp_path, monkeypatch):
    from backend.oct_analyzer.data_loader import load_oct_volume

    class FakeOct:
        @staticmethod
        def from_heyex_vol(path):
            assert path.endswith("scan.vol")
            return types.SimpleNamespace(
                volume=np.ones((2, 3, 4)),
                meta={"ScaleZ": 1.0, "ScaleX": 2.0, "Distance": 3.0},
            )

    eyepy = types.SimpleNamespace(Oct=FakeOct)
    monkeypatch.setitem(sys.modules, "eyepy", eyepy)
    path = tmp_path / "scan.vol"
    path.write_text("fake")

    volume, spacing = load_oct_volume(path)

    assert volume.shape == (2, 3, 4)
    assert spacing == (1.0, 2.0, 3.0)


def test_load_oct_volume_reads_dicom_with_mocked_simpleitk(tmp_path, monkeypatch):
    from backend.oct_analyzer.data_loader import load_oct_volume

    class FakeImage:
        def GetSpacing(self):
            return (0.1, 0.2, 0.3)

    simpleitk = types.SimpleNamespace(
        ReadImage=lambda path: FakeImage(),
        GetArrayFromImage=lambda image: np.zeros((4, 5, 6)),
    )
    monkeypatch.setitem(sys.modules, "SimpleITK", simpleitk)
    path = tmp_path / "scan.dcm"
    path.write_text("fake")

    volume, spacing = load_oct_volume(path)

    assert volume.shape == (4, 5, 6)
    assert spacing == (0.1, 0.2, 0.3)


def test_load_normalized_scan_reads_vol_with_normalized_contract(tmp_path, monkeypatch):
    import backend.oct_analyzer.data_loader as data_loader

    monkeypatch.setattr(
        data_loader,
        "load_oct_volume",
        lambda path: (np.ones((2, 3, 4)), (0.3, 0.2, 0.1)),
    )
    path = tmp_path / "scan.vol"
    path.write_text("fake")

    scan = data_loader.load_normalized_scan(path)

    assert scan.volume_shape == (2, 3, 4)
    assert scan.spacing_mm == (0.3, 0.2, 0.1)
    assert scan.source_format == "vol"


def test_load_normalized_scan_reads_dicom_with_pydicom_metadata(tmp_path, monkeypatch):
    from backend.oct_analyzer.data_loader import load_normalized_scan

    dataset = types.SimpleNamespace(
        pixel_array=np.arange(12, dtype=np.uint16).reshape(1, 3, 4),
        RescaleSlope="2",
        RescaleIntercept="-1",
        PixelSpacing=["0.2", "0.4"],
        SliceThickness="0.6",
        Modality="OCT",
        SOPClassUID="1.2.3",
    )
    monkeypatch.setitem(sys.modules, "pydicom", types.SimpleNamespace(dcmread=lambda path: dataset))
    path = tmp_path / "scan.dcm"
    path.write_text("fake")

    scan = load_normalized_scan(path)

    assert scan.volume[0, 0, 1] == 1.0
    assert scan.spacing_mm == (0.6, 0.2, 0.4)
    assert scan.metadata["rescale_slope"] == 2.0


def test_load_normalized_scan_reads_zip_stack_and_metadata(tmp_path):
    from PIL import Image
    from backend.oct_analyzer.data_loader import load_normalized_scan

    zip_path = tmp_path / "stack.zip"
    image_a = tmp_path / "slice_002.tif"
    image_b = tmp_path / "slice_001.bmp"
    Image.fromarray(np.full((3, 4), 20, dtype=np.uint8)).save(image_a)
    Image.fromarray(np.full((3, 4), 10, dtype=np.uint8)).save(image_b)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("spacing_z_mm,0.5\npixel_spacing_y,0.1\npixel_spacing_x,0.2\n")

    with ZipFile(zip_path, "w") as archive:
        archive.write(image_a, image_a.name)
        archive.write(image_b, image_b.name)
        archive.write(metadata, metadata.name)

    scan = load_normalized_scan(zip_path)

    assert scan.source_format == "image-stack"
    assert scan.volume_shape == (2, 3, 4)
    assert scan.volume[0, 0, 0] == 10
    assert scan.spacing_mm == (0.5, 0.1, 0.2)


def test_load_normalized_scan_rejects_bad_zip_and_unsupported(tmp_path):
    from backend.oct_analyzer.data_loader import load_normalized_scan

    with pytest.raises(FileNotFoundError):
        load_normalized_scan(tmp_path / "missing.dcm")

    zip_path = tmp_path / "empty.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("notes.txt", "not images")

    with pytest.raises(ValueError, match="does not contain"):
        load_normalized_scan(zip_path)

    unsupported = tmp_path / "scan.nii"
    unsupported.write_text("fake")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_normalized_scan(unsupported)


def test_load_normalized_scan_rejects_mismatched_zip_slices(tmp_path):
    from PIL import Image
    from backend.oct_analyzer.data_loader import load_normalized_scan

    zip_path = tmp_path / "mismatch.zip"
    image_a = tmp_path / "a.tif"
    image_b = tmp_path / "b.tif"
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(image_a)
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_b)
    with ZipFile(zip_path, "w") as archive:
        archive.write(image_a, image_a.name)
        archive.write(image_b, image_b.name)

    with pytest.raises(ValueError, match="same dimensions"):
        load_normalized_scan(zip_path)


def test_data_loader_helpers_cover_metadata_and_shape_edges(tmp_path):
    from backend.oct_analyzer.data_loader import _coerce_spacing, _ensure_zyx_volume, _spacing_from_metadata

    xml_path = tmp_path / "metadata.xml"
    xml_path.write_text("<scan><spacing_z_mm>bad</spacing_z_mm><pixel_spacing_y>0.3</pixel_spacing_y></scan>")
    zip_path = tmp_path / "xml_stack.zip"

    from PIL import Image
    from backend.oct_analyzer.data_loader import load_normalized_scan

    image_path = tmp_path / "slice.tif"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(image_path)
    with ZipFile(zip_path, "w") as archive:
        archive.write(image_path, image_path.name)
        archive.write(xml_path, xml_path.name)

    scan = load_normalized_scan(zip_path)

    assert scan.spacing_mm == (1.0, 0.3, 1.0)
    assert _ensure_zyx_volume(np.ones((2, 2))).shape == (1, 2, 2)
    with pytest.raises(ValueError, match="Expected 2D or 3D"):
        _ensure_zyx_volume(np.ones((1, 1, 1, 1)))
    assert _coerce_spacing((1.0, 2.0)) == (1.0, 1.0, 1.0)
    assert _spacing_from_metadata({"slice_thickness": "oops"}) == (1.0, 1.0, 1.0)


def test_moving_average_and_flatten_with_fallback_filter(monkeypatch):
    import backend.oct_analyzer.anatomical_flattener as flattener

    monkeypatch.setattr(flattener, "gaussian_filter1d", None)
    volume = torch.zeros((1, 5, 2, 2), dtype=torch.float32)
    volume[0, 3] = 10.0

    averaged = flattener._moving_average_z(volume[0].numpy(), window_size=3)
    flattened = flattener.flatten_volume_to_rpe(volume)

    assert averaged.shape == (5, 2, 2)
    assert flattened.shape == volume.shape
    assert flattened.device == volume.device


def test_flattener_import_handles_missing_scipy(monkeypatch):
    import backend.oct_analyzer.anatomical_flattener as anatomical_flattener

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy.ndimage":
            raise ModuleNotFoundError("No module named 'scipy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    reloaded = importlib.reload(anatomical_flattener)

    assert reloaded.gaussian_filter1d is None

    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(anatomical_flattener)


def test_flatten_uses_available_gaussian_filter(monkeypatch):
    import backend.oct_analyzer.anatomical_flattener as flattener

    calls = []

    def fake_filter(volume, sigma, axis):
        calls.append((sigma, axis))
        return volume

    monkeypatch.setattr(flattener, "gaussian_filter1d", fake_filter)
    volume = torch.zeros((1, 4, 1, 2), dtype=torch.float32)
    volume[0, 1, 0, 0] = 5.0
    volume[0, 3, 0, 1] = 5.0

    flattened = flattener.flatten_volume_to_rpe(volume)

    assert calls == [(3, 0)]
    assert flattened.shape == volume.shape


def test_model_fallback_and_runtime_configuration(monkeypatch):
    import backend.oct_analyzer.model as model_module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "monai.networks.nets":
            raise ModuleNotFoundError("No module named 'monai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(model_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(model_module.torch, "get_num_threads", lambda: 2)
    set_threads = []
    monkeypatch.setattr(model_module.torch, "set_num_threads", set_threads.append)

    model = model_module.get_3d_relaynet(num_layers=3)

    assert set_threads == [1]
    assert model(torch.zeros((1, 1, 4, 4, 4))).shape == (1, 3, 4, 4, 4)


def test_model_uses_monai_when_available(monkeypatch):
    class FakeUNet(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    nets = types.ModuleType("monai.networks.nets")
    nets.UNet = FakeUNet
    networks = types.ModuleType("monai.networks")
    networks.nets = nets
    monai = types.ModuleType("monai")
    monai.networks = networks

    monkeypatch.setitem(sys.modules, "monai", monai)
    monkeypatch.setitem(sys.modules, "monai.networks", networks)
    monkeypatch.setitem(sys.modules, "monai.networks.nets", nets)

    from backend.oct_analyzer.model import get_3d_relaynet

    model = get_3d_relaynet(num_layers=7)

    assert isinstance(model, FakeUNet)
    assert model.kwargs["out_channels"] == 7


def test_ordinal_anatomical_loss_returns_positive_scalar():
    from backend.oct_analyzer.losses import OrdinalAnatomicalLoss

    predictions = torch.randn((1, 3, 2, 2, 2), requires_grad=True)
    targets = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)

    loss = OrdinalAnatomicalLoss(num_classes=3)(predictions, targets)
    loss.backward()

    assert loss.item() > 0
    assert predictions.grad is not None


def test_spatial_continuity_loss():
    from backend.oct_analyzer.functions import spatial_continuity_loss

    predictions = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
    predictions[..., 1, 1] = 2.0

    assert spatial_continuity_loss(predictions).item() == 4.0


def test_train_step_with_mocked_components(monkeypatch):
    import backend.oct_analyzer.train as train

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, inputs):
            base = inputs * self.weight
            return torch.cat([base, -base], dim=1)

    model = TinyModel()
    monkeypatch.setattr(train, "device", torch.device("cpu"))
    monkeypatch.setattr(train, "model", model)
    monkeypatch.setattr(train, "criterion", lambda outputs, labels: outputs[:, 0].mean())
    monkeypatch.setattr(train, "optimizer", torch.optim.SGD(model.parameters(), lr=0.1))

    loss = train.train_step({
        "image": torch.ones((1, 1, 2, 2, 2)),
        "label": torch.zeros((1, 1, 2, 2, 2), dtype=torch.long),
    })

    assert loss > 0


class TinyIPNV2(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.kwargs = kwargs

    def forward(self, inputs):
        projection = inputs.mean(dim=2, keepdim=True) * self.scale
        logits = torch.cat([-projection, projection], dim=1)
        features = projection[:, 0, 0].unsqueeze(1)
        return logits, features


def test_ipnv2_adapter_runs_untrained_smoke_and_checkpoint(tmp_path, monkeypatch):
    import backend.oct_analyzer.ipnv2_adapter as adapter

    monkeypatch.delenv(adapter.IPNV2_CHECKPOINT_ENV, raising=False)
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    smoke = adapter.run_ipnv2_smoke_inference(
        volume,
        model_factory=TinyIPNV2,
        target_shape=(4, 5, 6),
    )

    assert smoke.available is True
    assert smoke.mode == "untrained_smoke"
    assert smoke.input_shape == (1, 1, 4, 5, 6)
    assert smoke.output_shape == (1, 2, 1, 5, 6)
    assert smoke.probability_map.shape == (5, 6)
    assert smoke.mask.shape == (5, 6)
    assert "random weights" in smoke.warning

    checkpoint = tmp_path / "ipnv2.pth"
    torch.save({"module.scale": torch.tensor(2.0)}, checkpoint)
    loaded = adapter.run_ipnv2_smoke_inference(
        volume,
        checkpoint_path=checkpoint,
        model_factory=TinyIPNV2,
        target_shape=(4, 5, 6),
    )

    assert loaded.mode == "checkpoint"
    assert loaded.warning == ""
    assert adapter.ipnv2_metadata(loaded, {"ipnv2_overlay": "preview/ipnv2_overlay"})["previews"]
    assert adapter.failed_ipnv2_metadata(RuntimeError("boom"))["available"] is False


def test_ipnv2_adapter_uses_env_and_validates_model_loading(tmp_path, monkeypatch):
    import backend.oct_analyzer.ipnv2_adapter as adapter

    checkpoint = tmp_path / "ipnv2_env.pth"
    torch.save({"state_dict": {"scale": torch.tensor(3.0)}}, checkpoint)
    monkeypatch.setenv(adapter.IPNV2_CHECKPOINT_ENV, str(checkpoint))

    result = adapter.run_ipnv2_smoke_inference(
        np.ones((1, 2, 2), dtype=np.float32),
        model_factory=TinyIPNV2,
        target_shape=(2, 3, 4),
    )

    assert result.mode == "checkpoint"
    assert adapter._resolve_checkpoint(tmp_path / "missing.pth") is None
    assert isinstance(adapter._build_model(None, 1, 1, (160, 100, 100)), torch.nn.Module)

    original_path = adapter.IPNV2_MODEL_PATH
    monkeypatch.setattr(adapter, "IPNV2_MODEL_PATH", tmp_path / "missing.py")
    with pytest.raises(FileNotFoundError):
        adapter._load_ipnv2_model_module()
    monkeypatch.setattr(adapter, "IPNV2_MODEL_PATH", original_path)
    real_spec = adapter.importlib.util.spec_from_file_location
    monkeypatch.setattr(adapter.importlib.util, "spec_from_file_location", lambda *args, **kwargs: None)
    with pytest.raises(ImportError):
        adapter._load_ipnv2_model_module()
    monkeypatch.setattr(adapter.importlib.util, "spec_from_file_location", real_spec)

    with pytest.raises(ValueError, match="3D volume"):
        adapter.run_ipnv2_smoke_inference(np.ones((2, 2), dtype=np.float32), model_factory=TinyIPNV2)


def test_mvp_pipeline_validates_crops_features_and_classifies():
    from backend.oct_analyzer.mvp_pipeline import (
        classify_layers,
        crop_black_padding,
        extract_layer_features,
        placeholder_segment_layers,
        second_order_reflectivity_energy,
        validate_volume,
    )

    volume = np.zeros((8, 6, 6), dtype=np.float32)
    volume[2:6, 1:5, 1:5] = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)

    validation = validate_volume(volume)
    cropped, crop_info = crop_black_padding(volume)
    labels = placeholder_segment_layers(cropped.shape)
    energy = second_order_reflectivity_energy(cropped)
    layers = extract_layer_features(cropped, labels)
    diagnosis, confidence = classify_layers(layers)

    assert validation["signal_range"] == [0.0, 63.0]
    assert cropped.shape == (4, 4, 4)
    assert crop_info["crop_applied"] is True
    assert energy.shape == cropped.shape
    assert len(layers) == 12
    assert len(layers[0]["cdf_deciles"]) == 9
    assert diagnosis in {"DR", "Healthy"}
    assert 0.0 <= confidence <= 1.0


def test_mvp_pipeline_handles_invalid_and_flat_volumes():
    from backend.oct_analyzer.mvp_pipeline import crop_black_padding, validate_volume

    flat = np.ones((2, 2, 2), dtype=np.float32)
    cropped, crop_info = crop_black_padding(flat)

    assert cropped.shape == flat.shape
    assert crop_info["crop_applied"] is False
    assert validate_volume(flat)["warnings"] == ["Volume has no intensity variation"]

    with pytest.raises(ValueError, match="3D OCT volume"):
        validate_volume(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty"):
        validate_volume(np.empty((0, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN"):
        validate_volume(np.array([[[np.nan]]], dtype=np.float32))


def test_process_scan_returns_completed_payload_and_previews(tmp_path, monkeypatch):
    import backend.oct_analyzer.mvp_pipeline as pipeline
    from backend.oct_analyzer.ipnv2_adapter import IPNV2Result
    from backend.oct_analyzer.scan_types import NormalizedScan

    monkeypatch.setattr(pipeline, "get_preprocessing_pipeline", lambda: lambda volume: torch.from_numpy(volume).unsqueeze(0).float())
    monkeypatch.setattr(pipeline, "flatten_volume_to_rpe", lambda tensor: tensor)
    monkeypatch.setattr(
        pipeline,
        "run_ipnv2_smoke_inference",
        lambda volume: IPNV2Result(
            available=True,
            mode="untrained_smoke",
            input_shape=(1, 1, 4, 5, 5),
            output_shape=(1, 2, 1, 5, 5),
            warning="IPN-V2 is running with random weights; output validates plumbing only.",
            probability_map=np.linspace(0, 1, 25, dtype=np.float32).reshape(5, 5),
            mask=np.ones((5, 5), dtype=np.uint8),
            reference_image=np.ones((5, 5), dtype=np.float32),
        ),
    )

    volume = np.zeros((6, 5, 5), dtype=np.float32)
    volume[1:5, 1:4, 1:4] = 5
    scan = NormalizedScan(volume=volume, spacing_mm=(0.5, 0.1, 0.1), source_format="dicom")

    result = pipeline.process_scan(scan, preview_dir=tmp_path)

    assert result["status"] == "completed"
    assert result["is_demo_model"] is True
    assert set(result["previews"]) == {"raw", "cropped", "overlay", "features", "ipnv2_probability", "ipnv2_overlay"}
    assert result["ipnv2"]["mode"] == "untrained_smoke"
    assert result["ipnv2"]["previews"] == {
        "ipnv2_probability": "preview/ipnv2_probability",
        "ipnv2_overlay": "preview/ipnv2_overlay",
    }
    assert (tmp_path / "overlay.png").exists()
    assert (tmp_path / "ipnv2_overlay.png").exists()


def test_process_scan_keeps_working_when_ipnv2_fails(tmp_path, monkeypatch):
    import backend.oct_analyzer.mvp_pipeline as pipeline
    from backend.oct_analyzer.scan_types import NormalizedScan

    monkeypatch.setattr(pipeline, "get_preprocessing_pipeline", lambda: lambda volume: torch.from_numpy(volume).unsqueeze(0).float())
    monkeypatch.setattr(pipeline, "flatten_volume_to_rpe", lambda tensor: tensor)

    def broken_ipnv2(volume):
        raise RuntimeError("no model")

    monkeypatch.setattr(pipeline, "run_ipnv2_smoke_inference", broken_ipnv2)
    scan = NormalizedScan(volume=np.ones((3, 3, 3), dtype=np.float32), spacing_mm=(1.0, 1.0, 1.0), source_format="test")

    result = pipeline.process_scan(scan, preview_dir=tmp_path)

    assert result["status"] == "completed"
    assert result["ipnv2"]["available"] is False
    assert "no model" in result["ipnv2"]["warning"]
    assert "ipnv2_overlay" not in result["previews"]


def test_preview_rejects_unknown_kind(tmp_path):
    from backend.oct_analyzer.preview import _ipnv2_overlay_image, preview_path

    assert preview_path(tmp_path, "ipnv2_overlay") == tmp_path / "ipnv2_overlay.png"
    assert preview_path(tmp_path, "raw") == tmp_path / "raw.png"
    resized_overlay = _ipnv2_overlay_image(
        reference=np.ones((3, 3), dtype=np.float32),
        probability=np.ones((2, 2), dtype=np.float32),
        mask=np.ones((2, 2), dtype=np.uint8),
    )
    assert resized_overlay.size == (3, 3)
    with pytest.raises(ValueError, match="Unsupported preview"):
        preview_path(tmp_path, "unknown")


def test_normalized_scan_rejects_non_3d_shape():
    from backend.oct_analyzer.scan_types import NormalizedScan

    scan = NormalizedScan(volume=np.ones((2, 2)), spacing_mm=(1.0, 1.0, 1.0), source_format="test")

    with pytest.raises(ValueError, match="Expected 3D"):
        scan.volume_shape


def test_api_upload_get_and_preview(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.oct_analyzer.api as api
    from backend.oct_analyzer.scan_types import NormalizedScan

    monkeypatch.setattr(api, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api, "PREVIEW_DIR", tmp_path / "previews")

    def fake_load(path):
        return NormalizedScan(
            volume=np.ones((3, 3, 3), dtype=np.float32),
            spacing_mm=(1.0, 1.0, 1.0),
            source_format="dicom",
        )

    monkeypatch.setattr(api, "load_normalized_scan", fake_load)

    def fake_process(scan, preview_dir):
        preview_dir.mkdir(parents=True, exist_ok=True)
        (preview_dir / "raw.png").write_bytes(b"png")
        return {
            "status": "completed",
            "diagnosis": "Healthy",
            "confidence": 1.0,
            "source_format": scan.source_format,
            "volume_shape": [3, 3, 3],
            "spacing_mm": [1.0, 1.0, 1.0],
            "is_demo_model": True,
            "qc": {"warnings": []},
            "layers": [],
            "previews": {"raw": "preview/raw"},
            "ipnv2": {
                "available": True,
                "mode": "untrained_smoke",
                "input_shape": [1, 1, 160, 100, 100],
                "output_shape": [1, 2, 1, 100, 100],
                "warning": "smoke",
                "previews": {"ipnv2_overlay": "preview/ipnv2_overlay"},
            },
            "metadata": {},
        }

    monkeypatch.setattr(api, "process_scan", fake_process)
    client = TestClient(api.app)

    response = client.post("/api/scans", files={"file": ("scan.dcm", b"fake", "application/dicom")})
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "completed"
    assert payload["previews"]["raw"].startswith("/api/scans/")
    assert payload["ipnv2"]["previews"]["ipnv2_overlay"].startswith("/api/scans/")

    scan_response = client.get(f"/api/scans/{payload['scan_id']}")
    preview_response = client.get(f"/api/scans/{payload['scan_id']}/preview/raw")

    assert scan_response.status_code == 200
    assert preview_response.status_code == 200
    assert preview_response.headers["content-type"] == "image/png"


def test_api_rejects_bad_uploads_and_missing_resources(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import backend.oct_analyzer.api as api

    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api, "PREVIEW_DIR", tmp_path / "previews")
    client = TestClient(api.app)

    bad_type = client.post("/api/scans", files={"file": ("scan.txt", b"fake", "text/plain")})
    missing_scan = client.get("/api/scans/missing")
    missing_preview = client.get("/api/scans/missing/preview/raw")

    assert bad_type.status_code == 400
    assert missing_scan.status_code == 404
    assert missing_preview.status_code == 404

    def broken_load(path):
        raise ValueError("broken scan")

    monkeypatch.setattr(api, "load_normalized_scan", broken_load)
    broken = client.post("/api/scans", files={"file": ("scan.dcm", b"fake", "application/dicom")})

    assert broken.status_code == 422
    assert broken.json()["detail"] == "broken scan"

    api.SCAN_STORE["known"] = {"scan_id": "known", "status": "completed"}
    unknown_preview = client.get("/api/scans/known/preview/unknown")
    absent_preview = client.get("/api/scans/known/preview/raw")

    assert unknown_preview.status_code == 404
    assert absent_preview.status_code == 404


def test_main_success_file_not_found_and_unexpected_error(monkeypatch, capsys):
    import backend.oct_analyzer.main as main

    monkeypatch.setattr(main, "load_oct_volume", lambda path: (np.zeros((2, 2, 2)), (1, 1, 1)))
    monkeypatch.setattr(main, "get_preprocessing_pipeline", lambda: lambda volume: torch.ones((1, 2, 2, 2)))
    monkeypatch.setattr(main, "flatten_volume_to_rpe", lambda tensor: tensor)

    main.main()
    assert "Success! Final Tensor Shape" in capsys.readouterr().out

    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(main, "load_oct_volume", missing)
    main.main()
    assert "file was not found" in capsys.readouterr().out

    def broken(path):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "load_oct_volume", broken)
    main.main()
    assert "An unexpected error occurred: boom" in capsys.readouterr().out
