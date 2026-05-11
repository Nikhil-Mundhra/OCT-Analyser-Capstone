import builtins
import importlib
import os
import sys
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import torch


def test_runtime_sets_macos_environment(monkeypatch):
    import src.runtime as runtime

    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    runtime.configure_runtime()

    assert runtime.os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert runtime.os.environ["OMP_NUM_THREADS"] == "1"


def test_runtime_ignores_non_macos(monkeypatch):
    import src.runtime as runtime

    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    runtime.configure_runtime()

    assert "KMP_DUPLICATE_LIB_OK" not in runtime.os.environ
    assert "OMP_NUM_THREADS" not in runtime.os.environ


def test_fallback_preprocessing_normalizes_volume():
    from src.pre_processing import _fallback_preprocessing

    tensor = _fallback_preprocessing(np.arange(27, dtype=np.float32).reshape(3, 3, 3))

    assert tensor.shape == (1, 3, 3, 3)
    assert tensor.dtype == torch.float32
    assert torch.isclose(tensor.min(), torch.tensor(0.0))
    assert torch.isclose(tensor.max(), torch.tensor(1.0))


def test_fallback_preprocessing_handles_flat_and_invalid_volumes():
    from src.pre_processing import _fallback_preprocessing

    flat = _fallback_preprocessing(np.ones((2, 2, 2), dtype=np.float32))
    assert torch.equal(flat, torch.zeros((1, 2, 2, 2)))

    with pytest.raises(ValueError, match="Expected a 3D OCT volume"):
        _fallback_preprocessing(np.ones((2, 2), dtype=np.float32))


def test_get_preprocessing_pipeline_uses_fallback_when_monai_missing(monkeypatch):
    import src.pre_processing as pre_processing

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

    from src.pre_processing import get_preprocessing_pipeline

    pipeline = get_preprocessing_pipeline()

    assert isinstance(pipeline, Compose)
    assert len(pipeline.transforms) == 3


def test_load_oct_volume_rejects_missing_and_unsupported_files(tmp_path):
    from src.data_loader import load_oct_volume

    with pytest.raises(FileNotFoundError):
        load_oct_volume(tmp_path / "missing.vol")

    unsupported = tmp_path / "scan.txt"
    unsupported.write_text("not an OCT file")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_oct_volume(unsupported)


def test_load_oct_volume_reads_vol_with_mocked_eyepy(tmp_path, monkeypatch):
    from src.data_loader import load_oct_volume

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
    from src.data_loader import load_oct_volume

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


def test_moving_average_and_flatten_with_fallback_filter(monkeypatch):
    import anatomical_flattener as flattener

    monkeypatch.setattr(flattener, "gaussian_filter1d", None)
    volume = torch.zeros((1, 5, 2, 2), dtype=torch.float32)
    volume[0, 3] = 10.0

    averaged = flattener._moving_average_z(volume[0].numpy(), window_size=3)
    flattened = flattener.flatten_volume_to_rpe(volume)

    assert averaged.shape == (5, 2, 2)
    assert flattened.shape == volume.shape
    assert flattened.device == volume.device


def test_flatten_uses_available_gaussian_filter(monkeypatch):
    import anatomical_flattener as flattener

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
    import src.model as model_module

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

    from src.model import get_3d_relaynet

    model = get_3d_relaynet(num_layers=7)

    assert isinstance(model, FakeUNet)
    assert model.kwargs["out_channels"] == 7


def test_ordinal_anatomical_loss_returns_positive_scalar():
    from losses import OrdinalAnatomicalLoss

    predictions = torch.randn((1, 3, 2, 2, 2), requires_grad=True)
    targets = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)

    loss = OrdinalAnatomicalLoss(num_classes=3)(predictions, targets)
    loss.backward()

    assert loss.item() > 0
    assert predictions.grad is not None


def test_spatial_continuity_loss():
    from functions import spatial_continuity_loss

    predictions = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
    predictions[..., 1, 1] = 2.0

    assert spatial_continuity_loss(predictions).item() == 4.0


def test_train_step_with_mocked_components(monkeypatch):
    import src.train as train

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


def test_main_success_file_not_found_and_unexpected_error(monkeypatch, capsys):
    import main

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
