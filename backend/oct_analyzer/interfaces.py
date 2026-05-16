from typing import Any

from pydantic import BaseModel, Field


class LayerFeature(BaseModel):
    name: str
    vote: str
    score: float
    cdf_deciles: list[float] = Field(default_factory=list)


class QualityControlReport(BaseModel):
    signal_range: list[float] = Field(default_factory=list)
    crop_applied: bool = False
    crop_bounds: list[int] = Field(default_factory=list)
    fovea_crop_bounds: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IPNV2Metadata(BaseModel):
    available: bool = False
    mode: str = "unavailable"
    input_shape: list[int] = Field(default_factory=list)
    output_shape: list[int] = Field(default_factory=list)
    warning: str = ""
    previews: dict[str, str] = Field(default_factory=dict)


class ScanResult(BaseModel):
    scan_id: str
    status: str
    filename: str | None = None
    diagnosis: str | None = None
    confidence: float | None = None
    source_format: str | None = None
    volume_shape: list[int] = Field(default_factory=list)
    spacing_mm: list[float] = Field(default_factory=list)
    is_demo_model: bool = True
    qc: QualityControlReport | None = None
    layers: list[LayerFeature] = Field(default_factory=list)
    previews: dict[str, str] = Field(default_factory=dict)
    ipnv2: IPNV2Metadata | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    detail: str | None = None
