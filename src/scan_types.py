from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SpacingZYX = tuple[float, float, float]


@dataclass(frozen=True)
class NormalizedScan:
    volume: np.ndarray
    spacing_mm: SpacingZYX
    source_format: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source_path: Path | None = None

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        shape = tuple(int(value) for value in self.volume.shape)
        if len(shape) != 3:
            raise ValueError(f"Expected 3D OCT volume, got shape {shape}")
        return shape
