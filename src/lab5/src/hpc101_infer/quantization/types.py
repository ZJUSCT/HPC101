from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import torch
from torch import nn

from hpc101_infer.quantization.config import QuantizationConfig

SCALE_DTYPES: Mapping[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class QuantizedWeight:
    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor | None
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    bits: int
    group_size: int
    symmetric: bool
    packing: str


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: torch.dtype
    shard: str


@dataclass(frozen=True)
class QuantizedModuleManifest:
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    qweight_key: str
    scales_key: str
    zeros_key: str | None
    bits: int
    group_size: int
    symmetric: bool
    packing: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_shape": list(self.original_shape),
            "padded_shape": list(self.padded_shape),
            "qweight_key": self.qweight_key,
            "scales_key": self.scales_key,
            "zeros_key": self.zeros_key,
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "packing": self.packing,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QuantizedModuleManifest":
        return cls(
            original_shape=tuple(raw["original_shape"]),
            padded_shape=tuple(raw["padded_shape"]),
            qweight_key=raw["qweight_key"],
            scales_key=raw["scales_key"],
            zeros_key=raw.get("zeros_key"),
            bits=raw["bits"],
            group_size=raw["group_size"],
            symmetric=raw["symmetric"],
            packing=raw["packing"],
        )


@dataclass
class LayerContext:
    layer_index: int
    layer_type: str
    layer: nn.Module
    activations: Mapping[str, torch.Tensor] | None
    config: QuantizationConfig
    target_modules: tuple[str, ...]


@dataclass
class LayerQuantizationResult:
    weights: dict[str, QuantizedWeight]
    metadata: dict[str, Any]


class QuantizationMethod(Protocol):
    name: str
    version: str

    def calibrate_layer(self, context: LayerContext) -> Any: ...

    def quantize_layer(
        self, context: LayerContext, state: Any
    ) -> LayerQuantizationResult: ...
