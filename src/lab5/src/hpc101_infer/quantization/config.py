from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QuantizationConfig:
    algorithm: str = "rtn"
    version: str = "1"
    bits: int = 4
    group_size: int = 128
    symmetric: bool = True
    scale_dtype: str = "float16"
    packing: str = "uint8_little_nibble"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    calibration: dict[str, Any] | None = None
    source_checkpoint: str | None = None
    propagate_quantized: bool = True

    def __post_init__(self) -> None:
        if self.algorithm not in {"rtn", "gptq"}:
            raise ValueError(f"unsupported quantization algorithm: {self.algorithm}")
        if self.bits != 4:
            raise ValueError("only 4-bit weight quantization is supported")
        if self.group_size not in {64, 128}:
            raise ValueError("group_size must be 64 or 128")
        if self.scale_dtype not in {"float16", "bfloat16"}:
            raise ValueError("scale_dtype must be float16 or bfloat16")
        if self.packing != "uint8_little_nibble":
            raise ValueError(f"unsupported packing layout: {self.packing}")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_modules"] = list(self.target_modules)
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QuantizationConfig":
        values = dict(raw)
        if "target_modules" in values:
            values["target_modules"] = tuple(values["target_modules"])
        return cls(**values)
