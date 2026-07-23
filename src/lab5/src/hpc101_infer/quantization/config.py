from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _load_yaml_mapping(path: str | Path) -> dict[str, object]:
    try:
        with Path(path).open() as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML: {error}") from error
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a YAML mapping")
    return raw


def _require_mapping(value: object, location: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a YAML mapping")
    return value


def _reject_unknown_options(
    values: Mapping[str, object],
    allowed: set[str],
    location: str,
) -> None:
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"unknown {location} option(s): {names}")


def _optional_string(value: object, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string")
    return value


def _optional_integer(value: object, location: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{location} must be an integer")
    return value


def _optional_boolean(value: object, location: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be true or false")
    return value


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
    source_checkpoint: str | None = field(
        default=None,
        metadata={"yaml": False},
    )
    propagate_quantized: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> QuantizationConfig:
        allowed_fields = {
            config_field.name
            for config_field in fields(cls)
            if config_field.init and config_field.metadata.get("yaml", True)
        }
        _reject_unknown_options(raw, allowed_fields, "config.quantization")
        values: dict[str, object] = {
            key: value for key, value in raw.items() if value is not None
        }

        for key in ("algorithm", "scale_dtype", "packing"):
            value = _optional_string(raw.get(key), f"config.quantization.{key}")
            if value is not None:
                values[key] = value

        version = raw.get("version")
        if version is not None:
            if not isinstance(version, (str, int)) or isinstance(version, bool):
                raise ValueError("config.quantization.version must be a string or integer")
            values["version"] = str(version)

        for key in ("bits", "group_size"):
            value = _optional_integer(raw.get(key), f"config.quantization.{key}")
            if value is not None:
                values[key] = value

        for key in ("symmetric", "propagate_quantized"):
            value = _optional_boolean(raw.get(key), f"config.quantization.{key}")
            if value is not None:
                values[key] = value

        target_modules = raw.get("target_modules")
        if target_modules is not None:
            if not isinstance(target_modules, (list, tuple)) or not all(
                isinstance(module, str) for module in target_modules
            ):
                raise ValueError(
                    "config.quantization.target_modules must be a list of strings"
                )
            values["target_modules"] = tuple(target_modules)

        calibration = raw.get("calibration")
        if calibration is not None:
            if not isinstance(calibration, dict):
                raise ValueError("config.quantization.calibration must be a mapping")
            values["calibration"] = calibration
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> QuantizationConfig:
        raw = _load_yaml_mapping(path)
        quantization = _require_mapping(
            raw.get("quantization"),
            "config.quantization",
        )
        return cls.from_mapping(quantization)

    def __post_init__(self) -> None:
        if self.algorithm not in {"rtn", "awq", "gptq"}:
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
    def from_dict(cls, raw: dict[str, Any]) -> QuantizationConfig:
        values = dict(raw)
        if "target_modules" in values:
            values["target_modules"] = tuple(values["target_modules"])
        return cls(**values)
