from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import yaml

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


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
class EngineConfig:
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    max_batch_size: int = 1
    scheduler_batch_size: int = 1
    max_sequence_length: int = 4096
    attention_backend: str = "eager"
    linear_backend: str = "bf16"
    scheduler_backend: str = "static_batch"
    seed: int = 0
    synchronize_metrics: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> EngineConfig:
        allowed_fields = {
            config_field.name for config_field in fields(cls) if config_field.init
        }
        _reject_unknown_options(raw, allowed_fields, "config.engine")
        values: dict[str, object] = {
            key: value for key, value in raw.items() if value is not None
        }

        dtype = raw.get("dtype")
        if dtype is not None:
            if not isinstance(dtype, str) or dtype not in DTYPES:
                choices = ", ".join(DTYPES)
                raise ValueError(f"config.engine.dtype must be one of: {choices}")
            values["dtype"] = DTYPES[dtype]

        for key in (
            "device",
            "attention_backend",
            "linear_backend",
            "scheduler_backend",
        ):
            value = _optional_string(raw.get(key), f"config.engine.{key}")
            if value is not None:
                values[key] = value

        for key in (
            "max_batch_size",
            "scheduler_batch_size",
            "max_sequence_length",
            "seed",
        ):
            value = _optional_integer(raw.get(key), f"config.engine.{key}")
            if value is not None:
                values[key] = value

        synchronize_metrics = _optional_boolean(
            raw.get("synchronize_metrics"),
            "config.engine.synchronize_metrics",
        )
        if synchronize_metrics is not None:
            values["synchronize_metrics"] = synchronize_metrics

        if "scheduler_batch_size" in values and "max_batch_size" not in values:
            values["max_batch_size"] = values["scheduler_batch_size"]
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        raw = _load_yaml_mapping(path)
        engine = _require_mapping(raw.get("engine"), "config.engine")
        return cls.from_mapping(engine)

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.scheduler_batch_size <= 0:
            raise ValueError("scheduler_batch_size must be positive")
        if self.scheduler_batch_size > self.max_batch_size:
            raise ValueError("scheduler_batch_size must not exceed max_batch_size")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.attention_backend != "eager":
            raise ValueError("only the eager attention backend is implemented")
        if self.linear_backend not in {"bf16", "int4_reference"}:
            raise ValueError("linear_backend must be bf16 or int4_reference")
        if self.scheduler_backend != "static_batch":
            raise ValueError("only the static_batch scheduler is implemented")
