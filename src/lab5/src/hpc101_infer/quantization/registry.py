from __future__ import annotations

from hpc101_infer.quantization.methods import (
    GPTQQuantizationMethod,
    RTNQuantizationMethod,
)
from hpc101_infer.quantization.types import QuantizationMethod

_METHODS = {
    "gptq": GPTQQuantizationMethod,
    "rtn": RTNQuantizationMethod,
}


def available_quantization_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))


def create_quantization_method(name: str) -> QuantizationMethod:
    try:
        method = _METHODS[name]
    except KeyError as error:
        raise ValueError(f"unknown quantization method: {name}") from error
    return method()
