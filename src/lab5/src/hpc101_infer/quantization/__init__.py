from hpc101_infer.quantization.config import QuantizationConfig
from hpc101_infer.quantization.packing import (
    dequantize_weight,
    pack_int4,
    unpack_int4,
)
from hpc101_infer.quantization.types import QuantizedWeight


def quantize_checkpoint(*args, **kwargs):
    from hpc101_infer.quantization.pipeline import quantize_checkpoint as run

    return run(*args, **kwargs)


__all__ = [
    "QuantizationConfig",
    "QuantizedWeight",
    "dequantize_weight",
    "pack_int4",
    "quantize_checkpoint",
    "unpack_int4",
]
