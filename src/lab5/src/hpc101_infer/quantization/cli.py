from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from transformers import AutoTokenizer

from hpc101_infer.data import load_token_ids
from hpc101_infer.quantization import quantize_checkpoint
from hpc101_infer.quantization.config import QuantizationConfig
from hpc101_infer.quantization.registry import available_quantization_methods


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1)")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a W4A16 Gemma 4 checkpoint")
    parser.add_argument("--model", required=True, help="source Hugging Face checkpoint")
    parser.add_argument("--output", required=True, help="output checkpoint directory")
    parser.add_argument(
        "--algorithm", choices=available_quantization_methods(), default="rtn"
    )
    parser.add_argument("--group-size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--asymmetric", action="store_true")
    parser.add_argument(
        "--scale-dtype", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-shard-size-mib", type=int, default=1024)
    parser.add_argument(
        "--calibration-file", help="path to calibration JSONL text file"
    )
    parser.add_argument(
        "--calibration-limit",
        type=_positive_int,
        help="use at most this many JSONL records",
    )
    parser.add_argument(
        "--calibration-micro-batch-size",
        type=_positive_int,
        default=1,
    )
    parser.add_argument(
        "--max-calibration-tokens",
        type=_positive_int,
        default=4096,
        help="maximum valid activation tokens captured per Linear module",
    )
    parser.add_argument(
        "--gptq-block-size",
        type=_positive_int,
        default=128,
    )
    parser.add_argument(
        "--gptq-damp-percent",
        type=_fraction,
        default=0.01,
    )
    parser.add_argument(
        "--verbose", action="store_true", help="enable verbose loss printing"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    calibration = None
    if args.algorithm == "gptq":
        calibration = {
            "gptq": {
                "block_size": args.gptq_block_size,
                "damp_percent": args.gptq_damp_percent,
            }
        }

    config = QuantizationConfig(
        algorithm=args.algorithm,
        group_size=args.group_size,
        symmetric=not args.asymmetric,
        scale_dtype=args.scale_dtype,
        calibration=calibration,
        source_checkpoint=args.model,
    )

    calibration_input_ids = None
    if args.calibration_file:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=True,
        )
        calibration_token_ids = load_token_ids(
            args.calibration_file,
            tokenizer,
            limit=args.calibration_limit,
        )
        if not calibration_token_ids:
            raise ValueError("calibration JSONL must contain at least one record")
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id for calibration")
        max_length = max(len(token_ids) for token_ids in calibration_token_ids)
        calibration_input_ids = torch.full(
            (len(calibration_token_ids), max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        for row, token_ids in enumerate(calibration_token_ids):
            calibration_input_ids[row, : len(token_ids)] = torch.tensor(token_ids)

    manifest = quantize_checkpoint(
        args.model,
        args.output,
        config,
        calibration_input_ids=calibration_input_ids,
        calibration_micro_batch_size=args.calibration_micro_batch_size,
        max_calibration_tokens=args.max_calibration_tokens,
        device=args.device,
        max_shard_size_bytes=args.max_shard_size_mib * 1024 * 1024,
        print_layer_loss=args.verbose,
    )
    print(f"quantized {len(manifest)} Linear modules into {args.output}")


if __name__ == "__main__":
    main()
