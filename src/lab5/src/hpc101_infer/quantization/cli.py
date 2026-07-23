from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import yaml
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


@dataclass(frozen=True)
class CalibrationInputConfig:
    path: str | None = None
    limit: int | None = None
    micro_batch_size: int = 1
    max_tokens: int = 4096

    @classmethod
    def from_yaml(cls, path: str | Path) -> CalibrationInputConfig:
        try:
            with Path(path).open() as stream:
                raw = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid YAML: {error}") from error
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a YAML mapping")
        calibration = raw.get("calibration", {})
        if calibration is None:
            return cls()
        if not isinstance(calibration, dict):
            raise ValueError("config.calibration must be a YAML mapping")

        allowed = {"path", "limit", "micro_batch_size", "max_tokens"}
        unknown = set(calibration) - allowed
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise ValueError(f"unknown config.calibration option(s): {names}")

        values: dict[str, object] = {}
        calibration_path = calibration.get("path")
        if calibration_path is not None:
            if not isinstance(calibration_path, str):
                raise ValueError("config.calibration.path must be a string")
            values["path"] = calibration_path
        for key in ("limit", "micro_batch_size", "max_tokens"):
            value = calibration.get(key)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"config.calibration.{key} must be a positive integer")
            values[key] = value
        return cls(**values)


def _option_is_present(arguments: Sequence[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def _symmetric_option_is_present(arguments: Sequence[str]) -> bool:
    return any(
        _option_is_present(arguments, option)
        for option in ("--symmetric", "--no-symmetric", "--asymmetric")
    )


def _gptq_defaults(config: QuantizationConfig) -> tuple[int, float]:
    calibration = config.calibration or {}
    gptq = calibration.get("gptq", {})
    if not isinstance(gptq, dict):
        raise ValueError("config.quantization.calibration.gptq must be a mapping")
    block_size = gptq.get("block_size", 128)
    damp_percent = gptq.get("damp_percent", 0.01)
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("GPTQ block_size must be a positive integer")
    if (
        not isinstance(damp_percent, (int, float))
        or isinstance(damp_percent, bool)
        or not 0.0 < float(damp_percent) < 1.0
    ):
        raise ValueError("GPTQ damp_percent must be in (0, 1)")
    return block_size, float(damp_percent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a W4A16 Gemma 4 checkpoint")
    parser.add_argument(
        "--config",
        help="YAML configuration path. Explicit command-line options take precedence.",
    )
    parser.add_argument("--model", required=True, help="source Hugging Face checkpoint")
    parser.add_argument("--output", required=True, help="output checkpoint directory")
    parser.add_argument(
        "--algorithm", choices=available_quantization_methods(), default="rtn"
    )
    parser.add_argument("--group-size", type=int, choices=(64, 128), default=128)
    parser.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use symmetric weight quantization.",
    )
    parser.add_argument(
        "--asymmetric",
        dest="symmetric",
        action="store_false",
        help="Alias for --no-symmetric.",
    )
    parser.add_argument(
        "--scale-dtype", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--device", default="cuda")
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
    return parser


def _quantization_overrides(
    args: argparse.Namespace,
    command_line_arguments: Sequence[str],
) -> dict[str, object]:
    overrides: dict[str, object] = {"source_checkpoint": args.model}
    if _option_is_present(command_line_arguments, "--algorithm"):
        overrides["algorithm"] = args.algorithm
    if _option_is_present(command_line_arguments, "--group-size"):
        overrides["group_size"] = args.group_size
    if _symmetric_option_is_present(command_line_arguments):
        overrides["symmetric"] = args.symmetric
    if _option_is_present(command_line_arguments, "--scale-dtype"):
        overrides["scale_dtype"] = args.scale_dtype
    return overrides


def _merge_quantization_config(
    base_config: QuantizationConfig,
    args: argparse.Namespace,
    command_line_arguments: Sequence[str],
) -> QuantizationConfig:
    config = replace(
        base_config,
        **_quantization_overrides(args, command_line_arguments),
    )
    if config.algorithm != "gptq":
        return config

    calibration = dict(config.calibration or {})
    gptq = dict(calibration.get("gptq", {}))
    gptq["block_size"] = args.gptq_block_size
    gptq["damp_percent"] = args.gptq_damp_percent
    calibration["gptq"] = gptq
    return replace(config, calibration=calibration)


def _set_quantization_argument_values(
    args: argparse.Namespace,
    config: QuantizationConfig,
) -> None:
    args.algorithm = config.algorithm
    args.group_size = config.group_size
    args.symmetric = config.symmetric
    args.asymmetric = not config.symmetric
    args.scale_dtype = config.scale_dtype


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    command_line_arguments = list(sys.argv[1:] if argv is None else argv)
    config_arguments = command_line_arguments

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args(config_arguments)

    quantization_config = QuantizationConfig()
    calibration_config = CalibrationInputConfig()
    parser = build_parser()
    if config_args.config is not None:
        try:
            quantization_config = QuantizationConfig.from_yaml(config_args.config)
            calibration_config = CalibrationInputConfig.from_yaml(config_args.config)
        except (OSError, ValueError) as error:
            parser.error(f"cannot load --config {config_args.config!r}: {error}")

    try:
        gptq_block_size, gptq_damp_percent = _gptq_defaults(quantization_config)
    except ValueError as error:
        parser.error(str(error))
    parser.set_defaults(
        algorithm=quantization_config.algorithm,
        group_size=quantization_config.group_size,
        symmetric=quantization_config.symmetric,
        scale_dtype=quantization_config.scale_dtype,
        gptq_block_size=gptq_block_size,
        gptq_damp_percent=gptq_damp_percent,
        calibration_file=calibration_config.path,
        calibration_limit=calibration_config.limit,
        calibration_micro_batch_size=calibration_config.micro_batch_size,
        max_calibration_tokens=calibration_config.max_tokens,
    )
    args = parser.parse_args(command_line_arguments)
    try:
        quantization_config = _merge_quantization_config(
            quantization_config,
            args,
            config_arguments,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    args.quantization_config = quantization_config
    _set_quantization_argument_values(args, quantization_config)
    return args


def main() -> None:
    args = parse_args()
    config = args.quantization_config

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
