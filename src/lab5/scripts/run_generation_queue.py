import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Sequence, TextIO

from tqdm.auto import tqdm

from hpc101_infer import (
    EngineConfig,
    GenerationRequest,
    InferenceEngine,
    Runner,
    SamplingParams,
)
from hpc101_infer.config import DTYPES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run queued static-batch generation from a JSONL request stream."
    )
    parser.add_argument(
        "--config",
        help="YAML configuration path. Explicit command-line options take precedence.",
    )
    parser.add_argument("--model", required=True, help="Model checkpoint directory.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path, or - to read from stdin.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output JSONL path, or - to write to stdout.",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional path for the aggregate summary JSON.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help="KV cache batch capacity. Defaults to --batch-size.",
    )
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--attention-backend", choices=("eager",), default="eager")
    parser.add_argument(
        "--linear-backend",
        choices=("bf16", "int4_reference"),
        default="bf16",
    )
    parser.add_argument(
        "--scheduler-backend",
        choices=("static_batch",),
        default="static_batch",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--synchronize-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize the device around individual metric regions.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show request completion progress.",
    )
    return parser


def _option_is_present(arguments: Sequence[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def _dtype_name(config: EngineConfig) -> str:
    return next(name for name, dtype in DTYPES.items() if dtype == config.dtype)


def _boolean_option_is_present(arguments: Sequence[str], option: str) -> bool:
    negative_option = f"--no-{option.removeprefix('--')}"
    return _option_is_present(arguments, option) or _option_is_present(
        arguments, negative_option
    )


def _engine_overrides(
    args: argparse.Namespace,
    command_line_arguments: Sequence[str],
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if _option_is_present(command_line_arguments, "--device"):
        overrides["device"] = args.device
    if _option_is_present(command_line_arguments, "--dtype"):
        overrides["dtype"] = DTYPES[args.dtype]
    if _option_is_present(command_line_arguments, "--batch-size"):
        overrides["scheduler_batch_size"] = args.batch_size
        if not _option_is_present(command_line_arguments, "--max-batch-size"):
            overrides["max_batch_size"] = args.batch_size
    if _option_is_present(command_line_arguments, "--max-batch-size"):
        overrides["max_batch_size"] = args.max_batch_size
    if _option_is_present(command_line_arguments, "--max-sequence-length"):
        overrides["max_sequence_length"] = args.max_sequence_length
    if _option_is_present(command_line_arguments, "--attention-backend"):
        overrides["attention_backend"] = args.attention_backend
    if _option_is_present(command_line_arguments, "--linear-backend"):
        overrides["linear_backend"] = args.linear_backend
    if _option_is_present(command_line_arguments, "--scheduler-backend"):
        overrides["scheduler_backend"] = args.scheduler_backend
    if _option_is_present(command_line_arguments, "--seed"):
        overrides["seed"] = args.seed
    if _boolean_option_is_present(command_line_arguments, "--synchronize-metrics"):
        overrides["synchronize_metrics"] = args.synchronize_metrics
    return overrides


def _set_engine_argument_values(
    args: argparse.Namespace,
    config: EngineConfig,
) -> None:
    args.device = config.device
    args.dtype = _dtype_name(config)
    args.batch_size = config.scheduler_batch_size
    args.max_batch_size = config.max_batch_size
    args.max_sequence_length = config.max_sequence_length
    args.attention_backend = config.attention_backend
    args.linear_backend = config.linear_backend
    args.scheduler_backend = config.scheduler_backend
    args.seed = config.seed
    args.synchronize_metrics = config.synchronize_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    command_line_arguments = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args(command_line_arguments)

    engine_config = EngineConfig()
    parser = build_parser()
    if config_args.config is not None:
        try:
            engine_config = EngineConfig.from_yaml(config_args.config)
        except (OSError, ValueError) as error:
            parser.error(f"cannot load --config {config_args.config!r}: {error}")

    parser.set_defaults(
        device=engine_config.device,
        dtype=_dtype_name(engine_config),
        batch_size=engine_config.scheduler_batch_size,
        max_batch_size=engine_config.max_batch_size,
        max_sequence_length=engine_config.max_sequence_length,
        attention_backend=engine_config.attention_backend,
        linear_backend=engine_config.linear_backend,
        scheduler_backend=engine_config.scheduler_backend,
        seed=engine_config.seed,
        synchronize_metrics=engine_config.synchronize_metrics,
    )
    args = parser.parse_args(command_line_arguments)
    try:
        engine_config = replace(
            engine_config,
            **_engine_overrides(args, command_line_arguments),
        )
    except ValueError as error:
        parser.error(str(error))
    args.engine_config = engine_config
    _set_engine_argument_values(args, engine_config)

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    return args


def open_input(path: str) -> TextIO:
    if path == "-":
        return sys.stdin
    return Path(path).open()


def open_output(path: str) -> TextIO:
    if path == "-":
        return sys.stdout
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("w")


def write_summary(path: str, summary: dict[str, object]) -> None:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def count_requests(path: str) -> int | None:
    if path == "-":
        return None
    with Path(path).open() as stream:
        return sum(1 for line in stream if line.strip())


def parse_request(
    record: object,
    *,
    default_max_new_tokens: int,
) -> GenerationRequest:
    if not isinstance(record, dict):
        raise ValueError("each request must be a JSON object")
    sampling_record = record.get("sampling_params", {})
    if not isinstance(sampling_record, dict):
        raise ValueError("sampling_params must be a JSON object")
    stop_token_ids = record.get("stop_token_ids")
    if stop_token_ids is not None and not isinstance(stop_token_ids, (list, tuple)):
        raise ValueError("stop_token_ids must be a JSON array")
    return GenerationRequest(
        prompt=record.get("prompt"),
        input_ids=record.get("input_ids"),
        max_new_tokens=record.get("max_new_tokens", default_max_new_tokens),
        stop_token_ids=None if stop_token_ids is None else tuple(stop_token_ids),
        sampling_params=SamplingParams(**sampling_record),
    )


def iter_requests(
    stream: TextIO,
    *,
    default_max_new_tokens: int,
):
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            yield parse_request(
                record,
                default_max_new_tokens=default_max_new_tokens,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid request on line {line_number}: {error}"
            ) from error


def main() -> None:
    args = parse_args()
    config = args.engine_config
    engine = InferenceEngine.from_pretrained(args.model, config)
    runner = Runner(engine)

    input_stream = open_input(args.input)
    output_stream = open_output(args.output)
    try:
        total_requests = count_requests(args.input) if args.progress else None
        started = perf_counter()
        with tqdm(
            total=total_requests,
            desc="generation",
            unit="request",
            disable=not args.progress,
            dynamic_ncols=True,
        ) as progress:
            outputs = runner.run(
                iter_requests(
                    input_stream,
                    default_max_new_tokens=args.max_new_tokens,
                ),
                on_completed=progress.update,
            )
        elapsed = perf_counter() - started
        for request_id, output in enumerate(outputs):
            output_stream.write(
                json.dumps(
                    {"request_id": request_id, **asdict(output)},
                    ensure_ascii=False,
                )
                + "\n"
            )
        output_stream.flush()
    finally:
        if input_stream is not sys.stdin:
            input_stream.close()
        if output_stream is not sys.stdout:
            output_stream.close()

    generated_tokens = sum(output.generated_tokens for output in outputs)
    summary = {
        "requests": len(outputs),
        "generated_tokens": generated_tokens,
        "elapsed_s": elapsed,
        "requests_per_s": len(outputs) / elapsed if elapsed else 0.0,
        "generated_tokens_per_s": generated_tokens / elapsed if elapsed else 0.0,
        "batch_size": args.batch_size,
    }
    print(json.dumps(summary), file=sys.stderr)
    if args.summary_output is not None:
        write_summary(args.summary_output, summary)


if __name__ == "__main__":
    main()
