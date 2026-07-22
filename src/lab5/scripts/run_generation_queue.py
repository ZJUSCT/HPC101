import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TextIO

import torch
from tqdm.auto import tqdm

from hpc101_infer import (
    EngineConfig,
    GenerationRequest,
    Runner,
    InferenceEngine,
    SamplingParams,
)

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run queued static-batch generation from a JSONL request stream."
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-synchronize-metrics",
        action="store_true",
        help="Do not synchronize the device around individual metric regions.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show request completion progress.",
    )
    return parser.parse_args()


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
    config = EngineConfig(
        dtype=DTYPES[args.dtype],
        device=args.device,
        max_batch_size=args.batch_size,
        scheduler_batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        seed=args.seed,
        synchronize_metrics=not args.no_synchronize_metrics,
    )
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
    print(
        json.dumps(
            {
                "requests": len(outputs),
                "generated_tokens": generated_tokens,
                "elapsed_s": elapsed,
                "requests_per_s": len(outputs) / elapsed if elapsed else 0.0,
                "generated_tokens_per_s": (
                    generated_tokens / elapsed if elapsed else 0.0
                ),
                "batch_size": args.batch_size,
            }
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
