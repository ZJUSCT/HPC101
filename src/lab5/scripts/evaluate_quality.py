from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from hpc101_infer import EngineConfig, InferenceEngine
from hpc101_infer.evaluation import evaluate_nll, load_quality_token_ids

DEFAULT_BUCKETS = (128, 256, 512, 1024, 2048)
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference(path: Path) -> dict[str, object]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reference, dict):
        raise ValueError("reference must contain a JSON object")
    return reference


def _reference_mean_nll(reference: Mapping[str, object]) -> float:
    if "mean_nll" in reference:
        return float(reference["mean_nll"])
    if "nll_sum" in reference and "token_count" in reference:
        return float(reference["nll_sum"]) / int(reference["token_count"])
    raise ValueError("reference must contain mean_nll or nll_sum and token_count")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("bucket lengths must be positive")
    return parsed


def _normalize_buckets(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError("at least one bucket length is required")
    if len(set(values)) != len(values):
        raise ValueError("bucket lengths must not contain duplicates")
    return tuple(sorted(values))


def _group_token_sequences(
    token_sequences: Sequence[Sequence[int]],
    buckets: Sequence[int],
    *,
    max_sequence_length: int,
) -> dict[int, list[Sequence[int]]]:
    allowed = set(buckets)
    grouped: dict[int, list[Sequence[int]]] = defaultdict(list)
    for sequence_number, token_ids in enumerate(token_sequences, start=1):
        sequence_length = len(token_ids)
        if sequence_length > max_sequence_length:
            raise ValueError(
                f"quality sequence {sequence_number} has {sequence_length} tokens, "
                f"exceeding --max-sequence-length={max_sequence_length}"
            )
        if sequence_length not in allowed:
            raise ValueError(
                f"quality sequence {sequence_number} has unexpected length "
                f"{sequence_length}; configured buckets are {tuple(buckets)}"
            )
        grouped[sequence_length].append(token_ids)
    if not grouped:
        raise ValueError("quality dataset must contain at least one sequence")
    return dict(sorted(grouped.items()))


def _evaluate_bucketed_nll(
    engine: InferenceEngine,
    grouped_sequences: Mapping[int, Sequence[Sequence[int]]],
    *,
    chunk_size: int,
    show_progress: bool = False,
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    by_bucket: dict[str, dict[str, float | int]] = {}
    nll_sum = 0.0
    token_count = 0
    sequence_count = 0

    total_sequences = sum(len(sequences) for sequences in grouped_sequences.values())
    with tqdm(
        total=total_sequences,
        desc="quality evaluation",
        unit="record",
        disable=not show_progress,
        dynamic_ncols=True,
    ) as progress:

        def track_progress(sequences: Sequence[Sequence[int]]):
            for token_ids in sequences:
                yield token_ids
                progress.update(1)

        for bucket, sequences in sorted(grouped_sequences.items()):
            progress.set_postfix(bucket=bucket)
            bucket_result = evaluate_nll(
                engine,
                track_progress(sequences),
                chunk_size=chunk_size,
            )
            by_bucket[str(bucket)] = bucket_result.to_dict()
            nll_sum += bucket_result.nll_sum
            token_count += bucket_result.token_count
            sequence_count += bucket_result.sequence_count

    mean_nll = nll_sum / token_count
    global_result: dict[str, float | int] = {
        "nll_sum": nll_sum,
        "token_count": token_count,
        "sequence_count": sequence_count,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }
    return global_result, by_bucket


def _bucket_mapping(result: Mapping[str, object], name: str) -> Mapping[str, object]:
    by_bucket = result.get("by_bucket")
    if not isinstance(by_bucket, Mapping):
        raise ValueError(f"{name} must contain a by_bucket object")
    return by_bucket


def _validate_reference(
    reference: Mapping[str, object],
    result: Mapping[str, object],
) -> None:
    for key in (
        "dataset_sha256",
        "chunk_size",
        "token_count",
        "sequence_count",
        "bucket_lengths",
    ):
        if key in reference and reference[key] != result[key]:
            raise ValueError(
                f"reference {key}={reference[key]!r} does not match "
                f"current result {result[key]!r}"
            )

    reference_buckets = _bucket_mapping(reference, "reference")
    result_buckets = _bucket_mapping(result, "current result")
    if set(reference_buckets) != set(result_buckets):
        raise ValueError(
            f"reference buckets {sorted(reference_buckets)} do not match "
            f"current buckets {sorted(result_buckets)}"
        )

    for bucket, current_value in result_buckets.items():
        reference_value = reference_buckets[bucket]
        if not isinstance(reference_value, Mapping) or not isinstance(
            current_value, Mapping
        ):
            raise ValueError(f"bucket {bucket} must contain result objects")
        for key in ("token_count", "sequence_count"):
            if reference_value.get(key) != current_value.get(key):
                raise ValueError(
                    f"reference bucket {bucket} {key}="
                    f"{reference_value.get(key)!r} does not match current "
                    f"{current_value.get(key)!r}"
                )


def _add_reference_metrics(
    result: dict[str, object],
    reference: Mapping[str, object],
    *,
    reference_path: Path,
) -> None:
    reference_mean_nll = _reference_mean_nll(reference)
    delta_nll = float(result["mean_nll"]) - reference_mean_nll
    result.update(
        {
            "reference": str(reference_path),
            "reference_mean_nll": reference_mean_nll,
            "delta_nll": delta_nll,
            "perplexity_ratio": math.exp(delta_nll),
        }
    )

    reference_buckets = _bucket_mapping(reference, "reference")
    result_buckets = _bucket_mapping(result, "current result")
    for bucket, current_value in result_buckets.items():
        reference_value = reference_buckets[bucket]
        if not isinstance(reference_value, Mapping) or not isinstance(
            current_value, dict
        ):
            raise ValueError(f"bucket {bucket} must contain result objects")
        bucket_reference_mean_nll = _reference_mean_nll(reference_value)
        bucket_delta_nll = float(current_value["mean_nll"]) - bucket_reference_mean_nll
        current_value.update(
            {
                "reference_mean_nll": bucket_reference_mean_nll,
                "delta_nll": bucket_delta_nll,
                "perplexity_ratio": math.exp(bucket_delta_nll),
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate teacher-forcing NLL through InferenceEngine."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--max-delta-nll", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument(
        "--linear-backend",
        choices=("bf16", "int4_reference"),
        default="int4_reference",
    )
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--buckets",
        nargs="+",
        type=_positive_int,
        default=DEFAULT_BUCKETS,
        metavar="LENGTH",
        help="accepted token sequence lengths",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--text-key", default="text")
    parser.add_argument(
        "--add-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show record-level evaluation progress",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_delta_nll is not None and args.reference is None:
        parser.error("--max-delta-nll requires --reference")
    try:
        buckets = _normalize_buckets(args.buckets)
    except ValueError as error:
        parser.error(str(error))
    if buckets[-1] > args.max_sequence_length:
        parser.error(
            f"largest bucket {buckets[-1]} exceeds "
            f"--max-sequence-length={args.max_sequence_length}"
        )

    engine = InferenceEngine.from_pretrained(
        str(args.model),
        EngineConfig(
            dtype=DTYPES[args.dtype],
            device=args.device,
            max_batch_size=1,
            scheduler_batch_size=1,
            max_sequence_length=args.max_sequence_length,
            linear_backend=args.linear_backend,
        ),
    )
    token_sequences = load_quality_token_ids(
        args.dataset,
        engine.tokenizer,
        text_key=args.text_key,
        limit=args.limit,
        add_special_tokens=args.add_special_tokens,
    )
    grouped_sequences = _group_token_sequences(
        token_sequences,
        buckets,
        max_sequence_length=args.max_sequence_length,
    )
    global_result, by_bucket = _evaluate_bucketed_nll(
        engine,
        grouped_sequences,
        chunk_size=args.chunk_size,
        show_progress=args.progress,
    )

    result: dict[str, Any] = {
        "schema_version": 2,
        "model": str(args.model),
        "linear_backend": args.linear_backend,
        "dataset": str(args.dataset),
        "dataset_sha256": _dataset_sha256(args.dataset),
        "chunk_size": args.chunk_size,
        "bucket_lengths": list(buckets),
        **global_result,
        "by_bucket": by_bucket,
    }
    if args.reference is not None:
        reference = _load_reference(args.reference)
        _validate_reference(reference, result)
        _add_reference_metrics(
            result,
            reference,
            reference_path=args.reference,
        )
        if args.max_delta_nll is not None:
            result["max_delta_nll"] = args.max_delta_nll
            result["passed"] = result["delta_nll"] <= args.max_delta_nll

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
