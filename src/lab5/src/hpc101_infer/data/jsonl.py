from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def iter_jsonl_texts(
    path: str | Path,
    text_key: str = "text",
) -> Iterator[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"expected a JSON object in {path} at line {line_number}"
                )
            if text_key not in record:
                raise ValueError(
                    f"missing {text_key!r} in {path} at line {line_number}"
                )
            text = record[text_key]
            if not isinstance(text, str):
                raise ValueError(
                    f"expected {text_key!r} to be a string in {path} "
                    f"at line {line_number}"
                )
            yield text


def load_jsonl_texts(
    path: str | Path,
    text_key: str = "text",
    limit: int | None = None,
) -> list[str]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return []

    texts: list[str] = []
    for text in iter_jsonl_texts(path, text_key=text_key):
        texts.append(text)
        if limit is not None and len(texts) >= limit:
            break
    return texts
