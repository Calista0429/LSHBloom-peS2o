from __future__ import annotations

import gzip
import io
import json
import math
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator


SUPPORTED_SOURCES = {"s2orc", "s2ag"}


def validate_record(record: dict, expected_source: str | None = None) -> dict:
    if not isinstance(record, dict):
        raise ValueError("each record must be a JSON object")
    for key in ("id", "source", "text"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"record field {key!r} must be a non-empty string")
    if record["source"] not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported source {record['source']}")
    if expected_source is not None and record["source"] != expected_source:
        raise ValueError(
            f"expected source {expected_source}, received {record['source']}"
        )
    return record


def iter_jsonl_gz(url: str) -> Iterator[dict]:
    with urllib.request.urlopen(url, timeout=300) as response:
        with gzip.GzipFile(fileobj=response) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text_stream:
                for line_number, line in enumerate(text_stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid JSON at line {line_number} from {url}"
                        ) from error
                    yield validate_record(record)


def collect_source_records(
    urls: Iterable[str], limits: dict[str, int | None]
) -> dict[str, list[dict]]:
    unknown_sources = set(limits) - SUPPORTED_SOURCES
    if unknown_sources:
        raise ValueError(f"unsupported requested sources: {sorted(unknown_sources)}")
    for source, limit in limits.items():
        if limit is not None and limit < 1:
            raise ValueError(f"limit for {source} must be positive or None")

    records = {source: [] for source in limits}

    def all_finite_limits_reached() -> bool:
        return all(
            limit is not None and len(records[source]) >= limit
            for source, limit in limits.items()
        )

    for url in urls:
        for record in iter_jsonl_gz(url):
            source = record["source"]
            if source not in records:
                continue
            limit = limits[source]
            if limit is None or len(records[source]) < limit:
                records[source].append(record)
            if all_finite_limits_reached():
                return records

    short = {
        source: {"expected": limit, "actual": len(records[source])}
        for source, limit in limits.items()
        if limit is not None and len(records[source]) < limit
    }
    if short:
        raise ValueError(f"validation streams ended before limits were met: {short}")
    return records


def iter_packed_sequences(
    records: Iterable[dict], tokenizer: Any, sequence_length: int
) -> Iterator[dict]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    token_buffer: list[int] = []
    origin_buffer: list[str] = []

    for raw_record in records:
        record = validate_record(raw_record)
        document_id = record["id"]
        token_ids = tokenizer.encode(record["text"], add_special_tokens=False)
        token_ids.append(eos_token_id)

        position = 0
        while position < len(token_ids):
            space = sequence_length - len(token_buffer)
            next_position = min(position + space, len(token_ids))
            piece = token_ids[position:next_position]
            token_buffer.extend(piece)
            origin_buffer.extend([document_id] * len(piece))
            position = next_position

            if len(token_buffer) == sequence_length:
                yield {
                    "input_ids": token_buffer,
                    "labels": token_buffer.copy(),
                    "document_ids": list(dict.fromkeys(origin_buffer)),
                }
                token_buffer = []
                origin_buffer = []

    if token_buffer:
        padding = sequence_length - len(token_buffer)
        yield {
            "input_ids": token_buffer + [eos_token_id] * padding,
            "labels": token_buffer + [-100] * padding,
            "document_ids": list(dict.fromkeys(origin_buffer)),
        }


def perplexity_from_loss(mean_loss: float) -> float:
    if not math.isfinite(mean_loss):
        raise ValueError("mean loss must be finite")
    try:
        return math.exp(mean_loss)
    except OverflowError:
        return float("inf")


def combine_source_metrics(metrics: Iterable[dict]) -> dict:
    metrics = list(metrics)
    total_tokens = sum(int(item["predicted_tokens"]) for item in metrics)
    if total_tokens <= 0:
        raise ValueError("predicted token total must be positive")
    total_nll = sum(float(item["negative_log_likelihood"]) for item in metrics)
    loss = total_nll / total_tokens
    return {
        "negative_log_likelihood": total_nll,
        "predicted_tokens": total_tokens,
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
    }


def write_result_json(path: str | Path, result: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return output_path
