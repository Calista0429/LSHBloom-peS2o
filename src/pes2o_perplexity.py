from __future__ import annotations

import gzip
import io
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


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


def count_shifted_targets(labels: Any) -> int:
    return int(labels[:, 1:].ne(-100).sum().item())


def _iter_batches(items: Iterable[dict], batch_size: int) -> Iterator[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch: list[dict] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def evaluate_source(
    model: Any,
    packed_sequences: Iterable[dict],
    source: str,
    batch_size: int,
    device: str,
    log_every_steps: int,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    import torch

    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported source {source}")
    if log_every_steps < 1:
        raise ValueError("log_every_steps must be positive")

    start = time.perf_counter()
    total_nll = 0.0
    predicted_tokens = 0
    input_tokens = 0
    sequence_count = 0
    batch_count = 0
    document_ids: list[str] = []
    seen_document_ids: set[str] = set()
    last_logged_batch = 0

    try:
        with torch.inference_mode():
            for batch_count, batch in enumerate(
                _iter_batches(packed_sequences, batch_size), start=1
            ):
                input_ids = torch.tensor(
                    [item["input_ids"] for item in batch],
                    dtype=torch.long,
                    device=device,
                )
                labels = torch.tensor(
                    [item["labels"] for item in batch],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = labels.ne(-100).long()
                batch_targets = count_shifted_targets(labels)
                if batch_targets < 1:
                    raise ValueError("a packed batch contained no prediction targets")

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                batch_loss = float(outputs.loss.item())
                if not math.isfinite(batch_loss):
                    raise ValueError(
                        f"non-finite loss for {source} at batch {batch_count}"
                    )

                total_nll += batch_loss * batch_targets
                predicted_tokens += batch_targets
                input_tokens += int(labels.ne(-100).sum().item())
                sequence_count += len(batch)
                for item in batch:
                    for document_id in item["document_ids"]:
                        if document_id not in seen_document_ids:
                            seen_document_ids.add(document_id)
                            document_ids.append(document_id)

                if progress_callback is not None and (
                    batch_count == 1 or batch_count % log_every_steps == 0
                ):
                    elapsed = max(time.perf_counter() - start, 1e-9)
                    running_loss = total_nll / predicted_tokens
                    progress_callback(
                        {
                            "source": source,
                            "batch": batch_count,
                            "sequences": sequence_count,
                            "predicted_tokens": predicted_tokens,
                            "loss": running_loss,
                            "perplexity": perplexity_from_loss(running_loss),
                            "tokens_per_second": predicted_tokens / elapsed,
                            "elapsed_seconds": elapsed,
                            "gpu_memory_gb": torch.cuda.memory_allocated() / (1024**3),
                        }
                    )
                    last_logged_batch = batch_count
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            f"CUDA ran out of memory with batch_size={batch_size}; "
            "reduce CONFIG['batch_size'] and rerun"
        ) from error

    if predicted_tokens < 1:
        raise ValueError(f"no prediction targets were evaluated for {source}")

    elapsed = max(time.perf_counter() - start, 1e-9)
    loss = total_nll / predicted_tokens
    result = {
        "source": source,
        "documents": len(document_ids),
        "document_ids": document_ids,
        "sequences": sequence_count,
        "batches": batch_count,
        "input_tokens": input_tokens,
        "predicted_tokens": predicted_tokens,
        "negative_log_likelihood": total_nll,
        "loss": loss,
        "perplexity": perplexity_from_loss(loss),
        "elapsed_seconds": elapsed,
        "tokens_per_second": predicted_tokens / elapsed,
    }

    if progress_callback is not None and last_logged_batch != batch_count:
        progress_callback({**result, "batch": batch_count})
    return result


def write_result_json(path: str | Path, result: dict) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return output_path
