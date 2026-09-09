from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np


def _iter_records(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at line {line_number} in {path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"record at line {line_number} must be an object")
            for field in ("id", "text"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValueError(
                        f"record field {field!r} at line {line_number} must be non-empty"
                    )
            yield record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pack_jsonl_gz_to_memmap(
    input_path: Path | str,
    output_path: Path | str,
    tokenizer: object,
    sequence_length: int,
    sequence_count: int,
) -> dict:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if sequence_count < 1:
        raise ValueError("sequence_count must be positive")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    required_tokens = sequence_length * sequence_count
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    tokens_written = 0
    documents_read = 0
    last_document_id = None
    last_document_tokens_used = 0
    last_document_complete = False

    try:
        packed = np.memmap(
            temporary_path, mode="w+", dtype=np.uint32, shape=(required_tokens,)
        )
        for record in _iter_records(input_path):
            token_ids = tokenizer.encode(record["text"], add_special_tokens=False)
            token_ids.append(eos_token_id)
            if token_ids and (min(token_ids) < 0 or max(token_ids) > np.iinfo(np.uint32).max):
                raise ValueError(f"token id outside uint32 range in document {record['id']}")

            documents_read += 1
            last_document_id = record["id"]
            remaining = required_tokens - tokens_written
            last_document_tokens_used = min(remaining, len(token_ids))
            end = tokens_written + last_document_tokens_used
            packed[tokens_written:end] = token_ids[:last_document_tokens_used]
            tokens_written = end
            last_document_complete = last_document_tokens_used == len(token_ids)
            if tokens_written == required_tokens:
                break

        packed.flush()
        del packed
        if tokens_written != required_tokens:
            raise ValueError(
                f"packing requires {required_tokens} tokens, found {tokens_written}"
            )
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dtype": "uint32",
        "sequence_length": sequence_length,
        "sequence_count": sequence_count,
        "input_tokens": required_tokens,
        "documents_read": documents_read,
        "last_document_id": last_document_id,
        "last_document_tokens_used": last_document_tokens_used,
        "last_document_complete": last_document_complete,
        "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
    }


class TokenMemmapDataset:
    def __init__(
        self, path: Path | str, sequence_length: int, sequence_count: int
    ) -> None:
        self.path = Path(path).resolve()
        self.sequence_length = sequence_length
        self.sequence_count = sequence_count
        expected_bytes = sequence_length * sequence_count * np.dtype(np.uint32).itemsize
        if self.path.stat().st_size != expected_bytes:
            raise ValueError(
                f"memmap size is {self.path.stat().st_size} bytes, expected {expected_bytes}"
            )
        self._tokens = np.memmap(
            self.path,
            mode="r",
            dtype=np.uint32,
            shape=(sequence_count, sequence_length),
        )

    def __len__(self) -> int:
        return self.sequence_count

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += self.sequence_count
        if index < 0 or index >= self.sequence_count:
            raise IndexError(index)
        return {"input_ids": self._tokens[index].astype(np.int64)}
