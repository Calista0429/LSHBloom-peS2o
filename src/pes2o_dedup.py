from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from datasketch import MinHash, MinHashLSH, MinHashLSHBloom

from src.pes2o_perplexity import validate_record


@dataclass(frozen=True)
class DedupDecision:
    record: dict
    minhashlsh_duplicate: bool
    lshbloom_duplicate: bool


def normalize_unigrams(text: str) -> tuple[bytes, ...]:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    tokens = tuple(sorted({token.lower().encode("utf-8") for token in text.split()}))
    if not tokens:
        raise ValueError("text contains no unigrams")
    return tokens


def per_filter_fp(effective_fp: float, bands: int) -> float:
    if not 0.0 < effective_fp < 1.0:
        raise ValueError("effective_fp must be between 0 and 1")
    if bands < 1:
        raise ValueError("bands must be positive")
    return -math.expm1(math.log1p(-effective_fp) / bands)


def make_minhash(text: str, num_perm: int, seed: int) -> MinHash:
    minhash = MinHash(num_perm=num_perm, seed=seed)
    minhash.update_batch(normalize_unigrams(text))
    return minhash


def build_indexes(
    threshold: float,
    num_perm: int,
    expected_documents: int,
    effective_fp: float,
    bloom_save_dir: Path | str | None = None,
) -> tuple[MinHashLSH, MinHashLSHBloom, dict]:
    minhash_index = MinHashLSH(threshold=threshold, num_perm=num_perm)
    filter_fp = per_filter_fp(effective_fp, minhash_index.b)
    bloom_index = MinHashLSHBloom(
        threshold=threshold,
        num_perm=num_perm,
        n=expected_documents,
        fp=filter_fp,
        save_dir=str(bloom_save_dir) if bloom_save_dir is not None else None,
        params=(minhash_index.b, minhash_index.r),
    )
    actual_effective_fp = -math.expm1(
        minhash_index.b * math.log1p(-filter_fp)
    )
    metadata = {
        "bands": minhash_index.b,
        "rows_per_band": minhash_index.r,
        "used_permutations": minhash_index.b * minhash_index.r,
        "per_filter_fp": filter_fp,
        "effective_fp": actual_effective_fp,
    }
    return minhash_index, bloom_index, metadata


def stream_decisions(
    records: Iterable[dict],
    minhash_index: MinHashLSH,
    bloom_index: MinHashLSHBloom,
    num_perm: int,
    seed: int,
) -> Iterator[DedupDecision]:
    seen_ids: set[str] = set()
    for raw_record in records:
        record = validate_record(raw_record)
        record_id = record["id"]
        if record_id in seen_ids:
            raise ValueError(f"duplicate record id: {record_id}")
        seen_ids.add(record_id)

        minhash = make_minhash(record["text"], num_perm=num_perm, seed=seed)
        minhashlsh_duplicate = bool(minhash_index.query(minhash))
        if not minhashlsh_duplicate:
            minhash_index.insert(record_id, minhash)

        lshbloom_duplicate = bool(bloom_index.query(minhash))
        if not lshbloom_duplicate:
            bloom_index.insert(minhash)

        yield DedupDecision(
            record=record,
            minhashlsh_duplicate=minhashlsh_duplicate,
            lshbloom_duplicate=lshbloom_duplicate,
        )
