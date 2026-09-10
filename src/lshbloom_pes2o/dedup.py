from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, TextIO

import datasketch
from datasketch import MinHash, MinHashLSH, MinHashLSHBloom

from lshbloom_pes2o.perplexity import validate_record


@dataclass(frozen=True)
class DedupDecision:
    record: dict
    minhashlsh_duplicate: bool
    lshbloom_duplicate: bool


@dataclass(frozen=True)
class DedupConfig:
    expected_documents: int = 5000
    threshold: float = 0.5
    num_perm: int = 256
    seed: int = 1
    effective_fp: float = 1e-10
    training_gate_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.expected_documents < 1:
            raise ValueError("expected_documents must be positive")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.num_perm < 2:
            raise ValueError("num_perm must be at least 2")
        if not 0.0 <= self.training_gate_rate <= 1.0:
            raise ValueError("training_gate_rate must be between 0 and 1")
        per_filter_fp(self.effective_fp, 1)


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
    actual_effective_fp = -math.expm1(minhash_index.b * math.log1p(-filter_fp))
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


def iter_local_jsonl_gz(path: Path | str) -> Iterator[dict]:
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at line {line_number} in {path}"
                ) from error
            yield validate_record(record)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _gzip_text_writer(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_stream, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def _write_record(stream: TextIO, record: dict) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")


def _token_count(text: str, tokenizer: object | None) -> int | None:
    if tokenizer is None:
        return None
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    return len(tokenizer.encode(text, add_special_tokens=False)) + 1


def _variant_metadata(
    staging_dir: Path,
    relative_path: str,
    retained_ids: list[str],
    removed_ids: list[str],
    token_count: int | None,
    input_documents: int,
) -> dict:
    path = staging_dir / relative_path
    return {
        "relative_path": relative_path,
        "documents": len(retained_ids),
        "removed_documents": len(removed_ids),
        "removal_rate": len(removed_ids) / input_documents,
        "token_count": token_count,
        "retained_ids": retained_ids,
        "removed_ids": removed_ids,
        "compressed_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare_variants(
    input_path: Path | str,
    output_dir: Path | str,
    config: DedupConfig,
    tokenizer: object | None = None,
    progress_callback: Callable[[int], None] | None = None,
    progress_every: int = 100,
) -> dict:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    relative_paths = {
        "raw": "raw/train.jsonl.gz",
        "minhashlsh": "minhashlsh/train.jsonl.gz",
        "lshbloom": "lshbloom/train.jsonl.gz",
    }

    with (
        tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}-staging-", dir=output_dir.parent
        ) as staging_name,
        tempfile.TemporaryDirectory(prefix="pes2o-lshbloom-index-") as bloom_index_dir,
    ):
        staging_dir = Path(staging_name)
        minhash_index, bloom_index, index_metadata = build_indexes(
            threshold=config.threshold,
            num_perm=config.num_perm,
            expected_documents=config.expected_documents,
            effective_fp=config.effective_fp,
            bloom_save_dir=bloom_index_dir,
        )

        retained_ids = {name: [] for name in relative_paths}
        removed_ids = {name: [] for name in relative_paths}
        token_counts: dict[str, int | None] = {
            name: 0 if tokenizer is not None else None for name in relative_paths
        }
        differing_ids: list[str] = []
        input_documents = 0

        with ExitStack() as stack:
            writers = {
                name: stack.enter_context(
                    _gzip_text_writer(staging_dir / relative_path)
                )
                for name, relative_path in relative_paths.items()
            }
            decisions = stream_decisions(
                iter_local_jsonl_gz(input_path),
                minhash_index=minhash_index,
                bloom_index=bloom_index,
                num_perm=config.num_perm,
                seed=config.seed,
            )
            for input_documents, decision in enumerate(decisions, start=1):
                record = decision.record
                record_id = record["id"]
                document_tokens = _token_count(record["text"], tokenizer)

                _write_record(writers["raw"], record)
                retained_ids["raw"].append(record_id)
                if document_tokens is not None:
                    token_counts["raw"] += document_tokens

                duplicate_flags = {
                    "minhashlsh": decision.minhashlsh_duplicate,
                    "lshbloom": decision.lshbloom_duplicate,
                }
                for name, is_duplicate in duplicate_flags.items():
                    if is_duplicate:
                        removed_ids[name].append(record_id)
                    else:
                        _write_record(writers[name], record)
                        retained_ids[name].append(record_id)
                        if document_tokens is not None:
                            token_counts[name] += document_tokens

                if decision.minhashlsh_duplicate != decision.lshbloom_duplicate:
                    differing_ids.append(record_id)
                if (
                    progress_callback is not None
                    and input_documents % progress_every == 0
                ):
                    progress_callback(input_documents)

        if input_documents != config.expected_documents:
            raise ValueError(
                f"expected {config.expected_documents} documents, found {input_documents}"
            )

        variants = {
            name: _variant_metadata(
                staging_dir=staging_dir,
                relative_path=relative_paths[name],
                retained_ids=retained_ids[name],
                removed_ids=removed_ids[name],
                token_count=token_counts[name],
                input_documents=input_documents,
            )
            for name in relative_paths
        }
        maximum_removal_rate = max(
            variants["minhashlsh"]["removal_rate"],
            variants["lshbloom"]["removal_rate"],
        )
        training_gate_passes = maximum_removal_rate >= config.training_gate_rate
        tokenizer_name = None
        if tokenizer is not None:
            tokenizer_name = getattr(
                tokenizer, "name_or_path", type(tokenizer).__name__
            )

        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "path": str(input_path),
                "compressed_bytes": input_path.stat().st_size,
                "sha256": sha256_file(input_path),
                "documents": input_documents,
            },
            "parameters": {
                "datasketch_version": datasketch.__version__,
                "normalization": "lowercase_whitespace_unigram_set",
                "threshold": config.threshold,
                "num_perm": config.num_perm,
                "seed": config.seed,
                "minhash_scheme": "affine32",
                "expected_documents": config.expected_documents,
                "requested_effective_fp": config.effective_fp,
                "tokenizer": tokenizer_name,
                "eos_tokens_per_document": 1 if tokenizer is not None else None,
            },
            "index": index_metadata,
            "variants": variants,
            "agreement": {
                "same_decisions": input_documents - len(differing_ids),
                "different_decisions": len(differing_ids),
                "agreement_rate": 1.0 - len(differing_ids) / input_documents,
                "differing_ids": differing_ids,
            },
            "training_gate": {
                "minimum_removal_rate": config.training_gate_rate,
                "maximum_observed_removal_rate": maximum_removal_rate,
                "passes": training_gate_passes,
                "next_action": (
                    "train_25m_tokens"
                    if training_gate_passes
                    else "expand_to_50000_documents"
                ),
            },
            "runtime_seconds": time.perf_counter() - start,
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_dir, output_dir)

    return manifest
