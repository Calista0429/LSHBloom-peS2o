from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

VARIANTS = ("raw", "minhashlsh", "lshbloom")
STAGE_SAMPLE_COUNTS = {"smoke": 10, "full": 1000}
METRIC_KEYS = {
    "acc": "acc,none",
    "acc_stderr": "acc_stderr,none",
    "acc_norm": "acc_norm,none",
    "acc_norm_stderr": "acc_norm_stderr,none",
}


def _finite_float(value: object, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def extract_sciq_metrics(
    result: dict,
    variant: str,
    stage: str,
    elapsed_seconds: float,
    peak_cuda_bytes: int,
) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    if stage not in STAGE_SAMPLE_COUNTS:
        raise ValueError(f"unsupported stage: {stage}")

    try:
        metrics = result["results"]["sciq"]
        effective = int(result["n-samples"]["sciq"]["effective"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid SciQ harness result structure") from error

    expected = STAGE_SAMPLE_COUNTS[stage]
    if effective != expected:
        raise ValueError(
            f"{stage} SciQ evaluation expected {expected} examples, received {effective}"
        )

    extracted = {}
    for output_name, harness_name in METRIC_KEYS.items():
        if harness_name not in metrics:
            raise ValueError(f"missing SciQ metric: {harness_name}")
        extracted[output_name] = _finite_float(metrics[harness_name], harness_name)

    elapsed = _finite_float(elapsed_seconds, "elapsed_seconds")
    if elapsed <= 0:
        raise ValueError("elapsed_seconds must be positive")
    try:
        peak_memory = int(peak_cuda_bytes)
    except (TypeError, ValueError) as error:
        raise ValueError("peak_cuda_bytes must be an integer") from error
    if peak_memory < 0:
        raise ValueError("peak_cuda_bytes must be non-negative")

    return {
        "variant": variant,
        "stage": stage,
        "examples": effective,
        **extracted,
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": peak_memory,
    }


def build_comparison(rows: list[dict]) -> list[dict]:
    indexed = {}
    for row in rows:
        variant = row.get("variant")
        if variant in indexed:
            raise ValueError(f"duplicate variant: {variant}")
        indexed[variant] = row

    missing = [variant for variant in VARIANTS if variant not in indexed]
    if missing:
        raise ValueError(f"missing variants: {', '.join(missing)}")

    for variant in VARIANTS:
        if indexed[variant].get("stage") != "full":
            raise ValueError("comparison requires full-stage rows")

    raw = indexed["raw"]
    comparison = []
    for variant in VARIANTS:
        row = dict(indexed[variant])
        row["delta_acc"] = float(row["acc"]) - float(raw["acc"])
        row["delta_acc_norm"] = float(row["acc_norm"]) - float(raw["acc_norm"])
        comparison.append(row)
    return comparison


def to_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=repr)]

    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return to_jsonable(converted)

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if converted is not value:
            return to_jsonable(converted)

    return str(value)


def write_json(path: str | Path, value: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path
