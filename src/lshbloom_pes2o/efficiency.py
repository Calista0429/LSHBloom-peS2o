from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path


VARIANTS = ("raw", "minhashlsh", "lshbloom")
CURVE_COLUMNS = (
    "variant",
    "global_step",
    "cumulative_train_tokens",
    "cumulative_train_gpu_hours",
    "validation_loss",
    "validation_perplexity",
    "sciq_acc",
    "sciq_acc_norm",
    "sciq_acc_stderr",
    "sciq_acc_norm_stderr",
    "sciq_examples",
)


def canonical_sha256(value: object) -> str:
    """Return a stable SHA-256 for JSON-compatible experiment metadata."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value must contain only finite JSON-compatible data") from error
    return hashlib.sha256(encoded).hexdigest()


def validate_shared_experiment_results(
    results: Iterable[Mapping[str, object]],
    expected_variants: Iterable[str] = VARIANTS,
) -> str:
    """Verify that separately produced variant results belong to one experiment."""
    results = [dict(result) for result in results]
    expected_variants = tuple(expected_variants)
    if len(results) != len(expected_variants):
        raise ValueError("one experiment result is required for every variant")

    variants = [result.get("variant") for result in results]
    if set(variants) != set(expected_variants) or len(set(variants)) != len(variants):
        raise ValueError("experiment result variants do not match the expected variants")

    verified = []
    for result in results:
        if result.get("schema_version") != 2:
            raise ValueError("experiment results require schema_version 2")
        identity = result.get("experiment_identity")
        fingerprint = result.get("experiment_fingerprint")
        if not isinstance(identity, Mapping) or not isinstance(fingerprint, str):
            raise ValueError("experiment identity and fingerprint are required")
        computed = canonical_sha256(identity)
        if computed != fingerprint:
            raise ValueError(
                f"experiment fingerprint is invalid for variant {result.get('variant')}"
            )
        verified.append(fingerprint)

    if len(set(verified)) != 1:
        raise ValueError("experiment fingerprints differ across variants")
    return verified[0]


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if converted < 1 or converted != value:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _finite_number(value: object, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def build_full_training_plan(
    token_counts: Mapping[str, int], sequence_length: int
) -> dict[str, dict[str, int]]:
    sequence_length = _positive_integer(sequence_length, "sequence_length")
    if not token_counts:
        raise ValueError("token_counts must not be empty")

    plan = {}
    for variant, raw_count in token_counts.items():
        available_tokens = _positive_integer(
            raw_count, f"available tokens for {variant}"
        )
        sequence_count = available_tokens // sequence_length
        if sequence_count < 1:
            raise ValueError(
                f"{variant} has no complete sequence of length {sequence_length}"
            )
        train_input_tokens = sequence_count * sequence_length
        plan[str(variant)] = {
            "available_tokens": available_tokens,
            "sequence_count": sequence_count,
            "train_input_tokens": train_input_tokens,
            "unused_tail_tokens": available_tokens - train_input_tokens,
        }
    return plan


def checkpoint_steps(total_optimizer_steps: int, save_steps: int) -> list[int]:
    total_optimizer_steps = _positive_integer(
        total_optimizer_steps, "total_optimizer_steps"
    )
    save_steps = _positive_integer(save_steps, "save_steps")
    steps = [0]
    steps.extend(range(save_steps, total_optimizer_steps + 1, save_steps))
    if steps[-1] != total_optimizer_steps:
        steps.append(total_optimizer_steps)
    return steps


def tokens_at_step(
    global_step: int,
    tokens_per_optimizer_step: int,
    total_train_tokens: int,
) -> int:
    if isinstance(global_step, bool):
        raise ValueError("global_step must be a non-negative integer")
    try:
        global_step = int(global_step)
    except (TypeError, ValueError) as error:
        raise ValueError("global_step must be a non-negative integer") from error
    if global_step < 0:
        raise ValueError("global_step must be a non-negative integer")
    tokens_per_optimizer_step = _positive_integer(
        tokens_per_optimizer_step, "tokens_per_optimizer_step"
    )
    total_train_tokens = _positive_integer(total_train_tokens, "total_train_tokens")
    return min(global_step * tokens_per_optimizer_step, total_train_tokens)


def _index_measurements(rows: Iterable[dict], label: str) -> dict[tuple[str, int], dict]:
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"each {label} measurement must be an object")
        variant = row.get("variant")
        if not isinstance(variant, str) or not variant:
            raise ValueError(f"each {label} measurement requires variant")
        step = row.get("global_step")
        if isinstance(step, bool):
            raise ValueError(f"each {label} measurement requires integer global_step")
        try:
            step = int(step)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"each {label} measurement requires integer global_step"
            ) from error
        if step < 0:
            raise ValueError(f"each {label} measurement requires non-negative global_step")
        key = (variant, step)
        if key in indexed:
            raise ValueError(f"duplicate {label} measurement: {variant} step {step}")
        indexed[key] = dict(row, global_step=step)
    return indexed


def merge_curve_measurements(
    validation_rows: Iterable[dict], sciq_rows: Iterable[dict]
) -> list[dict]:
    validation = _index_measurements(validation_rows, "validation")
    sciq = _index_measurements(sciq_rows, "SciQ")
    if validation.keys() != sciq.keys():
        missing_sciq = sorted(validation.keys() - sciq.keys())
        missing_validation = sorted(sciq.keys() - validation.keys())
        raise ValueError(
            "validation and SciQ measurement keys differ: "
            f"missing SciQ={missing_sciq}, missing validation={missing_validation}"
        )

    merged = []
    for key in sorted(validation, key=lambda item: (VARIANTS.index(item[0]) if item[0] in VARIANTS else len(VARIANTS), item[0], item[1])):
        row = dict(validation[key])
        for field, value in sciq[key].items():
            if field in ("variant", "global_step"):
                continue
            if field in row and row[field] != value:
                raise ValueError(f"conflicting field {field!r} for {key}")
            row[field] = value
        merged.append(row)

    expected_variants = tuple(dict.fromkeys(row["variant"] for row in merged))
    validate_curve_rows(merged, expected_variants=expected_variants)
    return merged


def validate_curve_rows(
    rows: Iterable[dict], expected_variants: Iterable[str] = VARIANTS
) -> list[dict]:
    rows = [dict(row) for row in rows]
    expected_variants = tuple(expected_variants)
    if not rows:
        raise ValueError("curve rows must not be empty")

    by_variant = {variant: [] for variant in expected_variants}
    for row in rows:
        missing = [column for column in CURVE_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"curve row is missing columns: {', '.join(missing)}")
        variant = row["variant"]
        if variant not in by_variant:
            raise ValueError(f"unexpected variant: {variant}")
        by_variant[variant].append(row)

        for field in (
            "global_step",
            "cumulative_train_tokens",
            "sciq_examples",
        ):
            value = _finite_number(row[field], field)
            if value < 0 or int(value) != value:
                raise ValueError(f"{field} must be a non-negative integer")
        for field in (
            "cumulative_train_gpu_hours",
            "validation_loss",
            "validation_perplexity",
            "sciq_acc",
            "sciq_acc_norm",
            "sciq_acc_stderr",
            "sciq_acc_norm_stderr",
        ):
            value = _finite_number(row[field], field)
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
        if not 0 <= float(row["sciq_acc"]) <= 1:
            raise ValueError("sciq_acc must be between zero and one")
        if not 0 <= float(row["sciq_acc_norm"]) <= 1:
            raise ValueError("sciq_acc_norm must be between zero and one")

    for variant, variant_rows in by_variant.items():
        if not variant_rows:
            raise ValueError(f"missing variant: {variant}")
        variant_rows.sort(key=lambda row: int(row["global_step"]))
        if int(variant_rows[0]["global_step"]) != 0:
            raise ValueError(f"{variant} curve must begin at global step zero")
        for previous, current in zip(variant_rows, variant_rows[1:]):
            if int(current["global_step"]) <= int(previous["global_step"]):
                raise ValueError(f"{variant} global steps must strictly increase")
            if int(current["cumulative_train_tokens"]) <= int(
                previous["cumulative_train_tokens"]
            ):
                raise ValueError(f"{variant} token counts must strictly increase")
            if float(current["cumulative_train_gpu_hours"]) <= float(
                previous["cumulative_train_gpu_hours"]
            ):
                raise ValueError(f"{variant} GPU hours must strictly increase")
    return rows


def write_curve_csv(path: str | Path, rows: Iterable[dict]) -> Path:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("curve rows must not be empty")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CURVE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path
