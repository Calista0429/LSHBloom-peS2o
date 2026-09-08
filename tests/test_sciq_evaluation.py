import json
import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from src.sciq_evaluation import (
    VARIANTS,
    build_comparison,
    extract_sciq_metrics,
    to_jsonable,
    write_json,
)


def harness_result(effective=1000, **metric_overrides):
    metrics = {
        "acc,none": 0.42,
        "acc_stderr,none": 0.01,
        "acc_norm,none": 0.48,
        "acc_norm_stderr,none": 0.02,
        "alias": "sciq",
    }
    metrics.update(metric_overrides)
    return {
        "results": {"sciq": metrics},
        "n-samples": {"sciq": {"original": 1000, "effective": effective}},
    }


def full_row(variant, acc, acc_norm):
    return {
        "variant": variant,
        "stage": "full",
        "examples": 1000,
        "acc": acc,
        "acc_stderr": 0.01,
        "acc_norm": acc_norm,
        "acc_norm_stderr": 0.02,
        "elapsed_seconds": 12.5,
        "peak_cuda_bytes": 1024,
    }


class MetricExtractionTests(unittest.TestCase):
    def test_extracts_full_sciq_metrics(self):
        row = extract_sciq_metrics(
            harness_result(),
            variant="raw",
            stage="full",
            elapsed_seconds=12.5,
            peak_cuda_bytes=1024,
        )

        self.assertEqual(
            row,
            {
                "variant": "raw",
                "stage": "full",
                "examples": 1000,
                "acc": 0.42,
                "acc_stderr": 0.01,
                "acc_norm": 0.48,
                "acc_norm_stderr": 0.02,
                "elapsed_seconds": 12.5,
                "peak_cuda_bytes": 1024,
            },
        )

    def test_accepts_exact_smoke_sample_count(self):
        row = extract_sciq_metrics(
            harness_result(effective=10), "lshbloom", "smoke", 1.0, 0
        )

        self.assertEqual(row["examples"], 10)

    def test_rejects_unknown_variant(self):
        with self.assertRaisesRegex(ValueError, "unsupported variant"):
            extract_sciq_metrics(harness_result(), "other", "full", 1.0, 0)

    def test_rejects_unknown_stage(self):
        with self.assertRaisesRegex(ValueError, "unsupported stage"):
            extract_sciq_metrics(harness_result(), "raw", "pilot", 1.0, 0)

    def test_rejects_missing_metric(self):
        result = harness_result()
        del result["results"]["sciq"]["acc_norm,none"]

        with self.assertRaisesRegex(ValueError, "missing SciQ metric"):
            extract_sciq_metrics(result, "raw", "full", 1.0, 0)

    def test_rejects_non_finite_metric(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            extract_sciq_metrics(
                harness_result(**{"acc_norm,none": math.nan}),
                "raw",
                "full",
                1.0,
                0,
            )

    def test_rejects_wrong_stage_sample_count(self):
        with self.assertRaisesRegex(ValueError, "expected 1000"):
            extract_sciq_metrics(
                harness_result(effective=999), "raw", "full", 1.0, 0
            )

    def test_rejects_non_positive_elapsed_time(self):
        with self.assertRaisesRegex(ValueError, "elapsed_seconds"):
            extract_sciq_metrics(harness_result(), "raw", "full", 0.0, 0)

    def test_rejects_negative_peak_memory(self):
        with self.assertRaisesRegex(ValueError, "peak_cuda_bytes"):
            extract_sciq_metrics(harness_result(), "raw", "full", 1.0, -1)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.raw = full_row("raw", 0.40, 0.45)
        self.minhash = full_row("minhashlsh", 0.41, 0.47)
        self.bloom = full_row("lshbloom", 0.39, 0.46)

    def test_orders_variants_and_computes_raw_relative_deltas(self):
        comparison = build_comparison([self.bloom, self.raw, self.minhash])

        self.assertEqual([row["variant"] for row in comparison], list(VARIANTS))
        self.assertEqual(comparison[0]["delta_acc"], 0.0)
        self.assertEqual(comparison[0]["delta_acc_norm"], 0.0)
        self.assertAlmostEqual(comparison[1]["delta_acc"], 0.01)
        self.assertAlmostEqual(comparison[1]["delta_acc_norm"], 0.02)
        self.assertAlmostEqual(comparison[2]["delta_acc"], -0.01)
        self.assertAlmostEqual(comparison[2]["delta_acc_norm"], 0.01)

    def test_does_not_mutate_input_rows(self):
        originals = [dict(self.raw), dict(self.minhash), dict(self.bloom)]

        build_comparison([self.raw, self.minhash, self.bloom])

        self.assertEqual([self.raw, self.minhash, self.bloom], originals)

    def test_rejects_missing_variant(self):
        with self.assertRaisesRegex(ValueError, "missing variants: lshbloom"):
            build_comparison([self.raw, self.minhash])

    def test_rejects_duplicate_variant(self):
        with self.assertRaisesRegex(ValueError, "duplicate variant: raw"):
            build_comparison([self.raw, dict(self.raw), self.minhash, self.bloom])

    def test_rejects_smoke_rows(self):
        smoke = dict(self.raw, stage="smoke", examples=10)

        with self.assertRaisesRegex(ValueError, "full-stage"):
            build_comparison([smoke, self.minhash, self.bloom])


class SerializationTests(unittest.TestCase):
    def test_converts_dataclass_path_tuple_and_scalar_like_value(self):
        @dataclass
        class Record:
            path: Path
            values: tuple

        class Scalar:
            def item(self):
                return 7

        converted = to_jsonable(Record(Path("result.json"), (Scalar(), 2)))

        self.assertEqual(converted, {"path": "result.json", "values": [7, 2]})

    def test_write_json_creates_parent_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"

            returned = write_json(path, {"score": 0.48})

            self.assertEqual(returned, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"score": 0.48})

    def test_write_json_rejects_non_finite_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_json(Path(directory) / "bad.json", {"score": math.inf})


if __name__ == "__main__":
    unittest.main()
