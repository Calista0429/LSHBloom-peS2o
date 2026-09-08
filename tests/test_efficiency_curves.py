import json
import tempfile
import unittest
from pathlib import Path

from src.efficiency_curves import (
    build_full_training_plan,
    canonical_sha256,
    checkpoint_steps,
    merge_curve_measurements,
    tokens_at_step,
    validate_curve_rows,
    validate_shared_experiment_results,
    write_curve_csv,
)


TOKEN_COUNTS = {
    "raw": 31_410_586,
    "minhashlsh": 28_805_685,
    "lshbloom": 28_752_978,
}


def ppl_row(variant, step, tokens, hours, perplexity):
    return {
        "variant": variant,
        "global_step": step,
        "cumulative_train_tokens": tokens,
        "cumulative_train_gpu_hours": hours,
        "validation_loss": 2.0,
        "validation_perplexity": perplexity,
    }


def sciq_row(variant, step, acc_norm):
    return {
        "variant": variant,
        "global_step": step,
        "sciq_acc": acc_norm - 0.01,
        "sciq_acc_norm": acc_norm,
        "sciq_acc_stderr": 0.01,
        "sciq_acc_norm_stderr": 0.01,
        "sciq_examples": 1000,
    }


class FullTrainingPlanTests(unittest.TestCase):
    def test_uses_every_complete_sequence_without_exceeding_each_variant(self):
        plan = build_full_training_plan(TOKEN_COUNTS, sequence_length=2048)

        self.assertEqual(
            plan,
            {
                "raw": {
                    "available_tokens": 31_410_586,
                    "sequence_count": 15_337,
                    "train_input_tokens": 31_410_176,
                    "unused_tail_tokens": 410,
                },
                "minhashlsh": {
                    "available_tokens": 28_805_685,
                    "sequence_count": 14_065,
                    "train_input_tokens": 28_805_120,
                    "unused_tail_tokens": 565,
                },
                "lshbloom": {
                    "available_tokens": 28_752_978,
                    "sequence_count": 14_039,
                    "train_input_tokens": 28_751_872,
                    "unused_tail_tokens": 1_106,
                },
            },
        )

    def test_rejects_dataset_too_small_for_one_sequence(self):
        with self.assertRaisesRegex(ValueError, "complete sequence"):
            build_full_training_plan({"raw": 1024}, sequence_length=2048)


class CheckpointAccountingTests(unittest.TestCase):
    def test_includes_base_regular_intervals_and_final_partial_update(self):
        self.assertEqual(
            checkpoint_steps(total_optimizer_steps=1918, save_steps=250),
            [0, 250, 500, 750, 1000, 1250, 1500, 1750, 1918],
        )

    def test_clamps_last_partial_optimizer_update_to_actual_token_total(self):
        self.assertEqual(
            tokens_at_step(
                global_step=1918,
                tokens_per_optimizer_step=16_384,
                total_train_tokens=31_410_176,
            ),
            31_410_176,
        )
        self.assertEqual(
            tokens_at_step(250, 16_384, 31_410_176),
            4_096_000,
        )


class CurveResultTests(unittest.TestCase):
    def test_accepts_only_results_with_the_same_verified_experiment_identity(self):
        identity = {
            "model_id": "Qwen/Qwen2.5-0.5B",
            "seed": 42,
            "validation_revision": "abc123",
            "probe_sha256": "f" * 64,
        }
        fingerprint = canonical_sha256(identity)
        results = [
            {
                "schema_version": 2,
                "variant": variant,
                "experiment_identity": identity,
                "experiment_fingerprint": fingerprint,
            }
            for variant in ("raw", "minhashlsh", "lshbloom")
        ]

        self.assertEqual(
            validate_shared_experiment_results(results),
            fingerprint,
        )

    def test_rejects_mismatched_or_tampered_experiment_results(self):
        identity = {"model_id": "Qwen/Qwen2.5-0.5B", "seed": 42}
        fingerprint = canonical_sha256(identity)
        results = [
            {
                "schema_version": 2,
                "variant": variant,
                "experiment_identity": identity,
                "experiment_fingerprint": fingerprint,
            }
            for variant in ("raw", "minhashlsh", "lshbloom")
        ]
        results[2] = {
            **results[2],
            "experiment_identity": {**identity, "seed": 43},
        }

        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_shared_experiment_results(results)

    def test_merges_validation_and_sciq_measurements_by_variant_and_step(self):
        rows = merge_curve_measurements(
            [
                ppl_row("raw", 0, 0, 0.0, 9.0),
                ppl_row("raw", 250, 4_096_000, 0.1, 7.0),
            ],
            [
                sciq_row("raw", 0, 0.70),
                sciq_row("raw", 250, 0.75),
            ],
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["cumulative_train_tokens"], 4_096_000)
        self.assertEqual(rows[1]["validation_perplexity"], 7.0)
        self.assertEqual(rows[1]["sciq_acc_norm"], 0.75)

    def test_rejects_a_missing_sciq_measurement(self):
        with self.assertRaisesRegex(ValueError, "measurement keys differ"):
            merge_curve_measurements(
                [
                    ppl_row("raw", 0, 0, 0.0, 9.0),
                    ppl_row("raw", 250, 4_096_000, 0.1, 7.0),
                ],
                [sciq_row("raw", 0, 0.70)],
            )

    def test_rejects_non_monotonic_token_or_gpu_time_curves(self):
        bad_rows = [
            {**ppl_row("raw", 0, 0, 0.0, 9.0), **sciq_row("raw", 0, 0.70)},
            {
                **ppl_row("raw", 250, 4_096_000, 0.1, 7.0),
                **sciq_row("raw", 250, 0.75),
            },
            {
                **ppl_row("raw", 500, 3_000_000, 0.2, 6.5),
                **sciq_row("raw", 500, 0.78),
            },
        ]

        with self.assertRaisesRegex(ValueError, "strictly increase"):
            validate_curve_rows(bad_rows, expected_variants=("raw",))

    def test_writes_stable_csv_columns_for_plotting_and_wandb(self):
        rows = merge_curve_measurements(
            [ppl_row("raw", 0, 0, 0.0, 9.0)],
            [sciq_row("raw", 0, 0.70)],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = write_curve_csv(Path(directory) / "curve.csv", rows)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            lines[0],
            "variant,global_step,cumulative_train_tokens,cumulative_train_gpu_hours,"
            "validation_loss,validation_perplexity,sciq_acc,sciq_acc_norm,"
            "sciq_acc_stderr,sciq_acc_norm_stderr,sciq_examples",
        )
        self.assertEqual(json.loads(json.dumps(rows))[0]["variant"], "raw")


if __name__ == "__main__":
    unittest.main()
