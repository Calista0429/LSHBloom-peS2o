import gzip
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.pes2o_dedup import (
    build_indexes,
    DedupConfig,
    normalize_unigrams,
    per_filter_fp,
    prepare_variants,
    stream_decisions,
)


class DedupCoreTests(unittest.TestCase):
    def test_normalize_unigrams_lowercases_splits_and_deduplicates(self):
        self.assertEqual(
            normalize_unigrams("  Alpha beta\nALPHA  "),
            (b"alpha", b"beta"),
        )

    def test_normalize_unigrams_rejects_text_without_tokens(self):
        with self.assertRaisesRegex(ValueError, "no unigrams"):
            normalize_unigrams(" \n\t ")

    def test_per_filter_fp_reconstructs_requested_effective_rate(self):
        effective_fp = 1e-10
        bands = 42

        result = per_filter_fp(effective_fp, bands)
        reconstructed = -math.expm1(bands * math.log1p(-result))

        self.assertGreater(result, 0.0)
        self.assertLess(result, effective_fp)
        self.assertAlmostEqual(reconstructed, effective_fp, delta=1e-20)

    def test_stream_keeps_first_and_removes_later_exact_duplicate(self):
        records = [
            {"id": "first", "source": "s2orc/train", "text": "alpha beta gamma"},
            {"id": "other", "source": "s2orc/train", "text": "delta epsilon zeta"},
            {"id": "copy", "source": "s2orc/train", "text": "Alpha beta gamma"},
        ]

        with tempfile.TemporaryDirectory() as directory:
            lsh, bloom, metadata = build_indexes(
                threshold=0.5,
                num_perm=64,
                expected_documents=len(records),
                effective_fp=1e-10,
                bloom_save_dir=Path(directory) / "bloom-index",
            )
            decisions = list(
                stream_decisions(
                    records,
                    minhash_index=lsh,
                    bloom_index=bloom,
                    num_perm=64,
                    seed=1,
                )
            )

        self.assertEqual(metadata["bands"], lsh.b)
        self.assertEqual(metadata["rows_per_band"], lsh.r)
        self.assertEqual(
            [(item.record["id"], item.minhashlsh_duplicate) for item in decisions],
            [("first", False), ("other", False), ("copy", True)],
        )
        self.assertEqual(
            [(item.record["id"], item.lshbloom_duplicate) for item in decisions],
            [("first", False), ("other", False), ("copy", True)],
        )

    def test_stream_rejects_repeated_record_id(self):
        records = [
            {"id": "same", "source": "s2orc/train", "text": "alpha beta"},
            {"id": "same", "source": "s2orc/train", "text": "gamma delta"},
        ]

        with tempfile.TemporaryDirectory() as directory:
            lsh, bloom, _ = build_indexes(
                threshold=0.5,
                num_perm=32,
                expected_documents=len(records),
                effective_fp=1e-10,
                bloom_save_dir=Path(directory) / "bloom-index",
            )
            with self.assertRaisesRegex(ValueError, "duplicate record id"):
                list(
                    stream_decisions(
                        records,
                        minhash_index=lsh,
                        bloom_index=bloom,
                        num_perm=32,
                        seed=1,
                    )
                )


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        return list(range(len(text.split())))


class PrepareVariantsTests(unittest.TestCase):
    def test_cli_can_run_directly_from_repository_root(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/prepare_pes2o_variants.py", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Create Raw, MinHashLSH", result.stdout)

    def test_prepare_writes_valid_variants_and_manifest(self):
        records = [
            {"id": "first", "source": "s2orc/train", "text": "alpha beta gamma"},
            {"id": "other", "source": "s2orc/train", "text": "delta epsilon zeta"},
            {"id": "copy", "source": "s2orc/train", "text": "Alpha beta gamma"},
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl.gz"
            with gzip.open(input_path, "wt", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")

            output_dir = root / "output"
            manifest = prepare_variants(
                input_path=input_path,
                output_dir=output_dir,
                config=DedupConfig(
                    expected_documents=3,
                    num_perm=64,
                    training_gate_rate=0.30,
                ),
                tokenizer=FakeTokenizer(),
            )

            def read_ids(relative_path):
                with gzip.open(output_dir / relative_path, "rt", encoding="utf-8") as stream:
                    return [json.loads(line)["id"] for line in stream]

            self.assertEqual(read_ids("raw/train.jsonl.gz"), ["first", "other", "copy"])
            self.assertEqual(read_ids("minhashlsh/train.jsonl.gz"), ["first", "other"])
            self.assertEqual(read_ids("lshbloom/train.jsonl.gz"), ["first", "other"])
            self.assertEqual(
                json.loads((output_dir / "manifest.json").read_text()), manifest
            )

        self.assertEqual(manifest["variants"]["raw"]["token_count"], 12)
        self.assertEqual(manifest["variants"]["minhashlsh"]["token_count"], 8)
        self.assertEqual(manifest["variants"]["lshbloom"]["removed_ids"], ["copy"])
        self.assertEqual(manifest["agreement"]["differing_ids"], [])
        self.assertTrue(manifest["training_gate"]["passes"])
        for name in ("raw", "minhashlsh", "lshbloom"):
            self.assertEqual(len(manifest["variants"][name]["sha256"]), 64)
            self.assertGreater(manifest["variants"][name]["compressed_bytes"], 0)

    def test_prepare_does_not_publish_partial_files_for_wrong_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl.gz"
            with gzip.open(input_path, "wt", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"id": "only", "source": "s2orc/train", "text": "alpha beta"}
                    )
                    + "\n"
                )

            output_dir = root / "output"
            with self.assertRaisesRegex(ValueError, "expected 2 documents"):
                prepare_variants(
                    input_path=input_path,
                    output_dir=output_dir,
                    config=DedupConfig(expected_documents=2, num_perm=32),
                )

            self.assertFalse((output_dir / "raw/train.jsonl.gz").exists())
            self.assertFalse((output_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
