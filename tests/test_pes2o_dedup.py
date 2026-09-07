import math
import tempfile
import unittest
from pathlib import Path

from src.pes2o_dedup import (
    build_indexes,
    normalize_unigrams,
    per_filter_fp,
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


if __name__ == "__main__":
    unittest.main()
