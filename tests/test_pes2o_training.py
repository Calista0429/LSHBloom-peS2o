import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lshbloom_pes2o.training import TokenMemmapDataset, pack_jsonl_gz_to_memmap


class FakeTokenizer:
    eos_token_id = 99
    name_or_path = "fake-tokenizer"

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        return [ord(character) - 96 for character in text]


def write_records(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


class TrainingDataTests(unittest.TestCase):
    def test_packer_adds_eos_and_fills_exact_full_sequences(self):
        records = [
            {"id": "a", "source": "s2orc/train", "text": "ab"},
            {"id": "b", "source": "s2orc/train", "text": "cdefgh"},
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl.gz"
            output_path = root / "tokens.uint32"
            write_records(input_path, records)

            metadata = pack_jsonl_gz_to_memmap(
                input_path=input_path,
                output_path=output_path,
                tokenizer=FakeTokenizer(),
                sequence_length=4,
                sequence_count=2,
            )
            values = np.memmap(output_path, mode="r", dtype=np.uint32).tolist()

        self.assertEqual(values, [1, 2, 99, 3, 4, 5, 6, 7])
        self.assertEqual(metadata["input_tokens"], 8)
        self.assertEqual(metadata["sequence_count"], 2)
        self.assertEqual(metadata["documents_read"], 2)
        self.assertEqual(metadata["last_document_id"], "b")
        self.assertEqual(len(metadata["sha256"]), 64)

    def test_packer_removes_partial_output_when_tokens_are_insufficient(self):
        records = [
            {"id": "a", "source": "s2orc/train", "text": "ab"},
            {"id": "b", "source": "s2orc/train", "text": "cd"},
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl.gz"
            output_path = root / "tokens.uint32"
            write_records(input_path, records)

            with self.assertRaisesRegex(ValueError, "requires 8 tokens, found 6"):
                pack_jsonl_gz_to_memmap(
                    input_path=input_path,
                    output_path=output_path,
                    tokenizer=FakeTokenizer(),
                    sequence_length=4,
                    sequence_count=2,
                )

            self.assertFalse(output_path.exists())

    def test_memmap_dataset_returns_int64_input_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.uint32"
            values = np.memmap(path, mode="w+", dtype=np.uint32, shape=(8,))
            values[:] = np.arange(8)
            values.flush()
            del values

            dataset = TokenMemmapDataset(path, sequence_length=4, sequence_count=2)

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[1]["input_ids"].dtype, np.int64)
            self.assertEqual(dataset[1]["input_ids"].tolist(), [4, 5, 6, 7])
            with self.assertRaises(IndexError):
                dataset[2]


if __name__ == "__main__":
    unittest.main()
