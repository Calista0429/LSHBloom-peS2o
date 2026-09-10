import gzip
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lshbloom_pes2o.perplexity as core
from lshbloom_pes2o.perplexity import (
    collect_source_records,
    combine_source_metrics,
    iter_jsonl_gz,
    iter_packed_sequences,
    validate_record,
    write_result_json,
)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        return [ord(char) - 96 for char in text]


class FakeResponse(io.BytesIO):
    def __init__(self, records):
        content = "".join(json.dumps(record) + "\n" for record in records)
        compressed = gzip.compress(content.encode("utf-8"))
        super().__init__(compressed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeMask:
    def __init__(self, values):
        self.values = values

    def sum(self):
        return FakeScalar(sum(sum(row) for row in self.values))


class FakeShiftedLabels:
    def __init__(self, values):
        self.values = values

    def ne(self, ignored_value):
        return FakeMask(
            [[value != ignored_value for value in row] for row in self.values]
        )


class FakeLabels:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        rows, columns = key
        if rows != slice(None) or columns != slice(1, None):
            raise AssertionError(f"unexpected tensor slice: {key}")
        return FakeShiftedLabels([row[1:] for row in self.values])


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        rows, columns = key
        if rows != slice(None) or not isinstance(columns, slice):
            raise AssertionError(f"unexpected tensor slice: {key}")
        return FakeTensor([row[columns] for row in self.values])

    def ne(self, ignored_value):
        return FakeTensor(
            [[value != ignored_value for value in row] for row in self.values]
        )

    def sum(self):
        return FakeScalar(sum(sum(row) for row in self.values))

    def long(self):
        return self


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    @staticmethod
    def memory_allocated():
        return 0


class FakeTorchModule:
    long = "long"
    cuda = FakeCuda()

    @staticmethod
    def tensor(values, dtype=None, device=None):
        return FakeTensor(values)

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeModel:
    def __init__(self, losses):
        self.losses = iter(losses)

    def __call__(self, **inputs):
        return SimpleNamespace(loss=FakeScalar(next(self.losses)))


class CoreTests(unittest.TestCase):
    def test_validate_record_rejects_wrong_source(self):
        with self.assertRaisesRegex(ValueError, "expected source s2orc"):
            validate_record({"id": "x", "source": "s2ag", "text": "valid"}, "s2orc")

    def test_validate_record_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "field 'text'"):
            validate_record({"id": "x", "source": "s2ag", "text": ""})

    def test_validate_record_accepts_validation_source_suffix(self):
        record = {"id": "x", "source": "s2orc/valid", "text": "valid"}

        self.assertIs(validate_record(record, "s2orc"), record)

    @patch("urllib.request.urlopen")
    def test_iter_jsonl_gz_streams_valid_records(self, urlopen):
        records = [
            {"id": "a", "source": "s2ag", "text": "one"},
            {"id": "b", "source": "s2ag", "text": "two"},
        ]
        urlopen.return_value = FakeResponse(records)

        self.assertEqual(list(iter_jsonl_gz("https://example.test/data.gz")), records)
        urlopen.assert_called_once_with("https://example.test/data.gz", timeout=300)

    @patch("lshbloom_pes2o.perplexity.iter_jsonl_gz")
    def test_collect_source_records_routes_until_each_limit(self, stream):
        stream.side_effect = [
            iter(
                [
                    {"id": "a1", "source": "s2ag", "text": "a"},
                    {"id": "a2", "source": "s2ag", "text": "b"},
                    {"id": "a3", "source": "s2ag", "text": "c"},
                ]
            ),
            iter(
                [
                    {"id": "o1", "source": "s2orc", "text": "d"},
                    {"id": "o2", "source": "s2orc", "text": "e"},
                ]
            ),
        ]

        result = collect_source_records(
            ["first.gz", "second.gz"], {"s2ag": 2, "s2orc": 1}
        )

        self.assertEqual([record["id"] for record in result["s2ag"]], ["a1", "a2"])
        self.assertEqual([record["id"] for record in result["s2orc"]], ["o1"])

    @patch("lshbloom_pes2o.perplexity.iter_jsonl_gz")
    def test_collect_source_records_routes_validation_source_suffixes(self, stream):
        stream.side_effect = [
            iter([{"id": "a1", "source": "s2ag/valid", "text": "a"}]),
            iter([{"id": "o1", "source": "s2orc/valid", "text": "b"}]),
        ]

        result = collect_source_records(
            ["first.gz", "second.gz"], {"s2ag": 1, "s2orc": 1}
        )

        self.assertEqual(result["s2ag"][0]["id"], "a1")
        self.assertEqual(result["s2orc"][0]["id"], "o1")

    def test_packing_adds_eos_and_masks_final_padding(self):
        records = [
            {"id": "a", "source": "s2orc", "text": "ab"},
            {"id": "b", "source": "s2orc", "text": "cde"},
        ]

        packed = list(iter_packed_sequences(records, FakeTokenizer(), 4))

        self.assertEqual(packed[0]["input_ids"], [1, 2, 99, 3])
        self.assertEqual(packed[0]["labels"], [1, 2, 99, 3])
        self.assertEqual(packed[1]["input_ids"], [4, 5, 99, 99])
        self.assertEqual(packed[1]["labels"], [4, 5, 99, -100])
        self.assertEqual([item["document_ids"] for item in packed], [["a", "b"], ["b"]])

    def test_combination_uses_predicted_token_weights(self):
        combined = combine_source_metrics(
            [
                {
                    "source": "s2orc",
                    "negative_log_likelihood": 20.0,
                    "predicted_tokens": 4,
                },
                {
                    "source": "s2ag",
                    "negative_log_likelihood": 10.0,
                    "predicted_tokens": 6,
                },
            ]
        )

        self.assertEqual(combined["negative_log_likelihood"], 30.0)
        self.assertEqual(combined["predicted_tokens"], 10)
        self.assertAlmostEqual(combined["loss"], 3.0)
        self.assertAlmostEqual(combined["perplexity"], math.exp(3.0))

    def test_result_json_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_result_json(
                Path(directory) / "result.json", {"perplexity": 2.5}
            )

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), {"perplexity": 2.5}
            )

    def test_count_shifted_targets_excludes_first_column_and_padding(self):
        labels = FakeLabels([[1, 2, -100], [3, 4, 5]])

        self.assertEqual(core.count_shifted_targets(labels), 3)

    def test_evaluate_source_weights_batches_by_target_count(self):
        packed = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "document_ids": ["a"]},
            {
                "input_ids": [4, 5, 99],
                "labels": [4, 5, -100],
                "document_ids": ["b"],
            },
            {"input_ids": [6, 7, 8], "labels": [6, 7, 8], "document_ids": ["c"]},
        ]
        progress = []

        with patch.dict(sys.modules, {"torch": FakeTorchModule()}):
            result = core.evaluate_source(
                model=FakeModel([2.0, 4.0]),
                packed_sequences=packed,
                source="s2orc",
                batch_size=2,
                device="cuda",
                log_every_steps=1,
                progress_callback=progress.append,
            )

        self.assertEqual(result["predicted_tokens"], 5)
        self.assertEqual(result["negative_log_likelihood"], 14.0)
        self.assertAlmostEqual(result["loss"], 2.8)
        self.assertAlmostEqual(result["perplexity"], math.exp(2.8))
        self.assertEqual(result["documents"], 3)
        self.assertEqual(result["sequences"], 3)
        self.assertEqual(result["batches"], 2)
        self.assertEqual(len(progress), 2)


if __name__ == "__main__":
    unittest.main()
