import gzip
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pes2o_perplexity import (
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

class CoreTests(unittest.TestCase):
    def test_validate_record_rejects_wrong_source(self):
        with self.assertRaisesRegex(ValueError, "expected source s2orc"):
            validate_record(
                {"id": "x", "source": "s2ag", "text": "valid"}, "s2orc"
            )

    def test_validate_record_rejects_empty_text(self):
        with self.assertRaisesRegex(ValueError, "field 'text'"):
            validate_record({"id": "x", "source": "s2ag", "text": ""})

    @patch("urllib.request.urlopen")
    def test_iter_jsonl_gz_streams_valid_records(self, urlopen):
        records = [
            {"id": "a", "source": "s2ag", "text": "one"},
            {"id": "b", "source": "s2ag", "text": "two"},
        ]
        urlopen.return_value = FakeResponse(records)

        self.assertEqual(list(iter_jsonl_gz("https://example.test/data.gz")), records)
        urlopen.assert_called_once_with("https://example.test/data.gz", timeout=300)

    @patch("src.pes2o_perplexity.iter_jsonl_gz")
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
        self.assertEqual(
            [item["document_ids"] for item in packed], [["a", "b"], ["b"]]
        )

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


if __name__ == "__main__":
    unittest.main()
