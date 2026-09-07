# Qwen peS2o Perplexity Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Colab notebook that evaluates Qwen2.5-0.5B Base on a deterministic 1,000-document peS2o V2 validation sample and records progress and results in W&B.

**Architecture:** Put deterministic data validation, token packing, and metric aggregation in a small tested Python module. Generate a standalone `.ipynb` that embeds that module, adds Colab configuration and orchestration cells, runs V100 inference, logs W&B metrics, and uploads a JSON artifact.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers, requests, Weights & Biases, Google Colab notebook JSON, unittest.

**Spec:** `docs/superpowers/specs/2026-09-08-qwen-pes2o-perplexity-colab-design.md`

## Global Constraints

- Model identifier: `Qwen/Qwen2.5-0.5B`.
- Use CUDA FP16, `model.eval()`, and `torch.inference_mode()`.
- Default sample: 320 S2ORC and 680 S2AG documents.
- Tokenize without automatic special tokens and append one EOS per document.
- Sequence length: 2,048. Default batch size: 4.
- Mask padding labels with `-100` and exclude padding and each sequence's first token from predicted-token counts.
- Derive overall perplexity from summed negative log-likelihood and summed predicted tokens.
- The notebook must run without importing project files.
- Log progress, final metrics, configuration, and the JSON result artifact to W&B.

---

## File Map

- `src/pes2o_perplexity.py`: streaming, validation, packing, aggregation, evaluation, and JSON helpers embedded in the notebook.
- `tests/test_pes2o_perplexity.py`: unit tests using synthetic records and a fake tokenizer.
- `scripts/build_colab_notebook.py`: deterministic standalone notebook generator.
- `tests/test_colab_notebook.py`: notebook JSON, syntax, and requirement tests.
- `notebooks/qwen_pes2o_validation_perplexity.ipynb`: final Colab artifact.

### Task 1: Deterministic data, packing, and aggregation

**Files:**
- Create: `src/__init__.py`
- Create: `src/pes2o_perplexity.py`
- Create: `tests/test_pes2o_perplexity.py`

**Interfaces:**
- Produces: `validate_record(record: dict, expected_source: str | None = None) -> dict`.
- Produces: `iter_jsonl_gz(url: str) -> Iterator[dict]`.
- Produces: `collect_source_records(urls: Iterable[str], limits: dict[str, int | None]) -> dict[str, list[dict]]`.
- Produces: `iter_packed_sequences(records: Iterable[dict], tokenizer: Any, sequence_length: int) -> Iterator[dict]`.
- Produces: `perplexity_from_loss(mean_loss: float) -> float`.
- Produces: `combine_source_metrics(metrics: Iterable[dict]) -> dict`.
- Produces: `write_result_json(path: str | Path, result: dict) -> Path`.

- [ ] **Step 1: Write failing validation and packing tests**

```python
class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        return [ord(char) - 96 for char in text]


class CoreTests(unittest.TestCase):
    def test_packing_adds_eos_and_masks_final_padding(self):
        records = [
            {"id": "a", "source": "s2orc", "text": "ab"},
            {"id": "b", "source": "s2orc", "text": "cde"},
        ]
        packed = list(iter_packed_sequences(records, FakeTokenizer(), 4))
        self.assertEqual(packed[0]["input_ids"], [1, 2, 99, 3])
        self.assertEqual(packed[1]["input_ids"], [4, 5, 99, 99])
        self.assertEqual(packed[1]["labels"], [4, 5, 99, -100])
        self.assertEqual(
            [item["document_ids"] for item in packed], [["a", "b"], ["b"]]
        )

    def test_validate_record_rejects_wrong_source(self):
        with self.assertRaisesRegex(ValueError, "expected source s2orc"):
            validate_record(
                {"id": "x", "source": "s2ag", "text": "valid"}, "s2orc"
            )
```

- [ ] **Step 2: Confirm the tests fail before implementation**

Run: `python3 -m unittest tests.test_pes2o_perplexity -v`

Expected: FAIL because `src.pes2o_perplexity` does not exist.

- [ ] **Step 3: Implement streaming, validation, and bounded packing**

Use `requests.get(url, stream=True, timeout=(30, 300))`,
`response.raise_for_status()`, `gzip.GzipFile`, and `io.TextIOWrapper`.
Validate `id`, `source`, and `text` as non-empty strings and require source to be
`s2orc` or `s2ag`. `collect_source_records` routes validated records from both
URLs until each requested count is met and raises an error if either count is
short.

```python
def validate_record(record, expected_source=None):
    for key in ("id", "source", "text"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"record field {key!r} must be a non-empty string")
    if record["source"] not in {"s2orc", "s2ag"}:
        raise ValueError(f"unsupported source {record['source']}")
    if expected_source is not None and record["source"] != expected_source:
        raise ValueError(
            f"expected source {expected_source}, received {record['source']}"
        )
    return record
```

The packing generator uses one list buffer, adds exactly one EOS per document,
yields every full block, and pads only the final block. `document_ids` lists all
documents contributing tokens to each block.

- [ ] **Step 4: Write failing aggregation and JSON tests**

```python
def test_combination_uses_predicted_token_weights(self):
    combined = combine_source_metrics([
        {"source": "s2orc", "negative_log_likelihood": 20.0, "predicted_tokens": 4},
        {"source": "s2ag", "negative_log_likelihood": 10.0, "predicted_tokens": 6},
    ])
    self.assertAlmostEqual(combined["loss"], 3.0)
    self.assertAlmostEqual(combined["perplexity"], math.exp(3.0))


def test_result_json_round_trips(self):
    with tempfile.TemporaryDirectory() as directory:
        path = write_result_json(Path(directory) / "result.json", {"perplexity": 2.5})
        self.assertEqual(json.loads(path.read_text()), {"perplexity": 2.5})
```

- [ ] **Step 5: Implement guarded aggregation and serialization**

Require finite loss and positive predicted-token totals. Calculate overall loss
from total negative log-likelihood divided by total predicted tokens. Write
UTF-8 JSON with two-space indentation and create the parent directory.

- [ ] **Step 6: Run Task 1 tests**

Run: `python3 -m unittest tests.test_pes2o_perplexity -v`

Expected: all Task 1 tests PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/__init__.py src/pes2o_perplexity.py tests/test_pes2o_perplexity.py
git commit -m "feat: add peS2o perplexity data helpers"
```

### Task 2: FP16 evaluation and progress metrics

**Files:**
- Modify: `src/pes2o_perplexity.py`
- Modify: `tests/test_pes2o_perplexity.py`

**Interfaces:**
- Consumes: dictionaries from `iter_packed_sequences`.
- Produces: `count_shifted_targets(labels: Any) -> int`.
- Produces: `evaluate_source(model: Any, packed_sequences: Iterable[dict], source: str, batch_size: int, device: str, log_every_steps: int, progress_callback: Callable[[dict], None] | None) -> dict`.

- [ ] **Step 1: Write failing shifted-target tests**

Guard tensor tests with `unittest.skipUnless(importlib.util.find_spec("torch"),
"torch unavailable")` so a minimal local environment can run pure tests.

```python
def test_count_shifted_targets_excludes_first_column_and_padding():
    labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
    assert count_shifted_targets(labels) == 3
```

- [ ] **Step 2: Confirm the new tests fail**

Run: `python3 -m unittest tests.test_pes2o_perplexity -v`

Expected: FAIL because `count_shifted_targets` and `evaluate_source` are absent.

- [ ] **Step 3: Implement evaluation without gradients**

Import torch inside evaluation functions. Form input, label, and attention-mask
tensors in batches and transfer them to CUDA. Evaluate under
`torch.inference_mode()`. Count targets with
`labels[:, 1:].ne(-100).sum()`, multiply mean loss by that batch count, and
accumulate negative log-likelihood, tokens, sequences, batches, elapsed time,
and throughput. Call the progress callback after the first batch and at each log
interval.

Catch `torch.cuda.OutOfMemoryError` and raise a `RuntimeError` containing the
active batch size and instruction to reduce it. Let other errors propagate.

- [ ] **Step 4: Run Task 2 tests and commit**

Run: `python3 -m unittest tests.test_pes2o_perplexity -v`

Expected: all tests PASS; torch-only tests may be skipped if torch is absent.

```bash
git add src/pes2o_perplexity.py tests/test_pes2o_perplexity.py
git commit -m "feat: evaluate packed causal language model loss"
```

### Task 3: Standalone Colab notebook

**Files:**
- Create: `scripts/build_colab_notebook.py`
- Create: `tests/test_colab_notebook.py`
- Create: `notebooks/qwen_pes2o_validation_perplexity.ipynb`

**Interfaces:**
- Consumes: exact source text from `src/pes2o_perplexity.py`.
- Produces: notebook cells for dependencies, embedded helpers, configuration,
  W&B login, model loading, smoke test, source evaluation, JSON output, artifact
  upload, and explicit run finish.

- [ ] **Step 1: Write failing notebook structure tests**

Load the notebook with `json`, require `nbformat == 4`, compile every code cell,
and assert the source contains the model ID, both official validation filenames,
320/680 limits, `wandb.login()`, `torch.float16`, `torch.inference_mode()`, and
`wandb.Artifact`. Assert it contains neither `from src` nor `import src`.

```python
def test_every_code_cell_compiles(self):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")
```

- [ ] **Step 2: Confirm the notebook test fails**

Run: `python3 -m unittest tests.test_colab_notebook -v`

Expected: FAIL because the notebook does not exist.

- [ ] **Step 3: Implement the notebook builder**

Use the standard library to produce notebook v4 JSON with Colab GPU metadata.
Install compatible major versions of Transformers, W&B, and requests. Embed the
exact core module source in a code cell.

```python
CONFIG = {
    "model_id": "Qwen/Qwen2.5-0.5B",
    "sequence_length": 2048,
    "batch_size": 4,
    "seed": 42,
    "log_every_steps": 10,
    "wandb_project": "lshbloom-pes2o",
    "run_name": "qwen2.5-0.5b-base-pes2o-valid-1000",
    "s2orc_documents": 320,
    "s2ag_documents": 680,
}
```

Use URLs ending in `validation-00000-of-00002.json.gz` and
`validation-00001-of-00002.json.gz`. Inspect each record's `source` and collect
until both quotas are met; do not trust filename-to-source mapping alone.

After CUDA checks, call `wandb.login()`, create the run, load tokenizer and model
with `torch_dtype=torch.float16`, and run a one-sequence finite-loss smoke test.
Then evaluate both sources, combine metrics, write
`/content/results/qwen2.5-0.5b-base-pes2o-valid-1000.json`, populate W&B summary,
upload the file as a `wandb.Artifact`, and always call `wandb.finish()`.

- [ ] **Step 4: Generate, test, and commit the notebook**

Run: `python3 scripts/build_colab_notebook.py`

Run: `python3 -m unittest tests.test_colab_notebook -v`

Expected: the notebook is created and all notebook tests PASS.

```bash
git add scripts/build_colab_notebook.py tests/test_colab_notebook.py notebooks/qwen_pes2o_validation_perplexity.ipynb
git commit -m "feat: add Qwen peS2o perplexity Colab notebook"
```

### Task 4: Final reproducibility verification

**Files:**
- Modify only if a verification step finds a concrete defect.

**Interfaces:**
- Consumes: source, tests, builder, and generated notebook.
- Produces: verified final notebook and clean implementation commits.

- [ ] **Step 1: Prove notebook generation is deterministic**

Run: `python3 scripts/build_colab_notebook.py`

Run: `git diff --exit-code -- notebooks/qwen_pes2o_validation_perplexity.ipynb`

Expected: both commands exit 0.

- [ ] **Step 2: Run the complete local test suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: every non-CUDA test passes. V100 inference is covered by the notebook's
Colab smoke test.

- [ ] **Step 3: Check the design acceptance criteria in the artifact**

Use a Python inspection to assert the notebook contains one default model ID,
the 320/680 limits, FP16 loading, inference mode, weighted aggregation, JSON
creation, W&B final metrics, artifact upload, and explicit `wandb.finish()`.

- [ ] **Step 4: Inspect repository state**

Run: `git status --short`

Expected: only pre-existing local data and cache directories are untracked; all
implementation files are committed.
