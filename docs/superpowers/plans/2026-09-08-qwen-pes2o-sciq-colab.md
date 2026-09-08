# Qwen peS2o SciQ Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Colab notebook that evaluates the Raw, MinHashLSH, and LSHBloom Qwen2.5-0.5B checkpoints on the complete zero-shot SciQ test split and records a fair comparison in W&B.

**Architecture:** Put result validation, JSON conversion, and Raw-relative comparison logic in a small dependency-light Python module. Embed that tested module into a generated notebook that reads Colab secrets, validates and downloads each S3 checkpoint, runs a 10-example smoke test followed by the complete SciQ evaluation through `lm-evaluation-harness`, releases each model, and uploads the final files to one W&B run.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, `lm-evaluation-harness==0.4.13`, boto3, pandas, W&B, Google Colab notebook JSON, unittest.

**Spec:** `docs/superpowers/specs/2026-09-08-qwen-pes2o-downstream-evaluation-design.md`

## Global Constraints

- Evaluate only the built-in `sciq` task.
- Evaluate variants in this order: `raw`, `minhashlsh`, `lshbloom`.
- Use zero-shot likelihood scoring with no chat template.
- Use CUDA FP16, batch size 8, and seed 42 by default.
- Run exactly 10 examples for smoke and use `limit=None` for the full test split.
- Treat `acc_norm,none` as primary and `acc,none` as secondary.
- Never print Colab secret values.
- Download and load one S3 checkpoint at a time, then delete its temporary copy.
- Log all results to one W&B comparison run and artifact.

---

### Task 1: SciQ Result Validation and Comparison

**Files:**
- Create: `src/sciq_evaluation.py`
- Create: `tests/test_sciq_evaluation.py`

**Interfaces:**
- Consumes: the dictionary returned by `lm_eval.simple_evaluate()`.
- Produces: `to_jsonable(value: object) -> object`.
- Produces: `extract_sciq_metrics(result: dict, variant: str, stage: str, elapsed_seconds: float, peak_cuda_bytes: int) -> dict`.
- Produces: `build_comparison(rows: list[dict]) -> list[dict]`.
- Produces: `write_json(path: str | Path, value: object) -> Path`.

- [ ] **Step 1: Write failing metric extraction and validation tests**

Create synthetic harness output with the exact v0.4.13 keys:

```python
RESULT = {
    "results": {
        "sciq": {
            "acc,none": 0.42,
            "acc_stderr,none": 0.01,
            "acc_norm,none": 0.48,
            "acc_norm_stderr,none": 0.02,
            "alias": "sciq",
        }
    },
    "n-samples": {"sciq": {"original": 1000, "effective": 1000}},
}

row = extract_sciq_metrics(RESULT, "raw", "full", 12.5, 1024)
assert row["acc_norm"] == 0.48
assert row["examples"] == 1000
```

Test rejection of an unknown variant, unknown stage, missing metric,
non-finite metric, non-positive elapsed time, and an effective sample count
other than 10 for smoke or 1,000 for full.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python3 -m unittest tests.test_sciq_evaluation -v`

Expected: FAIL because `src.sciq_evaluation` does not exist.

- [ ] **Step 3: Implement extraction and safe JSON conversion**

Define exact constants and validate every value:

```python
VARIANTS = ("raw", "minhashlsh", "lshbloom")
STAGES = {"smoke": 10, "full": 1000}

def extract_sciq_metrics(result, variant, stage, elapsed_seconds, peak_cuda_bytes):
    metrics = result["results"]["sciq"]
    effective = result["n-samples"]["sciq"]["effective"]
    # Require finite acc, acc_norm, their stderrs, and the exact stage size.
    return {
        "variant": variant,
        "stage": stage,
        "examples": effective,
        "acc": float(metrics["acc,none"]),
        "acc_stderr": float(metrics["acc_stderr,none"]),
        "acc_norm": float(metrics["acc_norm,none"]),
        "acc_norm_stderr": float(metrics["acc_norm_stderr,none"]),
        "elapsed_seconds": float(elapsed_seconds),
        "peak_cuda_bytes": int(peak_cuda_bytes),
    }
```

`to_jsonable` recursively converts dataclasses, mappings, sequences, NumPy
scalars and arrays, paths, and other unsupported values into JSON-safe data.
`write_json` creates the parent directory and writes UTF-8, indented JSON.

- [ ] **Step 4: Write failing Raw-relative comparison tests**

Verify that output order is Raw, MinHashLSH, LSHBloom; Raw deltas are exactly
zero; the other deltas use Raw; missing variants raise an error; duplicate
variants raise an error; and input rows are not mutated.

```python
comparison = build_comparison([lshbloom_row, raw_row, minhash_row])
assert [row["variant"] for row in comparison] == list(VARIANTS)
assert comparison[0]["delta_acc_norm"] == 0.0
assert comparison[1]["delta_acc_norm"] == minhash_row["acc_norm"] - raw_row["acc_norm"]
```

- [ ] **Step 5: Implement comparison and run Task 1 tests**

Copy each input row before adding `delta_acc` and `delta_acc_norm`. Require one
full-stage row for every variant and return rows in `VARIANTS` order.

Run: `python3 -m unittest tests.test_sciq_evaluation -v`

Expected: all Task 1 tests PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/sciq_evaluation.py tests/test_sciq_evaluation.py
git commit -m "feat: add SciQ result comparison helpers"
```

### Task 2: Standalone SciQ Colab Notebook

**Files:**
- Create: `scripts/build_sciq_notebook.py`
- Create: `tests/test_sciq_notebook.py`
- Create: `notebooks/qwen_pes2o_sciq_evaluation.ipynb`

**Interfaces:**
- Consumes: exact source text from `src/sciq_evaluation.py`.
- Produces: a deterministic notebook containing dependency installation, fixed configuration, secret loading, S3 validation/download, SciQ evaluation, cleanup, W&B logging, and result export.

- [ ] **Step 1: Write failing notebook structure tests**

Load the notebook as JSON, compile every code cell, and assert it contains:

```python
required = (
    '"lm-eval[hf]==0.4.13"',
    '"task": "sciq"',
    '"num_fewshot": 0',
    '"batch_size": 8',
    '"smoke_limit": 10',
    '"full_limit": None',
    '"dtype": "float16"',
    'apply_chat_template=False',
    'userdata.get("AWS_ACCESS_KEY_ID")',
    'userdata.get("AWS_SECRET_ACCESS_KEY")',
    'userdata.get("WANDB_API_KEY")',
    'simple_evaluate(',
    'tasks=[CONFIG["task"]]',
    'wandb.Artifact(',
)
```

Assert all three exact S3 prefixes occur, no task name other than `sciq` is
configured, secret variables are never passed to `print`, the embedded helper
equals the source module, and the generated notebook is deterministic.

- [ ] **Step 2: Run the notebook test and verify it fails**

Run: `python3 -m unittest tests.test_sciq_notebook -v`

Expected: FAIL because the generator and notebook do not exist.

- [ ] **Step 3: Build dependency, configuration, and secret cells**

Generate notebook v4 JSON using the same `markdown_cell` and `code_cell`
pattern as the existing notebook builders. Install:

```python
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "lm-eval[hf]==0.4.13", "wandb>=0.21,<1", "boto3>=1.35,<2",
])
```

Configure the exact checkpoint mapping, SciQ-only settings, `/content` paths,
and W&B names. Read required secrets with a helper that reports only the secret
name when missing. Use `ap-northeast-1` when `AWS_DEFAULT_REGION` is absent.

- [ ] **Step 4: Add S3 preflight and one-checkpoint download logic**

List all objects under each prefix and save `Key`, `Size`, and `ETag`. Require
`config.json`, `model.safetensors`, `tokenizer_config.json`, and
`tokenizer.json`. Download paths relative to the prefix and reject keys that
would escape the target directory.

```python
def download_checkpoint(variant, local_dir):
    objects = checkpoint_objects[variant]
    for item in objects:
        relative = Path(item["Key"]).relative_to(CHECKPOINTS[variant])
        destination = local_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, item["Key"], str(destination))
```

- [ ] **Step 5: Add identical smoke and full SciQ evaluation**

For each variant, download the checkpoint and call:

```python
result = simple_evaluate(
    model="hf",
    model_args={"pretrained": str(local_dir), "dtype": "float16"},
    tasks=[CONFIG["task"]],
    num_fewshot=CONFIG["num_fewshot"],
    batch_size=CONFIG["batch_size"],
    device="cuda:0",
    limit=limit,
    bootstrap_iters=CONFIG["bootstrap_iters"],
    log_samples=True,
    apply_chat_template=False,
    random_seed=CONFIG["seed"],
    numpy_random_seed=CONFIG["seed"],
    torch_random_seed=CONFIG["seed"],
    fewshot_random_seed=CONFIG["seed"],
)
```

Run with limit 10, validate and save the smoke result, then run with limit None,
validate and save the full result. Catch `torch.cuda.OutOfMemoryError` and raise
a message containing the active batch size. In `finally`, run garbage
collection, clear the CUDA cache, and remove the temporary checkpoint.

- [ ] **Step 6: Add comparison, W&B artifact, and failure output**

Build the three full rows with `build_comparison`, write `comparison.csv` and
`summary.json`, display the table, and log one W&B table. Write `failures.json`
if any stage fails. Always upload the result directory and call `wandb.finish()`
inside the outer `finally` block.

- [ ] **Step 7: Generate the notebook and run notebook tests**

Run: `python3 scripts/build_sciq_notebook.py`

Run: `python3 -m unittest tests.test_sciq_notebook -v`

Expected: notebook is created and all Task 2 tests PASS.

- [ ] **Step 8: Commit Task 2**

```bash
git add scripts/build_sciq_notebook.py tests/test_sciq_notebook.py notebooks/qwen_pes2o_sciq_evaluation.ipynb
git commit -m "feat: add three-model SciQ Colab evaluation"
```

### Task 3: Complete Verification

**Files:**
- Modify only if verification reveals a concrete defect.

**Interfaces:**
- Consumes: the helper module, generator, notebook, and all tests.
- Produces: a reproducible, locally validated Colab artifact.

- [ ] **Step 1: Regenerate and prove the notebook is deterministic**

Run: `python3 scripts/build_sciq_notebook.py`

Run: `git diff --exit-code -- notebooks/qwen_pes2o_sciq_evaluation.ipynb`

Expected: both commands exit 0.

- [ ] **Step 2: Run the complete local test suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: all tests pass. GPU, AWS, SciQ download, and W&B behavior are verified
by the notebook smoke stage because they require the Colab environment.

- [ ] **Step 3: Check source and notebook for unsafe secret handling**

Run:

```bash
rg -n 'print\([^\n]*(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|WANDB_API_KEY)' scripts src notebooks
```

Expected: no matches.

- [ ] **Step 4: Inspect the final change set**

Run: `git status --short`

Expected: no uncommitted implementation files; pre-existing ignored local
caches and experimental data do not appear.
