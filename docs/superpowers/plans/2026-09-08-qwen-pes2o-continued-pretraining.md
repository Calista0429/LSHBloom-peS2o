# Qwen peS2o Continued-Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a tested Colab notebook for three equal-token Qwen2.5-0.5B continued-pretraining runs with W&B, S3 checkpoints, and automatic validation perplexity.

**Architecture:** A tested training-data helper streams gzip JSONL into a fixed-size uint32 memory map, preserving document boundaries with EOS. A deterministic notebook generator embeds that helper and the already-tested perplexity helper into a Colab notebook, while separate notebook-structure tests enforce the experiment settings and safety checks.

**Tech Stack:** Python, NumPy memmap, PyTorch, Transformers Trainer, boto3, W&B, Colab Secrets.

**Spec:** `docs/superpowers/specs/2026-09-08-qwen-pes2o-continued-pretraining-design.md`

## Global Constraints

- Train exactly 12,207 full 2,048-token sequences for every data variant.
- Require a CUDA V100-compatible FP16 path and run a smoke test first.
- Read AWS and W&B credentials only from Colab Secrets.
- Verify downloaded dataset SHA-256 against the S3 manifest.
- Upload final checkpoint and result files under the variant-specific S3 prefix.
- Reuse the exact 320 S2ORC plus 680 S2AG validation collection logic.

---

### Task 1: Fixed-token packed training dataset

**Files:**
- Create: `src/pes2o_training.py`
- Create: `tests/test_pes2o_training.py`

**Interfaces:**
- Produces: `pack_jsonl_gz_to_memmap(input_path, output_path, tokenizer, sequence_length, sequence_count)` and `TokenMemmapDataset(path, sequence_length, sequence_count)`.

- [ ] Write failing tests for EOS-separated packing, exact full-sequence count, insufficient-token failure, metadata, and dataset item tensors.
- [ ] Run the focused tests and confirm the module is absent.
- [ ] Implement the minimal streaming packer and read-only dataset.
- [ ] Run focused and full test suites.
- [ ] Commit the training-data helper.

### Task 2: Deterministic Colab notebook

**Files:**
- Create: `scripts/build_training_notebook.py`
- Create: `notebooks/qwen_pes2o_continued_pretraining.ipynb`
- Create: `tests/test_training_notebook.py`

**Interfaces:**
- Consumes: `src/pes2o_training.py` and `src/pes2o_perplexity.py`.
- Produces: a Colab notebook with configuration, secret loading, S3 download verification, packing, smoke test, training, checkpoint upload, and validation PPL.

- [ ] Write failing notebook tests for valid JSON, compilable code cells, exact experiment constants, three allowed variants, Colab Secrets, W&B grouping, checksum checks, S3 output, smoke test, and embedded tested helpers.
- [ ] Run the focused tests and confirm the notebook/generator is absent.
- [ ] Implement the generator and create the notebook.
- [ ] Run notebook tests and regenerate to prove deterministic output.
- [ ] Run the full test suite and commit.

### Task 3: Acceptance verification

**Files:**
- Modify only when a failing acceptance check exposes a defect.

**Interfaces:**
- Consumes: generated notebook and S3 manifest.
- Produces: a ready-to-run notebook and simple three-run instructions.

- [ ] Compile every notebook code cell in a clean Python process.
- [ ] Check calculated sequence/token/update counts.
- [ ] Confirm every variant in the remote manifest contains enough tokens.
- [ ] Confirm the Git worktree is clean after commits.
- [ ] Report the notebook path, first setting to edit, expected runtime sequence, and result locations.
