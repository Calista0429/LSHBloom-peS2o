# peS2o Deduplication Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, run, and upload the 5,000-document Raw/MinHashLSH/LSHBloom data-preparation pilot.

**Architecture:** A small core module owns token normalization, Bloom error conversion, streaming decisions, atomic JSONL output, and manifest construction. A command-line script connects the core module to datasketch and optional Qwen tokenizer counts. Unit tests use tiny real MinHash indexes; an integration run processes the existing sample and the AWS CLI uploads verified artifacts.

**Tech Stack:** Python 3.9+, datasketch 2.0.0 with Bloom extras, standard-library unittest, transformers tokenizer, AWS CLI.

**Spec:** `docs/superpowers/specs/2026-09-08-pes2o-dedup-pilot-design.md`

## Global Constraints

- Use lowercase whitespace-delimited unigrams.
- Use threshold 0.5, 256 permutations, seed 1.
- Use effective LSHBloom false-positive overhead `1e-10` and derive the per-filter value from the actual band count.
- Preserve the first document in stream order.
- Do not overwrite or modify the source JSONL file.
- Do not write AWS credentials into files.

---

### Task 1: Core parameter and deduplication behavior

**Files:**
- Create: `src/pes2o_dedup.py`
- Create: `tests/test_pes2o_dedup.py`
- Create: `requirements-dedup.txt`

**Interfaces:**
- Produces: `normalize_unigrams(text)`, `per_filter_fp(effective_fp, bands)`, `make_minhash(text, num_perm, seed)`, and `stream_decisions(records, minhash_index, bloom_index, ...)`.

- [ ] Write failing tests showing normalization, error conversion, first-record retention, exact-duplicate removal, and agreement of real MinHashLSH and LSHBloom indexes.
- [ ] Run `python -m unittest tests.test_pes2o_dedup -v` and confirm failure because the module is absent.
- [ ] Implement the smallest core functions that satisfy those tests.
- [ ] Run the focused tests and the existing full suite.
- [ ] Commit the tested core behavior.

### Task 2: Files, manifest, and command-line interface

**Files:**
- Modify: `src/pes2o_dedup.py`
- Create: `scripts/prepare_pes2o_variants.py`
- Modify: `tests/test_pes2o_dedup.py`

**Interfaces:**
- Consumes: Task 1 core functions.
- Produces: atomic gzip JSONL outputs and a machine-readable `manifest.json` from `prepare_variants(input_path, output_dir, config, tokenizer=None)`.

- [ ] Write failing tests for valid gzip outputs, removal IDs, agreement metrics, hashes, and the 5% training gate.
- [ ] Run the focused tests and confirm the expected missing-interface failure.
- [ ] Implement atomic writers, manifest construction, optional tokenizer counting, and CLI arguments.
- [ ] Run focused and full tests.
- [ ] Commit the runnable preparation tool.

### Task 3: Real 5,000-document preparation

**Files:**
- Create outside Git: `data/experiments/pilot-5000/**`

**Interfaces:**
- Consumes: `/Users/huangbaoxi/Documents/ChatGPT/LSHBloom/data/train-5000.jsonl.gz`.
- Produces: the four local artifacts defined by the design spec.

- [ ] Create an isolated virtual environment and install `requirements-dedup.txt`.
- [ ] Run the CLI against the real sample with Qwen token counting enabled.
- [ ] Validate gzip streams, document IDs, SHA-256 hashes, counts, Bloom error conversion, and manifest schema.
- [ ] Apply the 5% training gate and record the decision.

### Task 4: S3 upload and remote verification

**Files:**
- No tracked files.

**Interfaces:**
- Consumes: verified local pilot artifacts.
- Produces: `s3://calista-bucket/pes2o/v2/experiments/pilot-5000/`.

- [ ] Upload the three JSONL files and manifest with `aws s3 cp`.
- [ ] List the remote prefix and compare remote object sizes with local sizes.
- [ ] Download the remote manifest to a temporary path and compare its SHA-256 hash with the local manifest.

### Task 5: Full verification and next experiment decision

**Files:**
- Modify only if verification exposes a defect, following a new failing test.

**Interfaces:**
- Consumes: local tests, local artifacts, and remote inventory.
- Produces: a concrete decision to train on 5,000 documents or enlarge to 50,000.

- [ ] Run the complete unittest suite from a clean process.
- [ ] Confirm the Git worktree contains only intended tracked changes.
- [ ] Report actual counts, removal rates, method agreement, token counts, S3 locations, and the next action.
