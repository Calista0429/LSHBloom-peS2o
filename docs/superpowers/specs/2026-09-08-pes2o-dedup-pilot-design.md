# peS2o Deduplication Pilot Design

## Goal

Create reproducible Raw, MinHashLSH, and LSHBloom variants from the existing 5,000-document peS2o training sample, upload them to S3, and decide whether the sample contains enough duplication to justify three continued-pretraining runs.

## Fixed experiment settings

- Input: `data/train-5000.jsonl.gz`, copied from peS2o V2 `s2orc/train`.
- Text representation: a set of lowercase whitespace-delimited unigrams.
- MinHash: `datasketch==2.0.0`, 256 permutations, deterministic seed 1.
- Duplicate threshold: Jaccard similarity 0.5.
- Streaming policy: retain the first record and discard each later record reported as a duplicate.
- LSHBloom: expected cardinality 5,000 and effective Bloom false-positive overhead `1e-10`.
- Per-filter false-positive rate: derive from the actual number of LSH bands with `1 - (1 - 1e-10) ** (1 / bands)`.
- Model tokenizer for counts: `Qwen/Qwen2.5-0.5B`, no added special tokens, plus one EOS token per document.
- Training gate: continue to the 25-million-token training comparison only if at least one deduplicated variant removes 5% or more documents; otherwise enlarge the natural sample to 50,000 documents first.

The unigram size, threshold, permutation count, and effective error setting follow the paper's best MinHash settings and its peS2o scaling experiment. The current datasketch release is pinned because version 2.0 changed its default MinHash permutation scheme.

## Outputs

Local output directory: `data/experiments/pilot-5000/`.

- `raw/train.jsonl.gz`
- `minhashlsh/train.jsonl.gz`
- `lshbloom/train.jsonl.gz`
- `manifest.json`

S3 prefix: `s3://calista-bucket/pes2o/v2/experiments/pilot-5000/` with the same relative paths.

The manifest records input identity, parameters, document counts, removal rates, retained and removed IDs, agreement between both methods, compressed file hashes and sizes, token counts when a tokenizer is available, runtime, and the training-gate decision.

## Safety and reproducibility

The script validates every input record, rejects duplicate IDs, writes to temporary files, and atomically replaces final outputs. It never edits the source sample. S3 upload runs only after local validation and SHA-256 calculation. AWS credentials remain in the local AWS CLI configuration and are never written to code or manifests.

