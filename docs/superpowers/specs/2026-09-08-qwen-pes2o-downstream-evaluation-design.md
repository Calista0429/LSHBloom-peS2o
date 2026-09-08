# Qwen peS2o Deduplication Downstream Evaluation Design

## Goal

Evaluate whether continued pretraining on Raw, MinHashLSH-deduplicated, or
LSHBloom-deduplicated peS2o data changes the quality of the resulting
Qwen2.5-0.5B Base model. The comparison must keep the evaluation data,
checkpoint loading, prompts, few-shot settings, batch settings, and metric
aggregation identical across all three models.

This is a pilot experiment. Each checkpoint saw 24,999,936 training tokens, so
differences between the three models can be compared directly, but small score
differences should be treated as uncertain rather than conclusive.

## Checkpoints

The notebook evaluates these immutable S3 prefixes:

- `s3://calista-bucket/pes2o/v2/experiments/pilot-5000/checkpoints/raw/final/`
- `s3://calista-bucket/pes2o/v2/experiments/pilot-5000/checkpoints/minhashlsh/final/`
- `s3://calista-bucket/pes2o/v2/experiments/pilot-5000/checkpoints/lshbloom/final/`

Each prefix contains a Transformers-compatible model and tokenizer, including
`config.json`, `model.safetensors`, and tokenizer files. The original
`Qwen/Qwen2.5-0.5B` checkpoint is an optional fourth reference. It is disabled
by default because the primary experiment compares the three equally trained
checkpoints.

## Evaluation Stages

### Stage 1: Smoke Test

Before a long run, evaluate a small deterministic slice of every task for all
three checkpoints. This verifies AWS credentials, checkpoint integrity, CUDA
inference, task availability, metric serialization, and W&B logging.

The smoke run uses the first 10 examples per task. Its scores are only pipeline
checks and must never be reported as experimental results.

### Stage 2: peS2o Validation Perplexity

Measure causal-language-model perplexity on the official peS2o V2 validation
split, which is disjoint from the 5,000 training documents used in this pilot.
Use the tokenizer stored with each checkpoint, tokenize without automatically
adding special tokens, append one EOS token per document, and pack each source
independently into 2,048-token sequences.

Report S2ORC, S2AG, and overall loss and perplexity. Overall perplexity is
computed from total negative log-likelihood divided by the total number of
predicted tokens; it is not the arithmetic mean of source perplexities.

Default scope is the deterministic 1,000-document validation sample already
defined for the project: 320 S2ORC documents and 680 S2AG documents. A single
configuration switch enables the complete validation split later.

### Stage 3: Downstream Tasks

Use EleutherAI `lm-evaluation-harness` with the Hugging Face backend. Evaluate
base-model likelihoods without a chat template and without generation sampling.

Run these tasks at zero-shot:

- `sciq`: science question answering and the most directly relevant task.
- `arc_easy` and `arc_challenge`: grade-school science questions.
- `openbookqa`: science facts plus reasoning.
- `hellaswag`: sentence-completion and commonsense reasoning.
- `piqa`: physical commonsense.
- `winogrande`: pronoun and commonsense reasoning.
- `boolq`: reading comprehension.

Run the `mmlu` group separately with five-shot examples to match the convention
used by DCLM. MMLU is a secondary result for this 0.5B pilot because a small
model may remain close to random-guess performance.

The harness version is pinned in the notebook. Before evaluation, the notebook
lists or validates the requested task names so a library update cannot silently
change the requested suite. The task list follows the official
[`lm-evaluation-harness` task catalog](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md).

## Metrics

Preserve every metric returned by the harness, including its standard errors.
Use the following as the primary values in the comparison table:

- `acc_norm` for multiple-choice tasks when available.
- `acc` when normalized accuracy is not defined.
- MMLU macro-average accuracy from the task group.
- peS2o negative-log-likelihood loss and perplexity.

For every task, report both the absolute score and the change relative to Raw:

```text
delta(method, task) = score(method, task) - score(raw, task)
```

Also compute an unweighted mean over the eight zero-shot task-level primary
scores. Do not mix MMLU or perplexity into that mean because they have different
scales and meanings.

The final interpretation uses the reported standard errors. A tiny score
difference smaller than roughly two combined standard errors is labeled
inconclusive. The notebook does not claim statistical significance from one
continued-pretraining run per dataset.

## Colab Workflow

The deliverable is one standalone Colab notebook plus small locally tested
helper modules used to generate it. A user runs it from top to bottom:

1. Select a V100 GPU runtime.
2. Store `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional
   `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`, and `WANDB_API_KEY` in Colab
   userdata.
3. Install pinned dependencies.
4. Read secrets with `google.colab.userdata.get`; never print them.
5. Download one checkpoint prefix from S3 to temporary Colab storage.
6. Run the smoke test or full evaluation for that checkpoint.
7. Save raw results, release GPU memory, and delete the temporary local model.
8. Repeat for the next checkpoint.
9. Build the cross-model comparison table and upload all outputs to W&B.

Processing one checkpoint at a time keeps Colab disk and GPU use low. Downloaded
models are disposable local copies; the S3 checkpoints remain unchanged.

## W&B and Saved Outputs

Create one W&B run for the complete comparison. Log:

- checkpoint S3 prefix and variant;
- checkpoint file metadata;
- evaluation library and package versions;
- GPU, dtype, batch size, task names, few-shot settings, limits, and seeds;
- per-task metrics and standard errors;
- per-source and overall peS2o perplexity;
- elapsed time and peak CUDA memory;
- cross-model score deltas relative to Raw.

Write outputs under `/content/results/`:

- one raw harness JSON file per model and stage;
- one perplexity JSON file per model;
- `comparison.csv` for easy plotting and inspection;
- `summary.json` containing configuration, primary scores, deltas, and run
  status;
- `failures.json` if a model or task fails.

Upload the directory as one W&B artifact. A failure in one model is recorded
and shown clearly; it must not be converted into a zero score.

## Reproducibility and Fairness Controls

- Evaluate checkpoint variants in fixed order: Raw, MinHashLSH, LSHBloom.
- Use seed 42 everywhere supported by the harness.
- Use CUDA FP16 inference and a fixed batch size that fits a V100.
- Do not update model parameters or create an optimizer.
- Do not apply a chat template because these are base-model checkpoints.
- Use identical task data and few-shot examples for every model.
- Record exact dependency versions and checkpoint object metadata.
- Never use the 5,000 training documents for perplexity evaluation.
- Keep task-level results; do not rely only on a single average.

## Validation

Local tests verify that the notebook is valid JSON, every code cell compiles,
the S3 paths and task settings are correct, secrets are not printed, aggregate
perplexity is token-weighted, Raw deltas are zero, missing metrics remain
missing rather than becoming zero, and the generated notebook is deterministic.

The Colab smoke stage then verifies the parts that require a V100, AWS, Hugging
Face datasets, and W&B before the full evaluation begins.

## Acceptance Criteria

The evaluation is complete when:

1. All three S3 checkpoints load successfully and are evaluated under identical
   settings.
2. The 1,000-document peS2o validation perplexity is available for every model.
3. All eight zero-shot tasks and five-shot MMLU complete for every model.
4. W&B contains configuration, per-task metrics, standard errors, timing, and
   final comparison tables.
5. The result artifact contains raw JSON outputs, `comparison.csv`, and
   `summary.json`.
6. Conclusions distinguish clear changes from differences too small to resolve
   with this single-run pilot.
