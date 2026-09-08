# Qwen peS2o SciQ Evaluation Design

## Goal

Compare the SciQ performance of the three Qwen2.5-0.5B Base checkpoints that
were continued-pretrained on Raw, MinHashLSH-deduplicated, and
LSHBloom-deduplicated peS2o data. This first downstream experiment evaluates
SciQ only. Other benchmarks and peS2o validation perplexity are outside this
stage.

Each checkpoint saw exactly 24,999,936 training tokens. The evaluation must
therefore keep every other condition identical and change only the checkpoint.

## Checkpoints

Evaluate these S3 prefixes in this fixed order:

1. `s3://calista-bucket/pes2o/v2/experiments/pilot-5000/checkpoints/raw/final/`
2. `s3://calista-bucket/pes2o/v2/experiments/pilot-5000/checkpoints/minhashlsh/final/`
3. `s3://calista-bucket/pes2o/v2/experiments/pilot-5000/checkpoints/lshbloom/final/`

Each prefix is a complete Transformers checkpoint with model weights,
configuration, and tokenizer files. The original Hugging Face Qwen checkpoint
is not included in this first comparison.

## SciQ Protocol

Use `lm-evaluation-harness==0.4.13` and its built-in `sciq` task. SciQ is a
four-choice English science question-answering dataset. The harness evaluates
the public SciQ test split by comparing the likelihood of each answer choice.

Evaluation settings:

- Task: `sciq`
- Test examples: the complete test split
- Few-shot examples: 0
- Model backend: Hugging Face causal language model
- Chat template: disabled
- Device: one CUDA GPU
- Dtype: FP16
- Batch size: 8 by default, configurable in one place
- Random seeds: 42
- Sample logging: enabled

These are base-model checkpoints, so the harness must score answer
continuations directly without a system prompt or chat formatting. No optimizer,
backward pass, or parameter update is created.

Before the full run, the notebook evaluates exactly 10 SciQ test examples for
all three models. These smoke-test scores only validate the pipeline and must
not be presented as experimental results.

The SciQ task definition follows the official
[`lm-evaluation-harness` configuration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/sciq/sciq.yaml).

## Metrics

Preserve all metrics returned by the harness. The final comparison emphasizes:

- `acc_norm`: primary metric; choice likelihood normalized by answer length.
- `acc`: secondary metric; unnormalized answer-choice likelihood.
- `acc_norm_stderr` and `acc_stderr`: uncertainty estimates reported by the
  harness.
- Number of evaluated examples and elapsed time.

For MinHashLSH and LSHBloom, report the difference from Raw:

```text
delta_acc_norm = method_acc_norm - raw_acc_norm
delta_acc = method_acc - raw_acc
```

Raw deltas are exactly zero. Missing or failed scores remain missing and are
never replaced with zero. Because this pilot contains one trained model per
dataset variant, small differences within the reported uncertainty are called
inconclusive.

## Colab Workflow

Deliver one standalone Google Colab notebook. The user runs it from top to
bottom on a V100:

1. Add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional
   `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`, and `WANDB_API_KEY` to Colab
   userdata.
2. Install pinned evaluation dependencies.
3. Read secrets through `google.colab.userdata.get` without printing them.
4. Confirm CUDA is available and record the GPU name.
5. Create one W&B run for the three-model SciQ comparison.
6. Download one checkpoint prefix to temporary Colab storage.
7. Load and evaluate that checkpoint, save its raw results, then release GPU
   memory and remove the temporary local copy.
8. Repeat for the next checkpoint.
9. Create and display the final comparison table.
10. Upload all result files to W&B as one artifact and finish the run.

Sequential processing requires only one roughly 2 GB checkpoint on local disk
and one model in GPU memory at a time. It does not modify S3.

## Outputs

Write the following files under `/content/results/sciq/`:

- `raw-smoke.json`, `minhashlsh-smoke.json`, and `lshbloom-smoke.json`
- `raw-full.json`, `minhashlsh-full.json`, and `lshbloom-full.json`
- `comparison.csv`
- `summary.json`
- `failures.json` only if any stage fails

W&B receives configuration, environment versions, per-model metrics, standard
errors, elapsed time, peak CUDA memory, the final comparison table, and a
single artifact containing the result directory. Smoke metrics use a separate
`smoke/` namespace so they cannot be mistaken for full results.

## Failure Handling

- Missing Colab secrets stop before any checkpoint download begins.
- An incomplete S3 checkpoint reports the missing required files.
- CUDA out-of-memory reports the active batch size and instructs the user to
  reduce `batch_size` and rerun.
- Non-finite or absent SciQ metrics fail that model instead of producing a
  misleading comparison.
- A model failure is written to `failures.json`, uploaded to W&B, and displayed
  clearly.

## Reproducibility Controls

- The same installed package versions remain active for all models.
- The same SciQ task revision, examples, prompt construction, and zero-shot
  setting are used for every checkpoint.
- Evaluation order is fixed as Raw, MinHashLSH, and LSHBloom.
- Seeds are fixed to 42 wherever supported.
- Checkpoint S3 object names, sizes, and ETags are saved in `summary.json`.
- Raw harness output and per-example samples are retained for audit.

## Validation

Local automated tests verify that the notebook is valid JSON, every code cell
compiles, the three S3 paths are exact, only the `sciq` task is requested,
full evaluation uses no limit, smoke evaluation uses 10 examples, zero-shot and
FP16 settings are fixed, secrets are never printed, result deltas are computed
against Raw, and missing metrics remain missing.

The Colab smoke stage verifies AWS access, S3 checkpoint completeness, model
loading, SciQ dataset access, CUDA inference, and W&B logging before the full
test split is evaluated.

## Acceptance Criteria

The stage is complete when:

1. All three checkpoints complete the full SciQ test split under identical
   settings.
2. `acc_norm`, `acc`, both standard errors, and elapsed time are recorded for
   every checkpoint.
3. `comparison.csv` correctly reports each model's score and change from Raw.
4. W&B contains the run configuration, metrics, final table, and result
   artifact.
5. The notebook can be rerun from a fresh Colab V100 session using only the
   configured userdata secrets.
