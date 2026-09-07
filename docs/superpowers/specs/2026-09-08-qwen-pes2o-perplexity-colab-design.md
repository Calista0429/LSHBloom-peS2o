# Qwen2.5-0.5B Base on peS2o Validation: Colab Evaluation Design

## Goal

Create a Google Colab notebook that measures the baseline perplexity of
`Qwen/Qwen2.5-0.5B` on a deterministic 1,000-document sample of peS2o V2
validation and records the evaluation in Weights & Biases (W&B). The notebook
must run on a Colab V100 without updating model parameters.

This first run validates the evaluation pipeline. A later run can set the
document limit to `None` to evaluate the complete peS2o V2 validation split.

## Data

peS2o V2 validation contains two sources:

- `s2orc`: 51,323 full-text papers.
- `s2ag`: 109,709 title-and-abstract documents.

The pilot sample preserves their approximate document ratio:

- 320 documents from `s2orc`.
- 680 documents from `s2ag`.

The notebook reads the official compressed JSONL validation shards from the
`allenai/peS2o` Hugging Face repository. It takes a deterministic prefix from
each source and records every sampled document ID in the result artifact. This
choice avoids downloading the complete validation split for the pilot and makes
the result exactly reproducible. The pilot is a pipeline check rather than an
unbiased estimate over all validation documents; the complete run removes this
sampling limitation.

Each input record must contain non-empty `id`, `source`, and `text` fields, and
its `source` must match the source being evaluated. Invalid records stop the run
with a clear error instead of being silently skipped.

## Tokenization and Packing

Use the tokenizer shipped with `Qwen/Qwen2.5-0.5B`.

For each source independently:

1. Tokenize `text` without automatically adding special tokens.
2. Append one EOS token after every document.
3. Concatenate the token stream.
4. Split it into sequences of 2,048 tokens.
5. Pad the final partial sequence and mask padding labels with `-100` so padding
   never contributes to loss.

Keeping the source streams separate allows the notebook to report independent
S2ORC and S2AG perplexities. Overall perplexity is computed from the combined
negative log-likelihood and predicted-token count, not by averaging the two
perplexity values.

## Model Evaluation

Load `Qwen/Qwen2.5-0.5B` in FP16, move it to the CUDA device, call
`model.eval()`, and run under `torch.inference_mode()`. No optimizer, backward
pass, or parameter update is created.

The model performs standard causal language modeling. For each batch, multiply
the model's mean loss by the exact number of valid next-token labels. Aggregate
negative log-likelihood over all batches, then calculate:

```text
mean_loss = total_negative_log_likelihood / total_predicted_tokens
perplexity = exp(mean_loss)
```

The first token of each packed sequence is not a prediction target because
causal language modeling shifts labels by one position. The notebook counts
predicted tokens after this shift, ensuring the denominator matches the loss.

Default evaluation settings:

- Sequence length: 2,048 tokens.
- Evaluation batch size: 4, with a single configuration value users may change.
- S2ORC documents: 320.
- S2AG documents: 680.
- Seed: 42.

If CUDA is unavailable, the notebook stops with an explicit message. If a batch
causes CUDA out-of-memory, it reports the current batch size and tells the user
to reduce the configuration value before rerunning.

## W&B Tracking

The notebook prompts for W&B authentication through `wandb.login()` and creates
one run with a configurable project name. It records:

- Model and tokenizer identifier.
- Dataset version and source URLs.
- Requested and actual document counts by source.
- Sequence length, batch size, dtype, GPU name, and seed.
- Running loss and perplexity.
- Predicted token count, processed sequence count, throughput, elapsed time, and
  allocated GPU memory.
- Final S2ORC, S2AG, and overall loss and perplexity.

Progress metrics are logged periodically rather than after every batch to avoid
network overhead. The run finishes explicitly even if the evaluation raises an
exception.

## Outputs

The notebook writes a JSON result file under `/content/results/`. It contains:

- Full run configuration.
- Environment and package versions.
- Sampled document IDs.
- Per-source document, sequence, token, loss, perplexity, throughput, and timing
  statistics.
- Correctly token-weighted overall loss and perplexity.

The JSON file is uploaded to the W&B run as an artifact. The notebook also
prints a compact final table so the result remains visible in Colab.

## Validation

Before evaluating the 1,000 documents, the notebook performs a short smoke test
that checks:

- CUDA and FP16 model inference work.
- Tokenized input and shifted target counts are non-zero.
- Loss and perplexity are finite.
- W&B accepts a test progress record.

The delivered `.ipynb` file will also be checked locally for valid notebook JSON
and Python syntax in every code cell. The local check cannot reproduce V100
inference; the Colab smoke test covers that environment-specific behavior.

## Acceptance Criteria

The notebook is ready when it:

1. Opens and executes as a valid Colab notebook.
2. Evaluates exactly 320 S2ORC and 680 S2AG documents by default.
3. Does not calculate gradients or update Qwen weights.
4. Reports finite per-source and overall perplexities using predicted-token
   weighted aggregation.
5. Logs progress and final metrics to W&B.
6. Saves and uploads the result JSON plus sampled document IDs as a W&B artifact.
7. Supports a later complete-validation run through configuration rather than
   code changes.
