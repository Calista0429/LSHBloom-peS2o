# Qwen peS2o Continued-Pretraining Design

## Goal

Measure how Raw, MinHashLSH, and LSHBloom peS2o variants affect Qwen2.5-0.5B Base under equal training compute, then evaluate each checkpoint on the same 1,000-document peS2o validation sample used for the base-model measurement.

## One notebook, three runs

The Colab notebook exposes one required setting, `VARIANT`, with values `raw`, `minhashlsh`, or `lshbloom`. Run the notebook once per variant in a fresh Colab V100 runtime. Each run downloads the matching JSONL gzip and manifest from `s3://calista-bucket/pes2o/v2/experiments/pilot-5000/`, verifies the SHA-256 checksum, trains, uploads its final checkpoint to the same experiment prefix, and evaluates validation perplexity.

AWS and W&B credentials come from Colab Secrets named `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, and `WANDB_API_KEY`. Cells must never print secret values.

## Fixed training settings

- Model: `Qwen/Qwen2.5-0.5B` Base.
- Training input: deterministic stream order, EOS between documents, packed into 2,048-token sequences.
- Budget: 12,207 full sequences = 24,999,936 input tokens for every variant.
- Precision: FP16 on V100.
- Per-device batch: 1 sequence.
- Gradient accumulation: 8, giving 16,384 input tokens per optimizer update except the final partial update.
- Learning rate: `5e-5`, cosine schedule, 3% warmup.
- Optimizer: PyTorch AdamW, weight decay 0.1, gradient clipping 1.0.
- One pass over the fixed packed budget, gradient checkpointing enabled, seed 42.
- Log every 10 optimizer steps and save every 250 steps, retaining the latest two checkpoints.
- W&B project: `lshbloom-pes2o`; group: `qwen2.5-0.5b-pes2o-dedup-25m`.

## Evaluation and outputs

After training, the notebook evaluates 320 S2ORC and 680 S2AG validation documents, with sequence length 2,048 and evaluation batch size 4. It logs token-weighted loss and perplexity by source and overall to the same W&B run.

The final checkpoint, tokenizer, training metrics, and evaluation JSON are uploaded to:

`s3://calista-bucket/pes2o/v2/experiments/pilot-5000/checkpoints/<variant>/`

The notebook performs a one-batch CUDA smoke test before the long training call and validates that every dataset contains at least 24,999,936 tokenizer tokens before training.

