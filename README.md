# LSHBloom Deduplication Experiment on peS2o

This project studies how different text deduplication methods affect continued
pretraining and downstream model performance. The same Qwen2.5-0.5B Base model
was continued-pretrained on three versions of a 5,000-document peS2o sample:

1. Raw data without deduplication
2. Data deduplicated with MinHashLSH
3. Data deduplicated with LSHBloom

All three models were trained with the same token budget and hyperparameters.
The resulting checkpoints were evaluated on the complete SciQ test split.

## Experimental Pipeline

```text
5,000 peS2o documents
        |
        +-- Raw -----------> Qwen checkpoint ----+
        +-- MinHashLSH ----> Qwen checkpoint ----+--> SciQ evaluation
        +-- LSHBloom ------> Qwen checkpoint ----+
```

The experiment changes only the training-data deduplication method. Model
architecture, starting weights, training tokens, optimization settings, and
evaluation settings remain fixed.

## Dataset Variants

| Variant | Documents retained | Documents removed | Removal rate | Available tokens |
|---|---:|---:|---:|---:|
| Raw | 5,000 | 0 | 0.00% | 31,410,586 |
| MinHashLSH | 4,617 | 383 | 7.66% | 28,805,685 |
| LSHBloom | 4,611 | 389 | 7.78% | 28,752,978 |

MinHashLSH and LSHBloom made the same keep-or-remove decision for 4,994
of the 5,000 documents. They differed on only six documents, corresponding to
99.88% decision agreement.

Deduplication settings:

- `datasketch==2.0.0`
- Jaccard threshold: `0.5`
- MinHash permutations: `256`
- LSHBloom requested effective false-positive rate: `1e-10`
- Text representation: lowercased whitespace-token unigram sets
- Seed: `1`

## Continued Pretraining

Each dataset variant was used to continue pretraining
`Qwen/Qwen2.5-0.5B` under identical conditions:

- Input tokens: `24,999,936`
- Sequence length: `2,048`
- Training sequences: `12,207`
- Training epochs: `1`
- Learning rate: `5e-5`
- Gradient accumulation steps: `8`
- Training seed: `42`
- Mixed precision: FP16

The final training losses were 2.3604 for Raw, 2.3620 for MinHashLSH, and
2.3614 for LSHBloom.

## SciQ Evaluation

The three final checkpoints were evaluated with
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
version `0.4.13`.

- Task: `sciq`
- Evaluation split: complete test split
- Test examples: `1,000`
- Few-shot examples: `0`
- Chat template: disabled
- Model precision: FP16
- Batch size: `8`
- Evaluation seed: `42`

SciQ presents four candidate answers for each science question. `acc` selects
the answer with the highest total likelihood. `acc_norm` normalizes likelihood
by answer length and is the primary metric in this experiment.

## Results

| Training data | Accuracy | Accuracy SE | Length-normalized accuracy | Normalized accuracy SE | Normalized change vs. Raw |
|---|---:|---:|---:|---:|---:|
| Raw | 89.7% | 0.96% | **88.1%** | 1.02% | 0.0 pp |
| MinHashLSH | **90.2%** | 0.94% | 86.9% | 1.07% | -1.2 pp |
| LSHBloom | 90.0% | 0.95% | 86.9% | 1.07% | -1.2 pp |

`pp` means percentage points. The full evaluation corresponds to approximately
897, 902, and 900 correct answers under unnormalized accuracy, and 881, 869,
and 869 correct answers under length-normalized accuracy.

## Interpretation

MinHashLSH and LSHBloom produced effectively identical SciQ performance. Their
length-normalized accuracies are equal, and their unnormalized accuracies differ
by only 0.2 percentage points. This is consistent with the two methods making
different retention decisions for only six of the 5,000 input documents.

Neither deduplicated checkpoint shows a clear improvement over Raw on the
primary metric. Both are 1.2 percentage points below Raw, while the reported
standard errors are approximately one percentage point per model. The aggregate
scores alone do not establish that Raw is better: the observed difference is
small relative to the uncertainty and may reflect training or evaluation
variation.

These results support a limited conclusion: at this pilot scale, LSHBloom
preserves downstream SciQ quality as well as MinHashLSH. They do not show that
deduplication harms model quality or that one deduplication method is generally
superior.

## Reproducing the SciQ Evaluation in Colab

Open
[`notebooks/qwen_pes2o_sciq_evaluation.ipynb`](notebooks/qwen_pes2o_sciq_evaluation.ipynb)
in Google Colab and select a V100 GPU runtime.

Add these values to Colab Secrets:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `WANDB_API_KEY`
- `AWS_SESSION_TOKEN` when using temporary AWS credentials
- `AWS_DEFAULT_REGION` optionally; the notebook defaults to `ap-northeast-1`

Run every cell from top to bottom. For each checkpoint, the notebook:

1. Verifies the required files in S3.
2. Downloads a temporary local copy.
3. Runs a 10-example smoke test.
4. Evaluates all 1,000 SciQ test examples.
5. Saves the raw metrics and per-example outputs.
6. Deletes the temporary checkpoint before loading the next model.

Results are written to `/content/results/sciq/` and uploaded to W&B as one
evaluation artifact. The final `comparison.csv` reports both absolute scores
and changes relative to Raw.

## Repository Structure

- `notebooks/qwen_pes2o_sciq_evaluation.ipynb`: complete Colab evaluation
- `notebooks/qwen_pes2o_continued_pretraining.ipynb`: continued-pretraining run
- `src/sciq_evaluation.py`: SciQ result validation and comparison helpers
- `src/pes2o_dedup.py`: peS2o deduplication implementation
- `src/pes2o_training.py`: fixed-token training-data packing
- `tests/`: reproducibility and correctness tests

## Limitations and Next Step

This is a small pilot with one training run per dataset variant and one
downstream benchmark. The strongest next analysis is a paired comparison of the
saved per-example SciQ outputs, using paired bootstrap confidence intervals or
McNemar's test. Larger training datasets and multiple training seeds are needed
before making general claims about the effect of deduplication on model quality.
