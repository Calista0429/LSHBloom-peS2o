import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CORE_PATH = ROOT / "src" / "pes2o_training.py"
EVALUATION_CORE_PATH = ROOT / "src" / "pes2o_perplexity.py"
OUTPUT_PATH = ROOT / "notebooks" / "plamo2_1b_pes2o_continued_pretraining.ipynb"


def source_lines(source):
    return source.splitlines(keepends=True)


def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(source)}


def code_cell(source, tags=None):
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source_lines(source),
    }


def build_notebook():
    training_core = TRAINING_CORE_PATH.read_text(encoding="utf-8")
    evaluation_core = EVALUATION_CORE_PATH.read_text(encoding="utf-8")
    cells = [
        markdown_cell(
            """# PLaMo 2 1B: Equal-25M-Token peS2o Continued Pretraining

This notebook trains one data variant per run: `raw`, `minhashlsh`, or `lshbloom`. Every run uses exactly 24,999,936 PLaMo tokenizer tokens, so the training-data deduplication method is the only experimental variable.

PLaMo 2 uses custom Mamba kernels with strict dependency requirements. Before connecting, select **Runtime > Change runtime type**, choose a **V100 GPU**, and choose the **2025.07 past runtime (Python 3.11)**. The setup cell installs PyTorch 2.5.1 and restarts the runtime once. After reconnection, run all cells again.

Add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `WANDB_API_KEY` to Colab Secrets. Temporary AWS credentials also require `AWS_SESSION_TOKEN`. Use a fresh runtime for each variant.
"""
        ),
        code_cell(
            """# Install the exact PLaMo-compatible runtime dependencies.
import importlib.metadata
import os
import subprocess
import sys


REQUIRED = {
    "torch_version": "2.5.1",
    "transformers_version": "4.57.1",
    "accelerate_version": "1.10.1",
    "numpy_version": "1.26.4",
    "numba_version": "0.60.0",
    "mamba_ssm_version": "2.2.4",
    "causal_conv1d_version": "1.4.0",
}

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(
        "Select Colab past runtime 2025.07, which provides Python 3.11, "
        f"then reconnect. Current Python: {sys.version.split()[0]}"
    )

try:
    installed_torch = importlib.metadata.version("torch").split("+")[0]
except importlib.metadata.PackageNotFoundError:
    installed_torch = None

if installed_torch != REQUIRED["torch_version"]:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
        check=False,
    )
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        f"torch=={REQUIRED['torch_version']}",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    ])
    print("PyTorch installed. Colab will restart; reconnect and run all cells again.")
    os.kill(os.getpid(), 9)

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    f"transformers=={REQUIRED['transformers_version']}",
    f"accelerate=={REQUIRED['accelerate_version']}",
    f"numpy=={REQUIRED['numpy_version']}",
    f"numba=={REQUIRED['numba_version']}",
    "wandb==0.21.1",
    "boto3>=1.35,<2",
    "packaging>=24,<26",
    "ninja>=1.11,<2",
    "wheel",
])

# Limit extension compilation memory on the Colab VM.
os.environ["MAX_JOBS"] = "2"
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "--no-build-isolation",
    f"causal-conv1d=={REQUIRED['causal_conv1d_version']}",
    f"mamba-ssm=={REQUIRED['mamba_ssm_version']}",
])
print("PLaMo dependencies installed.")
"""
        ),
        code_cell(training_core, tags=["training-core-library"]),
        code_cell(evaluation_core, tags=["evaluation-core-library"]),
        markdown_cell(
            """## 1. Select one variant and validate the runtime

Change only `VARIANT`. The notebook stops before downloading data if the Python, package, or GPU requirements are wrong.
"""
        ),
        code_cell(
            """import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

import boto3
import causal_conv1d
import mamba_ssm
import numpy as np
import torch
import transformers
import wandb
from google.colab import userdata
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


VARIANT = "raw"
ALLOWED_VARIANTS = {"raw", "minhashlsh", "lshbloom"}

CONFIG = {
    "model_id": "pfnet/plamo-2-1b",
    "model_revision": "92c75fd6eea9018bcb9c33ee8921589febe071fa",
    "required_gpu_substring": "V100",
    "torch_version": "2.5.1",
    "transformers_version": "4.57.1",
    "mamba_ssm_version": "2.2.4",
    "causal_conv1d_version": "1.4.0",
    "sequence_length": 2048,
    "sequence_count": 12_207,
    "train_input_tokens": 24_999_936,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.03,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "fp16": True,
    "seed": 42,
    "save_steps": 250,
    "logging_steps": 10,
    "eval_batch_size": 1,
    "s2orc_documents": 320,
    "s2ag_documents": 680,
    "wandb_project": "lshbloom-pes2o",
    "wandb_group": "plamo-2-1b-pes2o-dedup-25m",
}

if VARIANT not in ALLOWED_VARIANTS:
    raise ValueError(f"VARIANT must be one of {sorted(ALLOWED_VARIANTS)}")
if CONFIG["sequence_length"] * CONFIG["sequence_count"] != CONFIG["train_input_tokens"]:
    raise RuntimeError("Training token budget is internally inconsistent")
if sys.version_info[:2] != (3, 11):
    raise RuntimeError("PLaMo requires the Colab 2025.07 Python 3.11 runtime")
if Version(torch.__version__.split("+")[0]) != Version(CONFIG["torch_version"]):
    raise RuntimeError(f"Expected torch {CONFIG['torch_version']}, received {torch.__version__}")
if Version(transformers.__version__) != Version(CONFIG["transformers_version"]):
    raise RuntimeError(
        f"Expected transformers {CONFIG['transformers_version']}, received {transformers.__version__}"
    )
for distribution, expected in (
    ("mamba-ssm", CONFIG["mamba_ssm_version"]),
    ("causal-conv1d", CONFIG["causal_conv1d_version"]),
):
    actual = importlib.metadata.version(distribution)
    if Version(actual) != Version(expected):
        raise RuntimeError(f"Expected {distribution} {expected}, received {actual}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Colab.")
gpu_name = torch.cuda.get_device_name(0)
if CONFIG["required_gpu_substring"] not in gpu_name:
    raise RuntimeError(f"This experiment requires a V100; received {gpu_name}")

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])


def optional_secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None


try:
    AWS_ACCESS_KEY_ID = userdata.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = userdata.get("AWS_SECRET_ACCESS_KEY")
    WANDB_API_KEY = userdata.get("WANDB_API_KEY")
except Exception as error:
    raise RuntimeError("One or more required Colab secrets are unavailable") from error

for name, value in (
    ("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID),
    ("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY),
    ("WANDB_API_KEY", WANDB_API_KEY),
):
    if not value:
        raise RuntimeError(f"Missing required Colab secret: {name}")

session_kwargs = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "region_name": optional_secret("AWS_DEFAULT_REGION") or "ap-northeast-1",
}
AWS_SESSION_TOKEN = optional_secret("AWS_SESSION_TOKEN")
if AWS_SESSION_TOKEN:
    session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN
s3 = boto3.session.Session(**session_kwargs).client("s3")

S3_BUCKET = "calista-bucket"
S3_PREFIX = "pes2o/v2/experiments/pilot-5000/"
S3_URI = f"s3://{S3_BUCKET}/{S3_PREFIX}"
checkpoint_prefix = f"{S3_PREFIX}plamo-2-1b-25m/checkpoints/{VARIANT}"
WORK_DIR = Path(f"/content/plamo-2-1b-pes2o-{VARIANT}-25m")
DATA_PATH = WORK_DIR / "train.jsonl.gz"
MANIFEST_PATH = WORK_DIR / "manifest.json"
MEMMAP_PATH = WORK_DIR / "train-tokens.uint32"
TRAINER_DIR = WORK_DIR / "trainer"
FINAL_MODEL_DIR = WORK_DIR / "final"
RESULT_PATH = WORK_DIR / "result.json"
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("GPU:", gpu_name)
print("Variant:", VARIANT)
print("PLaMo training tokens:", f"{CONFIG['train_input_tokens']:,}")
print("S3 output:", f"s3://{S3_BUCKET}/{checkpoint_prefix}/")

# Fail before training if this runtime cannot persist its final output.
preflight_key = f"{checkpoint_prefix}/_write-preflight.txt"
try:
    s3.put_object(Bucket=S3_BUCKET, Key=preflight_key, Body=b"ok")
finally:
    s3.delete_object(Bucket=S3_BUCKET, Key=preflight_key)
print("S3 read/write preflight passed.")
"""
        ),
        markdown_cell(
            """## 2. Download the selected dataset and pack exactly 25 million PLaMo tokens

The source manifest token counts were produced with the earlier Qwen tokenizer, so they are recorded only as source metadata. This notebook does not use them to approve the PLaMo budget. The PLaMo tokenizer reads documents in their existing order, inserts EOS between documents, and stops after exactly 12,207 complete 2,048-token sequences. Packing fails if the selected variant does not contain enough PLaMo tokens.
"""
        ),
        code_cell(
            """s3.download_file(S3_BUCKET, f"{S3_PREFIX}manifest.json", str(MANIFEST_PATH))
source_manifest_sha256 = _sha256(MANIFEST_PATH)
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
variant_info = manifest["variants"][VARIANT]

s3.download_file(
    S3_BUCKET,
    f"{S3_PREFIX}{variant_info['relative_path']}",
    str(DATA_PATH),
)
actual_sha256 = _sha256(DATA_PATH)
if actual_sha256 != variant_info["sha256"]:
    raise RuntimeError("Dataset SHA-256 mismatch")

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_id"],
    revision=CONFIG["model_revision"],
    trust_remote_code=True,
)
if tokenizer.eos_token_id is None:
    raise RuntimeError("PLaMo tokenizer has no EOS token")

packing = pack_jsonl_gz_to_memmap(
    input_path=DATA_PATH,
    output_path=MEMMAP_PATH,
    tokenizer=tokenizer,
    sequence_length=CONFIG["sequence_length"],
    sequence_count=CONFIG["sequence_count"],
)
train_dataset = TokenMemmapDataset(
    MEMMAP_PATH,
    sequence_length=CONFIG["sequence_length"],
    sequence_count=CONFIG["sequence_count"],
)


def causal_lm_collator(features):
    input_ids = torch.from_numpy(
        np.stack([feature["input_ids"] for feature in features])
    ).long()
    return {"input_ids": input_ids, "labels": input_ids.clone()}


print(json.dumps(packing, indent=2))
print("Optimizer updates:", math.ceil(len(train_dataset) / CONFIG["gradient_accumulation_steps"]))
"""
        ),
        markdown_cell(
            """## 3. Start W&B and run a full forward/backward smoke test

PLaMo is loaded with its pinned remote model code. Parameters remain FP32 while V100 computation uses FP16 autocast. The smoke test uses a complete 2,048-token sequence and runs backward before the long training starts.
"""
        ),
        code_cell(
            """wandb.login(key=WANDB_API_KEY)
run = wandb.init(
    project=CONFIG["wandb_project"],
    group=CONFIG["wandb_group"],
    name=f"plamo-2-1b-{VARIANT}-25m",
    config={
        **CONFIG,
        "variant": VARIANT,
        "source_manifest_sha256": source_manifest_sha256,
        "dataset_sha256": actual_sha256,
        "source_manifest_token_count": variant_info["token_count"],
        "packing": packing,
        "gpu": gpu_name,
    },
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_id"],
    revision=CONFIG["model_revision"],
    trust_remote_code=True,
    torch_dtype=torch.float32,
).to("cuda")
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.train()

smoke_batch = {
    key: value.to("cuda")
    for key, value in causal_lm_collator([train_dataset[0]]).items()
}
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.float16):
    smoke_output = model(**smoke_batch)
if not torch.isfinite(smoke_output.loss):
    raise RuntimeError(f"Smoke test loss is not finite: {smoke_output.loss.item()}")
smoke_output.loss.backward()
smoke_peak_bytes = torch.cuda.max_memory_allocated()
model.zero_grad(set_to_none=True)
wandb.log({
    "smoke_test/passed": 1,
    "smoke_test/loss": smoke_output.loss.item(),
    "smoke_test/peak_cuda_bytes": smoke_peak_bytes,
})
print(
    f"Smoke test passed: loss={smoke_output.loss.item():.4f}, "
    f"peak GPU memory={smoke_peak_bytes / (1024**3):.2f} GiB"
)
del smoke_batch, smoke_output
gc.collect()
torch.cuda.empty_cache()
"""
        ),
        markdown_cell(
            """## 4. Continue pretraining and persist the final checkpoint

All three variants use the same 24,999,936-token budget and schedule. Adafactor is used for all three PLaMo runs because a full FP32 AdamW state does not fit reliably on a 16 GB V100. The final model is uploaded to S3 immediately after training, before validation begins.
"""
        ),
        code_cell(
            """training_args = TrainingArguments(
    output_dir=str(TRAINER_DIR),
    overwrite_output_dir=True,
    num_train_epochs=1.0,
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    lr_scheduler_type="cosine",
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    max_grad_norm=CONFIG["max_grad_norm"],
    fp16=CONFIG["fp16"],
    gradient_checkpointing=True,
    logging_steps=CONFIG["logging_steps"],
    logging_first_step=True,
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=1,
    save_only_model=True,
    report_to=["wandb"],
    run_name=f"plamo-2-1b-{VARIANT}-25m",
    seed=CONFIG["seed"],
    data_seed=CONFIG["seed"],
    dataloader_num_workers=2,
    remove_unused_columns=False,
    optim="adafactor",
    save_safetensors=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=causal_lm_collator,
    processing_class=tokenizer,
)
train_output = trainer.train()
train_metrics = dict(train_output.metrics)
train_metrics["train_input_tokens"] = CONFIG["train_input_tokens"]
train_metrics["variant"] = VARIANT
trainer.log_metrics("train", train_metrics)
trainer.save_metrics("train", train_metrics)
trainer.save_model(str(FINAL_MODEL_DIR))
tokenizer.save_pretrained(FINAL_MODEL_DIR)

for local_path in FINAL_MODEL_DIR.rglob("*"):
    if local_path.is_file():
        relative = local_path.relative_to(FINAL_MODEL_DIR).as_posix()
        s3.upload_file(
            str(local_path),
            S3_BUCKET,
            f"{checkpoint_prefix}/final/{relative}",
        )
s3.upload_file(
    str(TRAINER_DIR / "train_results.json"),
    S3_BUCKET,
    f"{checkpoint_prefix}/train_results.json",
)
run.summary["checkpoint_s3_uri"] = f"s3://{S3_BUCKET}/{checkpoint_prefix}/final/"
print("Training complete:", json.dumps(train_metrics, indent=2))
print("Checkpoint safely stored at:", run.summary["checkpoint_s3_uri"])
del trainer
gc.collect()
torch.cuda.empty_cache()
"""
        ),
        markdown_cell(
            """## 5. Evaluate the same 1,000-document peS2o validation sample

The fixed sample contains 320 S2ORC documents and 680 S2AG documents. The validation files are pinned to an immutable peS2o revision. Evaluation uses FP16 and batch size 1 for V100 memory safety.
"""
        ),
        code_cell(
            """VALIDATION_REVISION = "636a503e44a3ca1b58e01fb61eab0825cd574de0"
VALIDATION_URLS = [
    f"https://huggingface.co/datasets/allenai/peS2o/resolve/{VALIDATION_REVISION}/data/v2/validation-00000-of-00002.json.gz",
    f"https://huggingface.co/datasets/allenai/peS2o/resolve/{VALIDATION_REVISION}/data/v2/validation-00001-of-00002.json.gz",
]
source_limits = {
    "s2orc": CONFIG["s2orc_documents"],
    "s2ag": CONFIG["s2ag_documents"],
}
records_by_source = collect_source_records(VALIDATION_URLS, source_limits)


def log_eval_progress(metrics):
    prefix = f"eval/{metrics['source']}"
    wandb.log({
        f"{prefix}/batch": metrics["batch"],
        f"{prefix}/running_loss": metrics["loss"],
        f"{prefix}/running_perplexity": metrics["perplexity"],
        f"{prefix}/predicted_tokens": metrics["predicted_tokens"],
        f"{prefix}/tokens_per_second": metrics["tokens_per_second"],
    })


model.to(dtype=torch.float16)
model.eval()
source_results = {}
for source in ("s2orc", "s2ag"):
    source_results[source] = evaluate_source(
        model=model,
        packed_sequences=iter_packed_sequences(
            records_by_source[source], tokenizer, CONFIG["sequence_length"]
        ),
        source=source,
        batch_size=CONFIG["eval_batch_size"],
        device="cuda",
        log_every_steps=25,
        progress_callback=log_eval_progress,
    )
    print(
        f"{source}: loss={source_results[source]['loss']:.4f}, "
        f"PPL={source_results[source]['perplexity']:.4f}"
    )

overall = combine_source_metrics(source_results.values())
wandb.log({
    "eval/overall_loss": overall["loss"],
    "eval/overall_perplexity": overall["perplexity"],
    "eval/overall_predicted_tokens": overall["predicted_tokens"],
    "eval/s2orc_perplexity": source_results["s2orc"]["perplexity"],
    "eval/s2ag_perplexity": source_results["s2ag"]["perplexity"],
})
run.summary["eval/overall_perplexity"] = overall["perplexity"]
run.summary["eval/s2orc_perplexity"] = source_results["s2orc"]["perplexity"]
run.summary["eval/s2ag_perplexity"] = source_results["s2ag"]["perplexity"]
print(f"Overall: loss={overall['loss']:.4f}, PPL={overall['perplexity']:.4f}")
"""
        ),
        markdown_cell(
            """## 6. Save the reproducibility record to S3 and W&B

The result JSON records the pinned model and validation revisions, exact PLaMo packing hash, source file hash, software versions, training metrics, and validation metrics.
"""
        ),
        code_cell(
            """result = {
    "schema_version": 1,
    "variant": VARIANT,
    "config": CONFIG,
    "model": {
        "id": CONFIG["model_id"],
        "revision": CONFIG["model_revision"],
        "trust_remote_code": True,
    },
    "dataset": {
        "s3_uri": f"{S3_URI}{variant_info['relative_path']}",
        "source_file_sha256": actual_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_token_count_qwen_tokenizer": variant_info["token_count"],
        "plamo_packing": packing,
    },
    "training": train_metrics,
    "validation": {
        "dataset_revision": VALIDATION_REVISION,
        "urls": VALIDATION_URLS,
        "sources": source_results,
        "overall": overall,
    },
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mamba_ssm": importlib.metadata.version("mamba-ssm"),
        "causal_conv1d": importlib.metadata.version("causal-conv1d"),
        "wandb": wandb.__version__,
        "gpu": gpu_name,
    },
    "wandb_run_id": run.id,
    "wandb_run_url": run.url,
}
RESULT_PATH.write_text(
    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\\n",
    encoding="utf-8",
)
s3.upload_file(str(RESULT_PATH), S3_BUCKET, f"{checkpoint_prefix}/result.json")

artifact = wandb.Artifact(
    name=f"plamo-2-1b-{VARIANT}-25m-results",
    type="evaluation",
    metadata={
        "variant": VARIANT,
        "train_input_tokens": CONFIG["train_input_tokens"],
        "model_revision": CONFIG["model_revision"],
    },
)
artifact.add_file(str(RESULT_PATH))
run.log_artifact(artifact)
print("W&B:", run.url)
print("Checkpoint:", run.summary["checkpoint_s3_uri"])
print("Result:", f"s3://{S3_BUCKET}/{checkpoint_prefix}/result.json")
wandb.finish()
"""
        ),
        markdown_cell(
            """## Reading the three runs

Compare `eval/overall_perplexity`, `eval/s2orc_perplexity`, `eval/s2ag_perplexity`, and `train/train_loss` inside the W&B group `plamo-2-1b-pes2o-dedup-25m`. Lower perplexity is better. Because Raw, MinHashLSH, and LSHBloom each use the same PLaMo token budget, their final differences estimate the effect of which content remains after deduplication. They do not measure token-cost savings; the full-corpus efficiency experiment serves that separate question.
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "V100", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
