import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "qwen_pes2o_efficiency_training.ipynb"
CORE_PATHS = (
    ROOT / "src" / "pes2o_training.py",
    ROOT / "src" / "pes2o_perplexity.py",
    ROOT / "src" / "sciq_evaluation.py",
    ROOT / "src" / "efficiency_curves.py",
)


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
    cells = [
        markdown_cell(
            """# Qwen2.5-0.5B：peS2o 完整数据训练效率曲线

这个 notebook 每次运行一个数据版本：`raw`、`minhashlsh` 或 `lshbloom`。与原来的固定 2,500 万 token 实验不同，本实验把当前版本的所有完整 2,048-token 序列训练一个 epoch。

训练期间每 250 个 optimizer step 在同一份固定 validation probe 上测一次 perplexity，并保存一个只包含模型权重的临时 checkpoint。训练结束后，对 Base 和全部 checkpoint 测完整 1,000 条 SciQ。最终得到随累计训练 token 和纯训练 GPU 时间变化的曲线数据。

在 Colab 中选择 V100，设置 `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY` 和 `WANDB_API_KEY` Secrets。临时 AWS 凭证还需要 `AWS_SESSION_TOKEN`。每次只修改 `VARIANT`，三个版本分别使用新的 runtime 运行。
"""
        ),
        code_cell(
            """# Install pinned experiment dependencies.
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "transformers>=4.56,<5",
    "accelerate>=1.8,<2",
    "lm-eval[hf]==0.4.13",
    "wandb>=0.21,<1",
    "boto3>=1.35,<2",
])
print("Dependencies installed. Restart the runtime only if Colab asks for it.")
"""
        ),
    ]
    for core_path in CORE_PATHS:
        cells.append(
            code_cell(
                core_path.read_text(encoding="utf-8"),
                tags=[f"embedded-{core_path.stem}"],
            )
        )
    cells.extend(
        [
            markdown_cell(
                """## 1. Configuration and credentials

The only normal edit is `VARIANT`. The validation probe contains 128 fixed sequences. It is used for the curve only; the final checkpoint also receives the full 1,000-document peS2o validation evaluation.
"""
            ),
            code_cell(
                """import gc
import importlib.metadata
import itertools
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import torch
import transformers
import wandb
from google.colab import userdata
from lm_eval import simple_evaluate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


VARIANT = "raw"
ALLOWED_VARIANTS = set(VARIANTS)

CONFIG = {
    "model_id": "Qwen/Qwen2.5-0.5B",
    "sequence_length": 2048,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.03,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "fp16": True,
    "seed": 42,
    "curve_interval_steps": 250,
    "logging_steps": 10,
    "curve_validation_sequences": {"s2orc": 64, "s2ag": 64},
    "full_validation_documents": {"s2orc": 320, "s2ag": 680},
    "eval_batch_size": 4,
    "sciq_batch_size": 8,
    "sciq_examples": 1000,
    "sciq_bootstrap_iters": 1000,
    "wandb_project": "lshbloom-pes2o",
    "wandb_group": "qwen2.5-0.5b-pes2o-dedup-full-efficiency",
}

if VARIANT not in ALLOWED_VARIANTS:
    raise ValueError(f"VARIANT must be one of {sorted(ALLOWED_VARIANTS)}")
if sum(CONFIG["curve_validation_sequences"].values()) != 128:
    raise RuntimeError("Curve validation probe must contain exactly 128 sequences")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Colab.")

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

AWS_SESSION_TOKEN = optional_secret("AWS_SESSION_TOKEN")
AWS_DEFAULT_REGION = optional_secret("AWS_DEFAULT_REGION") or "ap-northeast-1"
session_kwargs = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "region_name": AWS_DEFAULT_REGION,
}
if AWS_SESSION_TOKEN:
    session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN
s3 = boto3.session.Session(**session_kwargs).client("s3")

S3_BUCKET = "calista-bucket"
S3_SOURCE_PREFIX = "pes2o/v2/experiments/pilot-5000/"
S3_OUTPUT_PREFIX = f"{S3_SOURCE_PREFIX}efficiency/{VARIANT}/"
WORK_DIR = Path(f"/content/pes2o-efficiency-{VARIANT}")
DATA_PATH = WORK_DIR / "train.jsonl.gz"
MANIFEST_PATH = WORK_DIR / "manifest.json"
MEMMAP_PATH = WORK_DIR / "train-tokens.uint32"
TRAINER_DIR = WORK_DIR / "trainer"
FINAL_MODEL_DIR = WORK_DIR / "final"
RESULTS_DIR = WORK_DIR / "results"
CURVE_RESULT_PATH = RESULTS_DIR / "curve-results.json"
CURVE_CSV_PATH = RESULTS_DIR / "curve-results.csv"
WORK_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("GPU:", torch.cuda.get_device_name(0))
print("Variant:", VARIANT)
print("S3 output:", f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}")
"""
            ),
            markdown_cell(
                """## 2. Download the selected dataset and use its full token budget

Only the incomplete tail shorter than 2,048 tokens is discarded. This is at most 2,047 tokens and is reported explicitly.
"""
            ),
            code_cell(
                """s3.download_file(S3_BUCKET, f"{S3_SOURCE_PREFIX}manifest.json", str(MANIFEST_PATH))
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
token_counts = {
    name: int(manifest["variants"][name]["token_count"])
    for name in VARIANTS
}
training_plan = build_full_training_plan(token_counts, CONFIG["sequence_length"])
variant_plan = training_plan[VARIANT]
variant_info = manifest["variants"][VARIANT]

s3.download_file(
    S3_BUCKET,
    f"{S3_SOURCE_PREFIX}{variant_info['relative_path']}",
    str(DATA_PATH),
)
if _sha256(DATA_PATH) != variant_info["sha256"]:
    raise RuntimeError("Dataset SHA-256 mismatch")

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"], use_fast=True)
packing = pack_jsonl_gz_to_memmap(
    input_path=DATA_PATH,
    output_path=MEMMAP_PATH,
    tokenizer=tokenizer,
    sequence_length=CONFIG["sequence_length"],
    sequence_count=variant_plan["sequence_count"],
)
train_dataset = TokenMemmapDataset(
    MEMMAP_PATH,
    sequence_length=CONFIG["sequence_length"],
    sequence_count=variant_plan["sequence_count"],
)

updates_per_epoch = math.ceil(
    len(train_dataset)
    / (CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"])
)
tokens_per_update = (
    CONFIG["sequence_length"]
    * CONFIG["per_device_train_batch_size"]
    * CONFIG["gradient_accumulation_steps"]
)
expected_steps = checkpoint_steps(updates_per_epoch, CONFIG["curve_interval_steps"])

print(json.dumps(variant_plan, indent=2))
print("Optimizer updates:", updates_per_epoch)
print("Curve steps:", expected_steps)
"""
            ),
            markdown_cell(
                """## 3. Build a small fixed validation probe

The same document order and sequence counts are used in every variant run. Perplexity at intermediate checkpoints is therefore directly comparable.
"""
            ),
            code_cell(
                """VALIDATION_URLS = [
    "https://huggingface.co/datasets/allenai/peS2o/resolve/main/data/v2/validation-00000-of-00002.json.gz",
    "https://huggingface.co/datasets/allenai/peS2o/resolve/main/data/v2/validation-00001-of-00002.json.gz",
]
validation_records = collect_source_records(
    VALIDATION_URLS,
    CONFIG["full_validation_documents"],
)


class PackedListDataset:
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


probe_sequences = []
for source in ("s2orc", "s2ag"):
    requested = CONFIG["curve_validation_sequences"][source]
    items = list(
        itertools.islice(
            iter_packed_sequences(
                validation_records[source], tokenizer, CONFIG["sequence_length"]
            ),
            requested,
        )
    )
    if len(items) != requested:
        raise RuntimeError(
            f"Validation probe has only {len(items)} {source} sequences; need {requested}"
        )
    probe_sequences.extend(items)
curve_eval_dataset = PackedListDataset(probe_sequences)


def causal_lm_collator(features):
    input_ids = torch.tensor(
        [feature["input_ids"] for feature in features], dtype=torch.long
    )
    labels = torch.tensor(
        [feature.get("labels", feature["input_ids"]) for feature in features],
        dtype=torch.long,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": labels.ne(-100).long(),
        "labels": labels,
    }


print("Curve validation sequences:", len(curve_eval_dataset))
"""
            ),
            markdown_cell(
                """## 4. Train one epoch and measure perplexity at fixed steps

The callback measures time spent inside optimizer steps. Validation and SciQ time are excluded from the GPU-hour x-axis, so the x-axis represents training cost.
"""
            ),
            code_cell(
                """wandb.login(key=WANDB_API_KEY)
run = wandb.init(
    project=CONFIG["wandb_project"],
    group=CONFIG["wandb_group"],
    name=f"qwen2.5-0.5b-{VARIANT}-full-efficiency",
    config={
        **CONFIG,
        "variant": VARIANT,
        "training_plan": training_plan,
        "dataset_sha256": variant_info["sha256"],
        "packing": packing,
        "gpu": torch.cuda.get_device_name(0),
    },
)


class TrainingClockCallback(TrainerCallback):
    def __init__(self, variant, tokens_per_update, total_train_tokens):
        self.variant = variant
        self.tokens_per_update = tokens_per_update
        self.total_train_tokens = total_train_tokens
        self.train_seconds = 0.0
        self.step_started = None
        self.validation_by_step = {}

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_started is not None:
            self.train_seconds += time.perf_counter() - self.step_started
            self.step_started = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        if "eval_loss" not in metrics:
            return
        loss = float(metrics["eval_loss"])
        step = int(state.global_step)
        row = {
            "variant": self.variant,
            "global_step": step,
            "cumulative_train_tokens": tokens_at_step(
                step, self.tokens_per_update, self.total_train_tokens
            ),
            "cumulative_train_gpu_hours": self.train_seconds / 3600.0,
            "validation_loss": loss,
            "validation_perplexity": perplexity_from_loss(loss),
        }
        self.validation_by_step[step] = row
        run.log(
            {
                "curve/global_step": step,
                "curve/cumulative_train_tokens": row["cumulative_train_tokens"],
                "curve/cumulative_train_gpu_hours": row["cumulative_train_gpu_hours"],
                "curve/validation_loss": row["validation_loss"],
                "curve/validation_perplexity": row["validation_perplexity"],
            }
        )


model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_id"],
    torch_dtype=torch.float32,
).to("cuda")
model.config.use_cache = False
model.gradient_checkpointing_enable()

clock = TrainingClockCallback(
    VARIANT,
    tokens_per_update=tokens_per_update,
    total_train_tokens=variant_plan["train_input_tokens"],
)
training_args = TrainingArguments(
    output_dir=str(TRAINER_DIR),
    overwrite_output_dir=True,
    num_train_epochs=1.0,
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=CONFIG["eval_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    lr_scheduler_type="cosine",
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    max_grad_norm=CONFIG["max_grad_norm"],
    fp16=CONFIG["fp16"],
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=CONFIG["curve_interval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["curve_interval_steps"],
    save_total_limit=None,
    save_only_model=True,
    logging_steps=CONFIG["logging_steps"],
    logging_first_step=True,
    report_to=["wandb"],
    run_name=f"qwen2.5-0.5b-{VARIANT}-full-efficiency",
    seed=CONFIG["seed"],
    data_seed=CONFIG["seed"],
    dataloader_num_workers=2,
    remove_unused_columns=False,
    prediction_loss_only=True,
    optim="adamw_torch",
    save_safetensors=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=curve_eval_dataset,
    data_collator=causal_lm_collator,
    processing_class=tokenizer,
    callbacks=[clock],
)

print("Running step-0 validation probe")
trainer.evaluate()
train_output = trainer.train()
if trainer.state.global_step != updates_per_epoch:
    raise RuntimeError(
        f"Expected {updates_per_epoch} optimizer steps, got {trainer.state.global_step}"
    )
print("Running final validation probe")
trainer.evaluate()

train_metrics = dict(train_output.metrics)
train_metrics.update({
    "variant": VARIANT,
    "train_input_tokens": variant_plan["train_input_tokens"],
    "pure_training_gpu_hours": clock.train_seconds / 3600.0,
})
trainer.save_model(str(FINAL_MODEL_DIR))
tokenizer.save_pretrained(FINAL_MODEL_DIR)
for checkpoint_dir in TRAINER_DIR.glob("checkpoint-*"):
    tokenizer.save_pretrained(checkpoint_dir)

missing_validation_steps = sorted(set(expected_steps) - set(clock.validation_by_step))
if missing_validation_steps:
    raise RuntimeError(f"Missing validation measurements at steps {missing_validation_steps}")
validation_rows = [clock.validation_by_step[step] for step in expected_steps]
print(json.dumps(train_metrics, indent=2))
"""
            ),
            markdown_cell(
                """## 5. Evaluate complete SciQ at every curve point

Step 0 uses the original Qwen Base model. Intermediate checkpoints are local model-only checkpoints. They are not uploaded to S3 after their metrics have been collected.
"""
            ),
            code_cell(
                """checkpoint_paths = {0: CONFIG["model_id"]}
for step in expected_steps[1:]:
    if step == updates_per_epoch:
        checkpoint_paths[step] = str(FINAL_MODEL_DIR)
    else:
        local_path = TRAINER_DIR / f"checkpoint-{step}"
        if not local_path.is_dir():
            raise RuntimeError(f"Missing local checkpoint: {local_path}")
        checkpoint_paths[step] = str(local_path)

del trainer, model
gc.collect()
torch.cuda.empty_cache()

sciq_rows = []
for step in expected_steps:
    checkpoint = checkpoint_paths[step]
    print(f"SciQ: {VARIANT}, step={step}, checkpoint={checkpoint}")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    harness_result = simple_evaluate(
        model="hf",
        model_args={"pretrained": checkpoint, "dtype": "float16"},
        tasks=["sciq"],
        num_fewshot=0,
        batch_size=CONFIG["sciq_batch_size"],
        device="cuda:0",
        limit=CONFIG["sciq_examples"],
        bootstrap_iters=CONFIG["sciq_bootstrap_iters"],
        log_samples=False,
        apply_chat_template=False,
        random_seed=CONFIG["seed"],
        numpy_random_seed=CONFIG["seed"],
        torch_random_seed=CONFIG["seed"],
        fewshot_random_seed=CONFIG["seed"],
    )
    extracted = extract_sciq_metrics(
        harness_result,
        variant=VARIANT,
        stage="full",
        elapsed_seconds=time.perf_counter() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    )
    row = {
        "variant": VARIANT,
        "global_step": step,
        "sciq_acc": extracted["acc"],
        "sciq_acc_norm": extracted["acc_norm"],
        "sciq_acc_stderr": extracted["acc_stderr"],
        "sciq_acc_norm_stderr": extracted["acc_norm_stderr"],
        "sciq_examples": extracted["examples"],
    }
    sciq_rows.append(row)
    run.log({
        "curve/global_step": step,
        "curve/sciq_acc": row["sciq_acc"],
        "curve/sciq_acc_norm": row["sciq_acc_norm"],
    })
    del harness_result
    gc.collect()
    torch.cuda.empty_cache()

curve_rows = merge_curve_measurements(validation_rows, sciq_rows)
write_curve_csv(CURVE_CSV_PATH, curve_rows)
run.log({
    "curve/results": wandb.Table(
        data=[[row[column] for column in CURVE_COLUMNS] for row in curve_rows],
        columns=list(CURVE_COLUMNS),
    )
})
display(curve_rows)
"""
            ),
            markdown_cell(
                """## 6. Full final peS2o validation evaluation

This final check uses the same 320 S2ORC and 680 S2AG documents as the earlier perplexity experiment. It is reported separately because the intermediate curve uses the smaller fixed probe.
"""
            ),
            code_cell(
                """full_records = validation_records
final_model = AutoModelForCausalLM.from_pretrained(
    FINAL_MODEL_DIR,
    torch_dtype=torch.float16,
).to("cuda")
final_model.eval()

full_source_results = {}
for source in ("s2orc", "s2ag"):
    full_source_results[source] = evaluate_source(
        model=final_model,
        packed_sequences=iter_packed_sequences(
            full_records[source], tokenizer, CONFIG["sequence_length"]
        ),
        source=source,
        batch_size=CONFIG["eval_batch_size"],
        device="cuda",
        log_every_steps=25,
    )
    print(
        f"{source}: PPL={full_source_results[source]['perplexity']:.4f}, "
        f"tokens={full_source_results[source]['predicted_tokens']:,}"
    )

full_overall = combine_source_metrics(full_source_results.values())
run.log({
    "final_full_validation/loss": full_overall["loss"],
    "final_full_validation/perplexity": full_overall["perplexity"],
})
print("Full final validation:", json.dumps(full_overall, indent=2))
del final_model, full_records, validation_records
gc.collect()
torch.cuda.empty_cache()
"""
            ),
            markdown_cell(
                """## 7. Save final model and curve results

The final model, JSON, and CSV are uploaded. Intermediate checkpoints are deliberately kept local to avoid unnecessary S3 storage cost.
"""
            ),
            code_cell(
                """result = {
    "schema_version": 1,
    "variant": VARIANT,
    "config": CONFIG,
    "training_plan": training_plan,
    "variant_plan": variant_plan,
    "packing": packing,
    "train_metrics": train_metrics,
    "curve_probe": {
        "sequences": CONFIG["curve_validation_sequences"],
        "rows": curve_rows,
    },
    "full_final_validation": {
        "sources": full_source_results,
        "overall": full_overall,
    },
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "lm_eval": importlib.metadata.version("lm-eval"),
        "wandb": wandb.__version__,
        "gpu": torch.cuda.get_device_name(0),
    },
    "wandb_run_id": run.id,
    "wandb_run_url": run.url,
}
CURVE_RESULT_PATH.write_text(
    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\\n",
    encoding="utf-8",
)

for local_path in FINAL_MODEL_DIR.rglob("*"):
    if local_path.is_file():
        relative = local_path.relative_to(FINAL_MODEL_DIR).as_posix()
        s3.upload_file(
            str(local_path),
            S3_BUCKET,
            f"{S3_OUTPUT_PREFIX}final/{relative}",
        )
s3.upload_file(
    str(CURVE_RESULT_PATH), S3_BUCKET, f"{S3_OUTPUT_PREFIX}curve-results.json"
)
s3.upload_file(
    str(CURVE_CSV_PATH), S3_BUCKET, f"{S3_OUTPUT_PREFIX}curve-results.csv"
)

artifact = wandb.Artifact(
    name=f"qwen2.5-0.5b-{VARIANT}-full-efficiency-results",
    type="evaluation",
    metadata={
        "variant": VARIANT,
        "train_input_tokens": variant_plan["train_input_tokens"],
    },
)
artifact.add_dir(str(RESULTS_DIR))
run.log_artifact(artifact)
run.summary["train_input_tokens"] = variant_plan["train_input_tokens"]
run.summary["pure_training_gpu_hours"] = train_metrics["pure_training_gpu_hours"]
run.summary["final_sciq_acc_norm"] = curve_rows[-1]["sciq_acc_norm"]
run.summary["final_curve_perplexity"] = curve_rows[-1]["validation_perplexity"]
run.summary["final_full_perplexity"] = full_overall["perplexity"]
run.summary["checkpoint_s3_uri"] = f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}final/"
print("W&B:", run.url)
print("Results:", f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}curve-results.json")
wandb.finish()
"""
            ),
        ]
    )

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
            "language_info": {"name": "python"},
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
