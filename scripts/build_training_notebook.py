import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CORE_PATH = ROOT / "src" / "pes2o_training.py"
EVALUATION_CORE_PATH = ROOT / "src" / "pes2o_perplexity.py"
OUTPUT_PATH = ROOT / "notebooks" / "qwen_pes2o_continued_pretraining.ipynb"


def source_lines(source):
    return source.splitlines(keepends=True)


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines(source),
    }


def code_cell(source, tags=None):
    metadata = {}
    if tags:
        metadata["tags"] = tags
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
            """# Qwen2.5-0.5B：peS2o 三组继续预训练实验

这个 notebook 每次训练一个数据版本，并在训练后自动计算同一套 1,000 篇 peS2o validation perplexity。

运行顺序：

1. 在 Colab 选择 **Runtime → Change runtime type → V100 GPU**。
2. 在 Colab Secrets 中加入 `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`WANDB_API_KEY`；临时 AWS 凭证还需加入 `AWS_SESSION_TOKEN`。
3. 第一次保持 `VARIANT = "raw"`，依次运行全部单元格。
4. 使用新的 V100 runtime，依次改成 `minhashlsh` 和 `lshbloom` 再各跑一次。

三次运行都固定为 24,999,936 个输入 token，其余训练参数完全相同。
"""
        ),
        code_cell(
            """# Install experiment dependencies.
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
    "wandb>=0.18,<1",
    "boto3>=1.35,<2",
])
print("Dependencies installed. If Colab asks for a restart, restart and run all cells again.")
"""
        ),
        code_cell(training_core, tags=["training-core-library"]),
        code_cell(evaluation_core, tags=["evaluation-core-library"]),
        markdown_cell(
            """## 1. 选择数据版本并检查环境

每个 Colab runtime 只跑一个版本。请只修改下一格的 `VARIANT`。
"""
        ),
        code_cell(
            """import json
import math
import os
import platform
import random
import shutil
import sys
from pathlib import Path

import boto3
import numpy as np
import torch
import transformers
import wandb
from google.colab import userdata
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


VARIANT = "raw"
ALLOWED_VARIANTS = {"raw", "minhashlsh", "lshbloom"}

CONFIG = {
    "model_id": "Qwen/Qwen2.5-0.5B",
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
    "eval_batch_size": 4,
    "s2orc_documents": 320,
    "s2ag_documents": 680,
    "wandb_project": "lshbloom-pes2o",
    "wandb_group": "qwen2.5-0.5b-pes2o-dedup-25m",
}

if VARIANT not in ALLOWED_VARIANTS:
    raise ValueError(f"VARIANT must be one of {sorted(ALLOWED_VARIANTS)}")
if CONFIG["sequence_length"] * CONFIG["sequence_count"] != CONFIG["train_input_tokens"]:
    raise RuntimeError("Training token budget is internally inconsistent")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Colab.")

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])

AWS_ACCESS_KEY_ID = userdata.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = userdata.get("AWS_SECRET_ACCESS_KEY")
WANDB_API_KEY = userdata.get("WANDB_API_KEY")
try:
    AWS_SESSION_TOKEN = userdata.get("AWS_SESSION_TOKEN")
except Exception:
    AWS_SESSION_TOKEN = None

session_kwargs = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "region_name": "ap-northeast-1",
}
if AWS_SESSION_TOKEN:
    session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN
s3 = boto3.session.Session(**session_kwargs).client("s3")

S3_BUCKET = "calista-bucket"
S3_PREFIX = "pes2o/v2/experiments/pilot-5000/"
S3_URI = "s3://calista-bucket/pes2o/v2/experiments/pilot-5000/"
WORK_DIR = Path(f"/content/pes2o-{VARIANT}-25m")
DATA_PATH = WORK_DIR / "train.jsonl.gz"
MANIFEST_PATH = WORK_DIR / "manifest.json"
MEMMAP_PATH = WORK_DIR / "train-tokens.uint32"
TRAINER_DIR = WORK_DIR / "trainer"
FINAL_MODEL_DIR = WORK_DIR / "final"
RESULT_PATH = WORK_DIR / "result.json"

WORK_DIR.mkdir(parents=True, exist_ok=True)
print("GPU:", torch.cuda.get_device_name(0))
print("Variant:", VARIANT)
print("Training tokens:", f"{CONFIG['train_input_tokens']:,}")
print("S3 input:", f"{S3_URI}{VARIANT}/train.jsonl.gz")
"""
        ),
        markdown_cell(
            """## 2. 从 S3 下载并校验训练数据

下载后会用 manifest 中的 SHA-256 检查文件。校验不通过时立即停止。
"""
        ),
        code_cell(
            """s3.download_file(S3_BUCKET, f"{S3_PREFIX}manifest.json", str(MANIFEST_PATH))
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
variant_info = manifest["variants"][VARIANT]

if variant_info["token_count"] < CONFIG["train_input_tokens"]:
    raise RuntimeError(
        f"{VARIANT} has only {variant_info['token_count']:,} tokens; "
        f"need {CONFIG['train_input_tokens']:,}"
    )

s3.download_file(
    S3_BUCKET,
    f"{S3_PREFIX}{variant_info['relative_path']}",
    str(DATA_PATH),
)
actual_sha256 = _sha256(DATA_PATH)
expected_sha256 = manifest["variants"][VARIANT]["sha256"]
if actual_sha256 != expected_sha256:
    raise RuntimeError("Dataset SHA-256 mismatch")

print(
    f"Verified {VARIANT}: {variant_info['documents']:,} documents, "
    f"{variant_info['token_count']:,} tokens"
)
"""
        ),
        markdown_cell(
            """## 3. 打包固定的 2,500 万 token

文档之间加入一个 EOS，然后按原始顺序连续打包。只使用最前面的 12,207 个完整序列，三组训练量严格相同。
"""
        ),
        code_cell(
            """tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"], use_fast=True)
if tokenizer.eos_token_id is None:
    raise RuntimeError("Tokenizer has no EOS token")

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
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


print(json.dumps(packing, indent=2))
print("Optimizer updates:", math.ceil(len(train_dataset) / CONFIG["gradient_accumulation_steps"]))
"""
        ),
        markdown_cell(
            """## 4. W&B、模型和单批 smoke test

Smoke test 会完成一次前向和反向传播。显存不足或数据形状错误会在长时间训练前暴露。
"""
        ),
        code_cell(
            """wandb.login(key=WANDB_API_KEY)
run = wandb.init(
    project=CONFIG["wandb_project"],
    group=CONFIG["wandb_group"],
    name=f"qwen2.5-0.5b-{VARIANT}-25m",
    config={
        **CONFIG,
        "variant": VARIANT,
        "dataset_manifest": f"{S3_URI}manifest.json",
        "dataset_sha256": actual_sha256,
        "dataset_documents": variant_info["documents"],
        "dataset_tokens": variant_info["token_count"],
        "packing": packing,
        "gpu": torch.cuda.get_device_name(0),
    },
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_id"],
    torch_dtype=torch.float32,
).to("cuda")
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.train()

smoke_batch = causal_lm_collator([train_dataset[0]])
smoke_batch = {key: value.to("cuda") for key, value in smoke_batch.items()}
with torch.autocast("cuda", dtype=torch.float16):
    smoke_output = model(**smoke_batch)
if not torch.isfinite(smoke_output.loss):
    raise RuntimeError(f"Smoke test loss is not finite: {smoke_output.loss.item()}")
smoke_output.loss.backward()
model.zero_grad(set_to_none=True)
wandb.log({"smoke_test/passed": 1, "smoke_test/loss": smoke_output.loss.item()})
print(f"Smoke test passed: loss={smoke_output.loss.item():.4f}")
del smoke_batch, smoke_output
torch.cuda.empty_cache()
"""
        ),
        markdown_cell(
            """## 5. 继续预训练

训练约有 1,526 个 optimizer step。W&B 每 10 step 记录一次，Colab 本地每 250 step 保存 checkpoint。
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
    save_total_limit=2,
    report_to=["wandb"],
    run_name=f"qwen2.5-0.5b-{VARIANT}-25m",
    seed=CONFIG["seed"],
    data_seed=CONFIG["seed"],
    dataloader_num_workers=2,
    remove_unused_columns=False,
    optim="adamw_torch",
    save_safetensors=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=causal_lm_collator,
)
train_output = trainer.train()
train_metrics = dict(train_output.metrics)
train_metrics["train_input_tokens"] = CONFIG["train_input_tokens"]
train_metrics["variant"] = VARIANT
trainer.log_metrics("train", train_metrics)
trainer.save_metrics("train", train_metrics)
trainer.save_state()
trainer.save_model(str(FINAL_MODEL_DIR))
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print("Training complete:", json.dumps(train_metrics, indent=2))
"""
        ),
        markdown_cell(
            """## 6. 训练后计算相同的 1,000 篇 validation PPL

与 Base 测试相同：320 篇 S2ORC、680 篇 S2AG、长度 2,048、batch size 4。
"""
        ),
        code_cell(
            """VALIDATION_URLS = [
    "https://huggingface.co/datasets/allenai/peS2o/resolve/main/data/v2/validation-00000-of-00002.json.gz",
    "https://huggingface.co/datasets/allenai/peS2o/resolve/main/data/v2/validation-00001-of-00002.json.gz",
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


model.eval()
source_results = {}
for source in ("s2orc", "s2ag"):
    packed_sequences = iter_packed_sequences(
        records_by_source[source], tokenizer, CONFIG["sequence_length"]
    )
    source_results[source] = evaluate_source(
        model=model,
        packed_sequences=packed_sequences,
        source=source,
        batch_size=CONFIG["eval_batch_size"],
        device="cuda",
        log_every_steps=10,
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
            """## 7. 保存结果并上传 S3

最终模型、tokenizer、训练指标和 PPL 结果会保存到对应 variant 的 checkpoint 目录。
"""
        ),
        code_cell(
            """result = {
    "variant": VARIANT,
    "config": CONFIG,
    "dataset": {
        "s3_uri": f"{S3_URI}{variant_info['relative_path']}",
        "sha256": actual_sha256,
        "documents": variant_info["documents"],
        "available_tokens": variant_info["token_count"],
        "packing": packing,
    },
    "training": train_metrics,
    "validation": {
        "urls": VALIDATION_URLS,
        "sources": source_results,
        "overall": overall,
    },
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "wandb": wandb.__version__,
        "gpu": torch.cuda.get_device_name(0),
    },
    "wandb_run_id": run.id,
    "wandb_run_url": run.url,
}
RESULT_PATH.write_text(json.dumps(result, indent=2) + "\\n", encoding="utf-8")

checkpoint_prefix = f"{S3_PREFIX}checkpoints/{VARIANT}"
for local_path in FINAL_MODEL_DIR.rglob("*"):
    if local_path.is_file():
        relative = local_path.relative_to(FINAL_MODEL_DIR).as_posix()
        s3.upload_file(
            str(local_path),
            S3_BUCKET,
            f"{checkpoint_prefix}/final/{relative}",
        )
s3.upload_file(str(RESULT_PATH), S3_BUCKET, f"{checkpoint_prefix}/result.json")
s3.upload_file(
    str(TRAINER_DIR / "train_results.json"),
    S3_BUCKET,
    f"{checkpoint_prefix}/train_results.json",
)

artifact = wandb.Artifact(
    name=f"qwen2.5-0.5b-{VARIANT}-25m-results",
    type="evaluation",
    metadata={"variant": VARIANT, "train_input_tokens": CONFIG["train_input_tokens"]},
)
artifact.add_file(str(RESULT_PATH))
run.log_artifact(artifact)
run.summary["checkpoint_s3_uri"] = f"s3://{S3_BUCKET}/{checkpoint_prefix}/final/"
print("W&B:", run.url)
print("Checkpoint:", run.summary["checkpoint_s3_uri"])
print("Result:", f"s3://{S3_BUCKET}/{checkpoint_prefix}/result.json")
wandb.finish()
"""
        ),
        markdown_cell(
            """## 如何判断结果

三组都完成后，在 W&B group `qwen2.5-0.5b-pes2o-dedup-25m` 中比较：

- `eval/overall_perplexity`：越低越好。
- `eval/s2orc_perplexity` 与 `eval/s2ag_perplexity`：观察影响来自完整论文还是标题摘要。
- `train/train_loss` 和训练速度：确认三组训练稳定且计算量相同。

Raw 与两种去重方法的差值才是主要结论；MinHashLSH 与 LSHBloom 预期非常接近。
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "V100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    build_notebook()
