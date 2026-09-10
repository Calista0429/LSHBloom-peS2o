import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "src" / "lshbloom_pes2o" / "sciq.py"
OUTPUT_PATH = ROOT / "notebooks" / "qwen" / "qwen_pes2o_sciq_evaluation.ipynb"


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
    core = CORE_PATH.read_text(encoding="utf-8")
    cells = [
        markdown_cell(
            """# Qwen2.5-0.5B：三个 peS2o checkpoint 的 SciQ 测评

这个 notebook 只测 **SciQ**，比较 Raw、MinHashLSH 和 LSHBloom 三个模型。

运行前：

1. 在 Colab 选择 **Runtime → Change runtime type → V100 GPU**。
2. 在 Colab Secrets 中加入 `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`WANDB_API_KEY`。
3. 临时 AWS 凭证还需加入 `AWS_SESSION_TOKEN`。区域可选填 `AWS_DEFAULT_REGION`，默认使用 `ap-northeast-1`。
4. 从上到下运行所有单元格。每个模型先测 10 条，成功后再测完整 1,000 条 SciQ test。

三个 checkpoint 会逐个下载、评测和删除本地副本；S3 中的文件不会改变。
"""
        ),
        code_cell(
            """# Install the exact evaluation version used by this notebook.
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "lm-eval[hf]==0.4.13",
    "wandb>=0.21,<1",
    "boto3>=1.35,<2",
])
print("Dependencies installed.")
"""
        ),
        code_cell(core, tags=["sciq-result-library"]),
        markdown_cell(
            """## 1. 固定配置并读取 Colab Secrets

一般只需要调整 `batch_size`。V100 如果显存不足，把 8 改为 4。
"""
        ),
        code_cell(
            """import gc
import importlib.metadata
import os
import platform
import shutil
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import boto3
import pandas as pd
import torch
import wandb
from google.colab import userdata
from lm_eval import simple_evaluate


CHECKPOINTS = {
    "raw": "s3://calista-bucket/pes2o/v2/experiments/pilot-5000/checkpoints/raw/final/",
    "minhashlsh": "s3://calista-bucket/pes2o/v2/experiments/pilot-5000/checkpoints/minhashlsh/final/",
    "lshbloom": "s3://calista-bucket/pes2o/v2/experiments/pilot-5000/checkpoints/lshbloom/final/",
}

CONFIG = {
    "task": "sciq",
    "num_fewshot": 0,
    "batch_size": 8,
    "smoke_limit": 10,
    "full_limit": None,
    "dtype": "float16",
    "bootstrap_iters": 1_000,
    "seed": 42,
    "wandb_project": "lshbloom-pes2o",
    "wandb_group": "qwen2.5-0.5b-pes2o-dedup-25m",
    "wandb_run_name": "qwen2.5-0.5b-sciq-comparison",
}

RESULTS_DIR = Path("/content/results/sciq")
CHECKPOINT_ROOT = Path("/content/sciq-checkpoints")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)


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

for secret_name, secret_value in (
    ("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID),
    ("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY),
    ("WANDB_API_KEY", WANDB_API_KEY),
):
    if not secret_value:
        raise RuntimeError(f"Missing required Colab secret: {secret_name}")

AWS_SESSION_TOKEN = optional_secret("AWS_SESSION_TOKEN")
AWS_DEFAULT_REGION = optional_secret("AWS_DEFAULT_REGION") or "ap-northeast-1"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Colab.")
if tuple(CHECKPOINTS) != VARIANTS:
    raise RuntimeError("Checkpoint evaluation order does not match VARIANTS")

session_kwargs = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "region_name": AWS_DEFAULT_REGION,
}
if AWS_SESSION_TOKEN:
    session_kwargs["aws_session_token"] = AWS_SESSION_TOKEN
s3 = boto3.session.Session(**session_kwargs).client("s3")

print("GPU:", torch.cuda.get_device_name(0))
print("Task: SciQ, zero-shot")
print("Variants:", ", ".join(VARIANTS))
"""
        ),
        markdown_cell(
            """## 2. 检查 S3 checkpoint

这里只读取文件列表和元数据。三个 checkpoint 都完整后才开始下载模型。
"""
        ),
        code_cell(
            """REQUIRED_CHECKPOINT_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
}


def split_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    prefix = parsed.path.lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


def list_checkpoint_objects(uri):
    bucket, prefix = split_s3_uri(uri)
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            if item["Key"].endswith("/"):
                continue
            objects.append({
                "Key": item["Key"],
                "Size": int(item["Size"]),
                "ETag": item["ETag"].strip('"'),
            })
    if not objects:
        raise RuntimeError(f"No checkpoint objects found under {uri}")

    relative_names = {item["Key"][len(prefix):] for item in objects}
    missing = sorted(REQUIRED_CHECKPOINT_FILES - relative_names)
    if missing:
        raise RuntimeError(f"Checkpoint {uri} is missing: {', '.join(missing)}")
    return objects


checkpoint_objects = {
    variant: list_checkpoint_objects(uri)
    for variant, uri in CHECKPOINTS.items()
}

for variant in VARIANTS:
    total_bytes = sum(item["Size"] for item in checkpoint_objects[variant])
    print(f"{variant}: {len(checkpoint_objects[variant])} files, {total_bytes / 1e9:.2f} GB")
"""
        ),
        markdown_cell(
            """## 3. 运行 SciQ

每个模型先运行 10 条 smoke test，然后运行完整 1,000 条。Smoke 分数仅用于检查代码，不作为实验结论。
"""
        ),
        code_cell(
            """def download_checkpoint(variant, local_dir):
    uri = CHECKPOINTS[variant]
    bucket, prefix = split_s3_uri(uri)
    shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    for item in checkpoint_objects[variant]:
        key = item["Key"]
        relative_text = key[len(prefix):]
        relative = PurePosixPath(relative_text)
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe checkpoint object key: {key}")
        destination = local_dir.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(destination))
        if destination.stat().st_size != item["Size"]:
            raise RuntimeError(f"Downloaded size mismatch: {key}")


def run_sciq(local_dir, variant, stage, limit):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = simple_evaluate(
            model="hf",
            model_args={
                "pretrained": str(local_dir),
                "dtype": CONFIG["dtype"],
            },
            tasks=[CONFIG["task"]],
            num_fewshot=CONFIG["num_fewshot"],
            batch_size=CONFIG["batch_size"],
            device="cuda:0",
            limit=limit,
            bootstrap_iters=CONFIG["bootstrap_iters"],
            log_samples=True,
            apply_chat_template=False,
            random_seed=CONFIG["seed"],
            numpy_random_seed=CONFIG["seed"],
            torch_random_seed=CONFIG["seed"],
            fewshot_random_seed=CONFIG["seed"],
        )
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            f"CUDA out of memory with batch_size={CONFIG['batch_size']}. "
            "Reduce CONFIG['batch_size'] and rerun the notebook."
        ) from error

    elapsed = time.perf_counter() - started
    if result is None:
        raise RuntimeError(f"lm-eval returned no result for {variant}/{stage}")
    peak_cuda_bytes = torch.cuda.max_memory_allocated()
    row = extract_sciq_metrics(
        result,
        variant=variant,
        stage=stage,
        elapsed_seconds=elapsed,
        peak_cuda_bytes=peak_cuda_bytes,
    )
    write_json(RESULTS_DIR / f"{variant}-{stage}.json", result)
    run.log({
        f"{stage}/{variant}/acc": row["acc"],
        f"{stage}/{variant}/acc_norm": row["acc_norm"],
        f"{stage}/{variant}/acc_stderr": row["acc_stderr"],
        f"{stage}/{variant}/acc_norm_stderr": row["acc_norm_stderr"],
        f"{stage}/{variant}/examples": row["examples"],
        f"{stage}/{variant}/elapsed_seconds": row["elapsed_seconds"],
        f"{stage}/{variant}/peak_cuda_gb": row["peak_cuda_bytes"] / 1e9,
    })
    return row


wandb.login(key=WANDB_API_KEY)
run = wandb.init(
    project=CONFIG["wandb_project"],
    group=CONFIG["wandb_group"],
    name=CONFIG["wandb_run_name"],
    config={
        **CONFIG,
        "checkpoints": CHECKPOINTS,
        "checkpoint_objects": checkpoint_objects,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "lm_eval": importlib.metadata.version("lm-eval"),
    },
)

smoke_rows = []
full_rows = []
failures = []
comparison = None
comparison_frame = None

try:
    for variant in VARIANTS:
        local_dir = CHECKPOINT_ROOT / variant
        print(f"\\n=== {variant}: downloading checkpoint ===")
        try:
            download_checkpoint(variant, local_dir)
            print(f"=== {variant}: smoke test (10 examples) ===")
            smoke_rows.append(
                run_sciq(
                    local_dir,
                    variant,
                    stage="smoke",
                    limit=CONFIG["smoke_limit"],
                )
            )
            print(f"=== {variant}: full SciQ test (1000 examples) ===")
            full_rows.append(
                run_sciq(
                    local_dir,
                    variant,
                    stage="full",
                    limit=CONFIG["full_limit"],
                )
            )
        except Exception as error:
            failure = {
                "variant": variant,
                "error_type": type(error).__name__,
                "message": str(error),
            }
            failures.append(failure)
            print(f"FAILED {variant}: {failure['error_type']}: {failure['message']}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            shutil.rmtree(local_dir, ignore_errors=True)

    if failures:
        write_json(RESULTS_DIR / "failures.json", failures)
    else:
        comparison = build_comparison(full_rows)
        comparison_frame = pd.DataFrame(comparison)
        comparison_frame.to_csv(RESULTS_DIR / "comparison.csv", index=False)
        run.log({"sciq/comparison": wandb.Table(dataframe=comparison_frame)})
        display(comparison_frame)

    summary = {
        "status": "failed" if failures else "complete",
        "config": CONFIG,
        "checkpoints": CHECKPOINTS,
        "checkpoint_objects": checkpoint_objects,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "lm_eval": importlib.metadata.version("lm-eval"),
            "wandb": wandb.__version__,
            "boto3": boto3.__version__,
        },
        "smoke": smoke_rows,
        "full": full_rows,
        "comparison": comparison,
        "failures": failures,
    }
    write_json(RESULTS_DIR / "summary.json", summary)
finally:
    artifact = wandb.Artifact(
        name="qwen2.5-0.5b-pes2o-sciq-results",
        type="evaluation",
    )
    artifact.add_dir(str(RESULTS_DIR))
    run.log_artifact(artifact)
    wandb.finish()

if failures:
    raise RuntimeError(
        "SciQ evaluation did not complete for every model. "
        "See failures.json and the W&B artifact."
    )

print("\\nSciQ evaluation complete. Primary metric: acc_norm.")
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "V100",
                "provenance": [],
            },
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
    notebook = build_notebook()
    OUTPUT_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
