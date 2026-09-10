import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "src" / "lshbloom_pes2o" / "perplexity.py"
OUTPUT_PATH = ROOT / "notebooks" / "qwen" / "qwen_pes2o_validation_perplexity.ipynb"


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
    core_source = CORE_PATH.read_text(encoding="utf-8")
    cells = [
        markdown_cell(
            """# Qwen2.5-0.5B Base 在 peS2o validation 上的 Perplexity

This notebook evaluates; it does not train or update the model.

默认评测 1,000 篇文档：320 篇 S2ORC 完整论文和 680 篇 S2AG 标题摘要。
它会将进度、最终 loss/perplexity 和结果 JSON 保存到 Weights & Biases。

运行前请在 Colab 选择 **Runtime → Change runtime type → V100 GPU**，然后依次运行所有单元格。
"""
        ),
        code_cell(
            """# Install only the packages that Colab does not reliably preinstall.
# Equivalent command: python -m pip install transformers wandb
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "transformers>=4.46,<5",
    "wandb>=0.18,<1",
])
print("Dependencies installed. If Colab asks for a runtime restart, restart and run all cells again.")
"""
        ),
        code_cell(core_source, tags=["core-library"]),
        code_cell(
            """import platform
import random
import sys
from pathlib import Path

import torch
import transformers
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer


CONFIG = {
    "model_id": "Qwen/Qwen2.5-0.5B",
    "sequence_length": 2048,
    "batch_size": 4,
    "seed": 42,
    "log_every_steps": 10,
    "wandb_project": "lshbloom-pes2o",
    "run_name": "qwen2.5-0.5b-base-pes2o-valid-1000",
    "s2orc_documents": 320,
    "s2ag_documents": 680,
}

VALIDATION_URLS = [
    "https://huggingface.co/datasets/allenai/peS2o/resolve/main/data/v2/validation-00000-of-00002.json.gz",
    "https://huggingface.co/datasets/allenai/peS2o/resolve/main/data/v2/validation-00001-of-00002.json.gz",
]
OUTPUT_PATH = Path("/content/results/qwen2.5-0.5b-base-pes2o-valid-1000.json")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. In Colab, select Runtime > Change runtime type > GPU.")

random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])

print("GPU:", torch.cuda.get_device_name(0))
print("Configuration:", json.dumps(CONFIG, indent=2))
"""
        ),
        markdown_cell(
            """## 读取固定的 1,000 篇 validation 文档

代码检查每条记录的 `source` 字段，并在两个官方 validation 文件中收集到 320 篇 S2ORC 和 680 篇 S2AG 后停止。
"""
        ),
        code_cell(
            """source_limits = {
    "s2orc": CONFIG["s2orc_documents"],
    "s2ag": CONFIG["s2ag_documents"],
}
records_by_source = collect_source_records(VALIDATION_URLS, source_limits)

actual_counts = {
    source: len(records) for source, records in records_by_source.items()
}
for source, limit in source_limits.items():
    if limit is not None and actual_counts[source] != limit:
        raise RuntimeError(
            f"Unexpected sample count for {source}: "
            f"expected {limit}, received {actual_counts[source]}"
        )

print("Loaded documents:", actual_counts)
print("First IDs:", {source: records[0]["id"] for source, records in records_by_source.items()})
"""
        ),
        markdown_cell(
            """## 加载模型并评测

`wandb.login()` 会显示登录入口。模型以 FP16 加载到 V100，只执行前向计算。首先运行一个很小的 smoke test，成功后再计算 1,000 篇文档。
"""
        ),
        code_cell(
            """wandb.login()
run = wandb.init(
    project=CONFIG["wandb_project"],
    name=CONFIG["run_name"],
    config={
        **CONFIG,
        "dataset": "allenai/peS2o",
        "dataset_version": "v2",
        "validation_urls": VALIDATION_URLS,
        "gpu": torch.cuda.get_device_name(0),
    },
)


def log_progress(metrics):
    prefix = f"eval/{metrics['source']}"
    payload = {
        f"{prefix}/batch": metrics["batch"],
        f"{prefix}/sequences": metrics["sequences"],
        f"{prefix}/predicted_tokens": metrics["predicted_tokens"],
        f"{prefix}/running_loss": metrics["loss"],
        f"{prefix}/running_perplexity": metrics["perplexity"],
        f"{prefix}/tokens_per_second": metrics["tokens_per_second"],
        f"{prefix}/elapsed_seconds": metrics["elapsed_seconds"],
    }
    if "gpu_memory_gb" in metrics:
        payload[f"{prefix}/gpu_memory_gb"] = metrics["gpu_memory_gb"]
    wandb.log(payload)


try:
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"])
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Qwen tokenizer does not define an EOS token")

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_id"],
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()

    smoke_length = min(128, CONFIG["sequence_length"])
    smoke_sequence = next(
        iter_packed_sequences(records_by_source["s2ag"][:1], tokenizer, smoke_length)
    )
    smoke_input_ids = torch.tensor(
        [smoke_sequence["input_ids"]], dtype=torch.long, device="cuda"
    )
    smoke_labels = torch.tensor(
        [smoke_sequence["labels"]], dtype=torch.long, device="cuda"
    )
    smoke_attention_mask = smoke_labels.ne(-100).long()
    smoke_targets = count_shifted_targets(smoke_labels)
    if smoke_targets < 1:
        raise RuntimeError("Smoke test produced no prediction targets")

    with torch.inference_mode():
        smoke_output = model(
            input_ids=smoke_input_ids,
            attention_mask=smoke_attention_mask,
            labels=smoke_labels,
        )
    smoke_loss = float(smoke_output.loss.item())
    if not math.isfinite(smoke_loss):
        raise RuntimeError(f"Smoke test produced non-finite loss: {smoke_loss}")
    wandb.log({
        "smoke_test/passed": 1,
        "smoke_test/loss": smoke_loss,
        "smoke_test/predicted_tokens": smoke_targets,
    })
    print(f"Smoke test passed: loss={smoke_loss:.4f}")

    del smoke_input_ids, smoke_labels, smoke_attention_mask, smoke_output
    torch.cuda.empty_cache()

    source_results = {}
    for source in ("s2orc", "s2ag"):
        print(f"Evaluating {source}: {len(records_by_source[source])} documents")
        packed_sequences = iter_packed_sequences(
            records_by_source[source], tokenizer, CONFIG["sequence_length"]
        )
        source_results[source] = evaluate_source(
            model=model,
            packed_sequences=packed_sequences,
            source=source,
            batch_size=CONFIG["batch_size"],
            device="cuda",
            log_every_steps=CONFIG["log_every_steps"],
            progress_callback=log_progress,
        )
        print(
            f"{source}: loss={source_results[source]['loss']:.4f}, "
            f"perplexity={source_results[source]['perplexity']:.4f}"
        )

    overall = combine_source_metrics(source_results.values())
    if not math.isfinite(overall["perplexity"]):
        raise RuntimeError(f"Overall perplexity is not finite: {overall['perplexity']}")

    result = {
        "config": CONFIG,
        "dataset": {
            "name": "allenai/peS2o",
            "version": "v2",
            "split": "validation",
            "urls": VALIDATION_URLS,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "wandb": wandb.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "sources": source_results,
        "overall": overall,
        "wandb_run_id": run.id,
        "wandb_run_url": run.url,
    }
    result_path = write_result_json(OUTPUT_PATH, result)

    for source, metrics in source_results.items():
        for key in ("documents", "input_tokens", "predicted_tokens", "loss", "perplexity", "tokens_per_second"):
            run.summary[f"final/{source}/{key}"] = metrics[key]
    for key in ("predicted_tokens", "loss", "perplexity"):
        run.summary[f"final/overall/{key}"] = overall[key]

    artifact = wandb.Artifact(
        name=f"{CONFIG['run_name']}-results-{run.id}",
        type="evaluation-results",
        description="Qwen2.5-0.5B Base perplexity on the peS2o V2 validation pilot",
    )
    artifact.add_file(str(result_path))
    run.log_artifact(artifact)

    print("\\nFinal results")
    print("-------------")
    for source, metrics in source_results.items():
        print(
            f"{source:5s} | documents={metrics['documents']:4d} | "
            f"tokens={metrics['predicted_tokens']:9d} | "
            f"loss={metrics['loss']:.4f} | ppl={metrics['perplexity']:.4f}"
        )
    print(
        f"overall | tokens={overall['predicted_tokens']:9d} | "
        f"loss={overall['loss']:.4f} | ppl={overall['perplexity']:.4f}"
    )
    print("Result JSON:", result_path)
    print("W&B run:", run.url)
finally:
    wandb.finish()
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "qwen_pes2o_validation_perplexity.ipynb",
                "provenance": [],
            },
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
    notebook = build_notebook()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
