import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "src" / "lshbloom_pes2o" / "efficiency.py"
OUTPUT_PATH = ROOT / "notebooks" / "qwen" / "qwen_pes2o_efficiency_curves.ipynb"


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
    core = CORE_PATH.read_text(encoding="utf-8")
    cells = [
        markdown_cell(
            """# peS2o Deduplication Training-Efficiency Curves

Run this notebook after the `raw`, `minhashlsh`, and `lshbloom` full-corpus training runs are complete. It downloads the three measured result files from S3, validates checkpoint ordering and experiment identity, creates CSV files and four comparison plots, and uploads the outputs to W&B and S3.

This notebook does not require a GPU. Add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `WANDB_API_KEY` to Colab Secrets. Temporary AWS credentials also require `AWS_SESSION_TOKEN`.
"""
        ),
        code_cell(
            """import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "boto3>=1.35,<2",
    "wandb>=0.21,<1",
    "pandas>=2.2,<3",
    "matplotlib>=3.9,<4",
])
print("Dependencies installed.")
"""
        ),
        code_cell(core, tags=["embedded-efficiency-curves"]),
        markdown_cell(
            """## 1. Download and validate the three real experiment results"""
        ),
        code_cell(
            """import json
import os
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from google.colab import userdata


CONFIG = {
    "wandb_project": "lshbloom-pes2o",
    "wandb_group": "qwen2.5-0.5b-pes2o-dedup-full-efficiency",
    "wandb_run_name": "qwen2.5-0.5b-full-efficiency-curves",
}
S3_BUCKET = "calista-bucket"
S3_PREFIX = "pes2o/v2/experiments/pilot-5000/efficiency/"
RESULT_KEYS = {
    "raw": f"{S3_PREFIX}raw/curve-results.json",
    "minhashlsh": f"{S3_PREFIX}minhashlsh/curve-results.json",
    "lshbloom": f"{S3_PREFIX}lshbloom/curve-results.json",
}
OUTPUT_DIR = Path("/content/results/efficiency-curves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def optional_secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None


AWS_ACCESS_KEY_ID = userdata.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = userdata.get("AWS_SECRET_ACCESS_KEY")
WANDB_API_KEY = userdata.get("WANDB_API_KEY")
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
aws_session_token = optional_secret("AWS_SESSION_TOKEN")
if aws_session_token:
    session_kwargs["aws_session_token"] = aws_session_token
s3 = boto3.session.Session(**session_kwargs).client("s3")

all_rows = []
raw_results = {}
for variant in VARIANTS:
    destination = OUTPUT_DIR / f"{variant}-curve-results.json"
    s3.download_file(S3_BUCKET, RESULT_KEYS[variant], str(destination))
    result = json.loads(destination.read_text(encoding="utf-8"))
    if result.get("variant") != variant:
        raise RuntimeError(
            f"Expected {variant} result, received {result.get('variant')}"
        )
    rows = result["curve_probe"]["rows"]
    validate_curve_rows(rows, expected_variants=(variant,))
    all_rows.extend(rows)
    raw_results[variant] = result

experiment_fingerprint = validate_shared_experiment_results(raw_results.values())
validate_curve_rows(all_rows, expected_variants=VARIANTS)
frame = pd.DataFrame(all_rows, columns=CURVE_COLUMNS)
frame.to_csv(OUTPUT_DIR / "all-curve-results.csv", index=False)
print("Verified shared experiment fingerprint:", experiment_fingerprint)
display(frame)
"""
        ),
        markdown_cell(
            """## 2. Calculate final cost and quality differences

Token savings use Raw as the baseline. GPU-hour savings use measured training time and can differ slightly from token savings because throughput is not perfectly constant.
"""
        ),
        code_cell(
            """final_rows = (
    frame.sort_values(["variant", "global_step"])
    .groupby("variant", sort=False)
    .tail(1)
    .set_index("variant")
    .loc[list(VARIANTS)]
    .reset_index()
)
raw_tokens = float(final_rows.loc[final_rows["variant"] == "raw", "cumulative_train_tokens"].iloc[0])
raw_hours = float(final_rows.loc[final_rows["variant"] == "raw", "cumulative_train_gpu_hours"].iloc[0])
raw_acc = float(final_rows.loc[final_rows["variant"] == "raw", "sciq_acc_norm"].iloc[0])
raw_ppl = float(final_rows.loc[final_rows["variant"] == "raw", "validation_perplexity"].iloc[0])

summary = final_rows[[
    "variant",
    "cumulative_train_tokens",
    "cumulative_train_gpu_hours",
    "validation_perplexity",
    "sciq_acc_norm",
]].copy()
summary["token_saving_vs_raw"] = 1.0 - summary["cumulative_train_tokens"] / raw_tokens
summary["gpu_hour_saving_vs_raw"] = 1.0 - summary["cumulative_train_gpu_hours"] / raw_hours
summary["delta_sciq_acc_norm_vs_raw"] = summary["sciq_acc_norm"] - raw_acc
summary["delta_validation_perplexity_vs_raw"] = summary["validation_perplexity"] - raw_ppl
summary.to_csv(OUTPUT_DIR / "final-efficiency-summary.csv", index=False)
display(summary)
"""
        ),
        markdown_cell(
            """## 3. Plot perplexity and SciQ accuracy against tokens and GPU hours"""
        ),
        code_cell(
            """LABELS = {
    "raw": "Raw",
    "minhashlsh": "MinHashLSH",
    "lshbloom": "LSHBloom",
}
COLORS = {
    "raw": "#4C78A8",
    "minhashlsh": "#F58518",
    "lshbloom": "#54A24B",
}

plt.style.use("seaborn-v0_8-whitegrid")
figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
plot_specs = (
    ("cumulative_train_tokens", "validation_perplexity", "Cumulative training tokens (millions)", "Validation perplexity", 1e6),
    ("cumulative_train_tokens", "sciq_acc_norm", "Cumulative training tokens (millions)", "SciQ normalized accuracy (%)", 1e6),
    ("cumulative_train_gpu_hours", "validation_perplexity", "Cumulative training GPU hours", "Validation perplexity", 1.0),
    ("cumulative_train_gpu_hours", "sciq_acc_norm", "Cumulative training GPU hours", "SciQ normalized accuracy (%)", 1.0),
)

for axis, (x_column, y_column, x_label, y_label, x_divisor) in zip(axes.flat, plot_specs):
    for variant in VARIANTS:
        subset = frame[frame["variant"] == variant].sort_values("global_step")
        x_values = subset[x_column] / x_divisor
        if y_column == "sciq_acc_norm":
            y_values = subset[y_column] * 100.0
            y_errors = subset["sciq_acc_norm_stderr"] * 100.0
            axis.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                color=COLORS[variant],
                marker="o",
                markersize=4,
                linewidth=2,
                capsize=2,
                label=LABELS[variant],
            )
        else:
            y_values = subset[y_column]
            axis.plot(
                x_values,
                y_values,
                color=COLORS[variant],
                marker="o",
                markersize=4,
                linewidth=2,
                label=LABELS[variant],
            )
        axis.scatter(
            [x_values.iloc[-1]],
            [y_values.iloc[-1]],
            color=COLORS[variant],
            marker="D",
            s=42,
            zorder=4,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(f"{y_label} vs. {x_label.lower()}")
    axis.grid(True, alpha=0.25)

handles, labels = axes[0, 0].get_legend_handles_labels()
figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
figure.suptitle(
    "Qwen2.5-0.5B continued pretraining on complete peS2o variants",
    fontsize=15,
    y=1.03,
)
figure.text(
    0.5,
    -0.02,
    "Diamonds mark the end of one epoch. Accuracy bars are lm-eval standard errors.",
    ha="center",
    fontsize=10,
)

png_path = OUTPUT_DIR / "efficiency-curves.png"
pdf_path = OUTPUT_DIR / "efficiency-curves.pdf"
figure.savefig(png_path, dpi=200, bbox_inches="tight")
figure.savefig(pdf_path, bbox_inches="tight")
plt.show()
"""
        ),
        markdown_cell("""## 4. Upload the comparison to W&B and S3"""),
        code_cell(
            """wandb.login(key=WANDB_API_KEY)
run = wandb.init(
    project=CONFIG["wandb_project"],
    group=CONFIG["wandb_group"],
    name=CONFIG["wandb_run_name"],
    config={
        **CONFIG,
        "source_results": RESULT_KEYS,
        "experiment_fingerprint": experiment_fingerprint,
    },
)
run.log({
    "efficiency/curves": wandb.Image(str(png_path)),
    "efficiency/all_points": wandb.Table(dataframe=frame),
    "efficiency/final_summary": wandb.Table(dataframe=summary),
})

artifact = wandb.Artifact(
    name="qwen2.5-0.5b-pes2o-full-efficiency-curves",
    type="evaluation",
)
artifact.add_dir(str(OUTPUT_DIR))
run.log_artifact(artifact)

summary_prefix = f"{S3_PREFIX}summary/"
for local_path in OUTPUT_DIR.iterdir():
    if local_path.is_file():
        s3.upload_file(str(local_path), S3_BUCKET, f"{summary_prefix}{local_path.name}")

run.summary["results_s3_uri"] = f"s3://{S3_BUCKET}/{summary_prefix}"
run.summary["experiment_fingerprint"] = experiment_fingerprint
print("W&B:", run.url)
print("S3:", run.summary["results_s3_uri"])
wandb.finish()
"""
        ),
        markdown_cell(
            """## Reading the curves

A useful efficiency result has two properties: the deduplicated curve reaches the Raw final score at an earlier x position, and its diamond endpoint is within the predeclared acceptable quality loss. The plots alone do not remove training randomness; repeat the three runs with more seeds before making a strong claim.
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
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
