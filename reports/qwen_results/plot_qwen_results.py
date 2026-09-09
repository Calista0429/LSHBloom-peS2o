"""Create static figures and summary tables from the saved Qwen results."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


REPORT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = REPORT_DIR / "source"
REPO_ROOT = REPORT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.efficiency_curves import (  # noqa: E402
    validate_curve_rows,
    validate_shared_experiment_results,
)


VARIANTS = ("raw", "minhashlsh", "lshbloom")
LABELS = {
    "raw": "Raw",
    "minhashlsh": "MinHashLSH",
    "lshbloom": "LSHBloom",
}
COLORS = {
    "raw": "#4C78A8",
    "minhashlsh": "#F58518",
    "lshbloom": "#54A24B",
}


def load_json(name: str) -> dict:
    return json.loads((SOURCE_DIR / name).read_text(encoding="utf-8"))


def load_sciq() -> dict[str, dict[str, float]]:
    with (SOURCE_DIR / "fixed-sciq.csv").open(encoding="utf-8", newline="") as stream:
        return {
            row["variant"]: {
                key: int(value) if key == "examples" else float(value)
                for key, value in row.items()
                if key != "variant"
            }
            for row in csv.DictReader(stream)
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)


def main() -> None:
    fixed = {
        variant: load_json(f"fixed-{variant}.json") for variant in VARIANTS
    }
    fixed_configs = {
        json.dumps(result["config"], sort_keys=True) for result in fixed.values()
    }
    if len(fixed_configs) != 1:
        raise ValueError("Fixed-token configurations differ across variants")
    for source in ("s2orc", "s2ag"):
        document_id_sets = {
            tuple(result["validation"]["sources"][source]["document_ids"])
            for result in fixed.values()
        }
        predicted_token_counts = {
            result["validation"]["sources"][source]["predicted_tokens"]
            for result in fixed.values()
        }
        if len(document_id_sets) != 1 or len(predicted_token_counts) != 1:
            raise ValueError(f"Fixed-token {source} validation samples differ")

    sciq = load_sciq()
    fixed_rows = []
    for variant in VARIANTS:
        result = fixed[variant]
        overall = result["validation"]["overall"]
        sources = result["validation"]["sources"]
        fixed_rows.append(
            {
                "variant": variant,
                "train_input_tokens": result["config"]["train_input_tokens"],
                "train_runtime_seconds": result["training"]["train_runtime"],
                "train_loss": result["training"]["train_loss"],
                "validation_perplexity": overall["perplexity"],
                "s2orc_perplexity": sources["s2orc"]["perplexity"],
                "s2ag_perplexity": sources["s2ag"]["perplexity"],
                "sciq_acc": sciq[variant]["acc"],
                "sciq_acc_stderr": sciq[variant]["acc_stderr"],
                "sciq_acc_norm": sciq[variant]["acc_norm"],
                "sciq_acc_norm_stderr": sciq[variant]["acc_norm_stderr"],
            }
        )
    write_csv(REPORT_DIR / "fixed_token_summary.csv", fixed_rows)

    efficiency = {
        variant: load_json(f"efficiency-{variant}.json")
        for variant in ("raw", "minhashlsh")
    }
    validate_shared_experiment_results(
        efficiency.values(), expected_variants=("raw", "minhashlsh")
    )
    raw_probe = efficiency["raw"]["curve_probe"]
    minhash_probe = efficiency["minhashlsh"]["curve_probe"]
    if raw_probe["probe_sha256"] != minhash_probe["probe_sha256"]:
        raise ValueError("Validation probe hashes differ")
    if raw_probe["sequence_token_sha256"] != minhash_probe["sequence_token_sha256"]:
        raise ValueError("Validation sequence hashes differ")
    for source in ("s2orc", "s2ag"):
        raw_source = efficiency["raw"]["full_final_validation"]["sources"][source]
        minhash_source = efficiency["minhashlsh"]["full_final_validation"]["sources"][source]
        if raw_source["document_ids"] != minhash_source["document_ids"]:
            raise ValueError(f"Full-final {source} validation documents differ")
        if raw_source["predicted_tokens"] != minhash_source["predicted_tokens"]:
            raise ValueError(f"Full-final {source} predicted-token counts differ")

    curve_rows = []
    for variant, result in efficiency.items():
        rows = result["curve_probe"]["rows"]
        validate_curve_rows(rows, expected_variants=(variant,))
        curve_rows.extend(rows)
    curve_rows.sort(key=lambda row: (VARIANTS.index(row["variant"]), row["global_step"]))
    write_csv(REPORT_DIR / "efficiency_curve_partial.csv", curve_rows)

    endpoint_rows = []
    for variant, result in efficiency.items():
        plan = result["variant_plan"]
        final_curve = result["curve_probe"]["rows"][-1]
        full_final = result["full_final_validation"]["overall"]
        endpoint_rows.append(
            {
                "variant": variant,
                "train_input_tokens": plan["train_input_tokens"],
                "pure_training_gpu_hours": result["train_metrics"][
                    "pure_training_gpu_hours"
                ],
                "curve_probe_perplexity": final_curve["validation_perplexity"],
                "curve_probe_sciq_acc": final_curve["sciq_acc"],
                "curve_probe_sciq_acc_norm": final_curve["sciq_acc_norm"],
                "full_validation_perplexity": full_final["perplexity"],
            }
        )
    write_csv(REPORT_DIR / "efficiency_endpoint_summary_partial.csv", endpoint_rows)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#333333",
            "text.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    labels = [LABELS[variant] for variant in VARIANTS]
    colors = [COLORS[variant] for variant in VARIANTS]
    x = list(range(len(VARIANTS)))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    ppls = [row["validation_perplexity"] for row in fixed_rows]
    axes[0].scatter(x, ppls, c=colors, s=75, zorder=3)
    for index, value in enumerate(ppls):
        axes[0].annotate(f"{value:.4f}", (index, value), xytext=(0, 9), textcoords="offset points", ha="center")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Validation perplexity (lower is better)")
    axes[0].set_title("peS2o validation")
    axes[0].set_ylim(min(ppls) - 0.0015, max(ppls) + 0.0025)
    style_axis(axes[0])

    offsets = (-0.09, 0.09)
    metrics = (
        ("sciq_acc", "sciq_acc_stderr", "Accuracy", "o"),
        ("sciq_acc_norm", "sciq_acc_norm_stderr", "Normalized accuracy", "s"),
    )
    for offset, (column, error_column, label, marker) in zip(offsets, metrics):
        values = [row[column] * 100 for row in fixed_rows]
        errors = [row[error_column] * 100 for row in fixed_rows]
        axes[1].errorbar(
            [value + offset for value in x],
            values,
            yerr=errors,
            fmt=marker,
            color="#333333" if column == "sciq_acc_norm" else "#8A8A8A",
            markerfacecolor="white" if column == "sciq_acc_norm" else "#8A8A8A",
            markersize=6,
            capsize=3,
            linewidth=1.3,
            label=label,
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("SciQ score (%)")
    axes[1].set_title("SciQ, 1,000 examples")
    axes[1].legend(frameon=False, fontsize=9)
    style_axis(axes[1])

    width = 0.34
    s2orc = [row["s2orc_perplexity"] for row in fixed_rows]
    s2ag = [row["s2ag_perplexity"] for row in fixed_rows]
    axes[2].bar([value - width / 2 for value in x], s2orc, width, color="#72B7B2", label="S2ORC")
    axes[2].bar([value + width / 2 for value in x], s2ag, width, color="#E45756", label="S2AG")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Validation perplexity (lower is better)")
    axes[2].set_title("peS2o source breakdown")
    axes[2].legend(frameon=False)
    axes[2].set_ylim(9.5, 15.2)
    style_axis(axes[2])

    figure.suptitle("Qwen2.5-0.5B: equal-token comparison (24,999,936 tokens)", fontsize=15, fontweight="bold")
    figure.savefig(REPORT_DIR / "fixed_token_comparison.png", dpi=220, bbox_inches="tight")
    figure.savefig(REPORT_DIR / "fixed_token_comparison.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    plot_specs = (
        (axes[0, 0], "cumulative_train_tokens", "validation_perplexity", None, "Training tokens", "Probe perplexity (lower is better)"),
        (axes[0, 1], "cumulative_train_gpu_hours", "validation_perplexity", None, "Training GPU hours", "Probe perplexity (lower is better)"),
        (axes[1, 0], "cumulative_train_tokens", "sciq_acc_norm", "sciq_acc_norm_stderr", "Training tokens", "SciQ normalized accuracy (%)"),
        (axes[1, 1], "cumulative_train_gpu_hours", "sciq_acc_norm", "sciq_acc_norm_stderr", "Training GPU hours", "SciQ normalized accuracy (%)"),
    )
    for axis, x_column, y_column, error_column, x_label, y_label in plot_specs:
        for variant in ("raw", "minhashlsh"):
            rows = efficiency[variant]["curve_probe"]["rows"]
            x_values = [row[x_column] for row in rows]
            if x_column == "cumulative_train_tokens":
                x_values = [value / 1_000_000 for value in x_values]
            scale = 100 if y_column.startswith("sciq") else 1
            y_values = [row[y_column] * scale for row in rows]
            kwargs = {
                "color": COLORS[variant],
                "marker": "o",
                "markersize": 4.5,
                "linewidth": 2,
                "label": LABELS[variant],
            }
            if error_column:
                errors = [row[error_column] * scale for row in rows]
                axis.errorbar(x_values, y_values, yerr=errors, capsize=2, **kwargs)
            else:
                axis.plot(x_values, y_values, **kwargs)
            axis.scatter(x_values[-1], y_values[-1], color=COLORS[variant], marker="D", s=45, zorder=4)
        axis.set_xlabel(f"{x_label} (millions)" if x_column == "cumulative_train_tokens" else x_label)
        axis.set_ylabel(y_label)
        style_axis(axis)
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}M"))
    axes[1, 0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}M"))
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.subplots_adjust(top=0.87, bottom=0.12, hspace=0.34, wspace=0.22)
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Qwen2.5-0.5B: full-corpus efficiency curves",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.025,
        "Diamonds mark one-epoch endpoints. LSHBloom is pending because its curve result is absent from S3.",
        ha="center",
        fontsize=9.5,
    )
    figure.savefig(REPORT_DIR / "efficiency_curves_partial.png", dpi=220, bbox_inches="tight")
    figure.savefig(REPORT_DIR / "efficiency_curves_partial.pdf", bbox_inches="tight")
    plt.close(figure)

    raw_endpoint = next(row for row in endpoint_rows if row["variant"] == "raw")
    minhash_endpoint = next(row for row in endpoint_rows if row["variant"] == "minhashlsh")
    print(
        json.dumps(
            {
                "minhash_token_saving_percent": 100
                * (1 - minhash_endpoint["train_input_tokens"] / raw_endpoint["train_input_tokens"]),
                "minhash_gpu_hour_saving_percent": 100
                * (1 - minhash_endpoint["pure_training_gpu_hours"] / raw_endpoint["pure_training_gpu_hours"]),
                "minhash_full_validation_ppl_change_percent": 100
                * (minhash_endpoint["full_validation_perplexity"] / raw_endpoint["full_validation_perplexity"] - 1),
                "fixed_minhash_ppl_change_percent": 100 * (ppls[1] / ppls[0] - 1),
                "fixed_lshbloom_ppl_change_percent": 100 * (ppls[2] / ppls[0] - 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
