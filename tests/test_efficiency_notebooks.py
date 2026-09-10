import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_GENERATOR = (
    ROOT / "scripts" / "notebooks" / "build_efficiency_training_notebook.py"
)
PLOT_GENERATOR = ROOT / "scripts" / "notebooks" / "build_efficiency_plot_notebook.py"
TRAIN_NOTEBOOK = ROOT / "notebooks" / "qwen" / "qwen_pes2o_efficiency_training.ipynb"
PLOT_NOTEBOOK = ROOT / "notebooks" / "qwen" / "qwen_pes2o_efficiency_curves.ipynb"


def code_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


class EfficiencyNotebookTests(unittest.TestCase):
    def test_generators_create_compilable_colab_notebooks(self):
        for generator in (TRAIN_GENERATOR, PLOT_GENERATOR):
            result = subprocess.run(
                [sys.executable, str(generator)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        for path in (TRAIN_NOTEBOOK, PLOT_NOTEBOOK):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")
            for index, source in enumerate(code_cells(path)):
                compile(source, f"{path.name}-cell-{index}", "exec")

    def test_training_notebook_uses_full_variant_budget_and_records_curve_points(self):
        sources = code_cells(TRAIN_NOTEBOOK)
        code = "\n".join(sources)

        self.assertIn("build_full_training_plan", code)
        self.assertIn('manifest["variants"][name]["token_count"]', code)
        self.assertIn('"model_revision":', code)
        self.assertIn('revision=CONFIG["model_revision"]', code)
        self.assertIn('eval_strategy="steps"', code)
        self.assertIn('eval_steps=CONFIG["curve_interval_steps"]', code)
        self.assertIn('save_steps=CONFIG["curve_interval_steps"]', code)
        self.assertIn("cumulative_train_gpu_hours", code)
        self.assertIn('CONFIG["curve_validation_sequences"]', code)
        self.assertIn("simple_evaluate(", code)
        self.assertIn('CONFIG["sciq_examples"]', code)
        self.assertIn('CONFIG["sciq_smoke_examples"]', code)
        self.assertIn('"sciq_smoke_examples": 10', code)
        self.assertIn("bootstrap_iters=100", code)
        self.assertIn('"smoke/sciq_examples": smoke_metrics["examples"]', code)
        self.assertIn('"required_gpu_substring": "V100"', code)
        self.assertIn('if CONFIG["required_gpu_substring"] not in gpu_name:', code)
        self.assertGreaterEqual(code.count("torch.cuda.synchronize()"), 2)
        self.assertIn("merge_curve_measurements", code)
        self.assertNotIn("resolve/main", code)
        self.assertIn("VALIDATION_REVISION", code)
        self.assertIn("probe_sha256", code)
        self.assertIn("experiment_fingerprint", code)
        self.assertIn("source_manifest_sha256", code)
        self.assertIn("SCIQ_REVISION", code)
        self.assertIn('"dataset_kwargs": {"revision": SCIQ_REVISION}', code)
        self.assertIn("tasks=[SCIQ_TASK]", code)
        self.assertIn('lr_scheduler_type="constant_with_warmup"', code)
        self.assertIn('warmup_steps=CONFIG["warmup_steps"]', code)
        self.assertNotIn('warmup_ratio=CONFIG["warmup_ratio"]', code)
        self.assertIn("training-progress.json", code)
        self.assertIn("sciq-progress.json", code)
        self.assertGreaterEqual(code.count("s3.upload_file("), 5)
        self.assertLess(
            code.index('f"{S3_OUTPUT_PREFIX}final/{relative}"'),
            code.index("sciq_rows = []"),
        )
        self.assertIn("curve-results.json", code)
        self.assertIn("save_only_model=True", code)
        self.assertIn('np.stack([feature["input_ids"] for feature in features])', code)
        self.assertIn(
            "data=[[row[column] for column in CURVE_COLUMNS] for row in curve_rows]",
            code,
        )

    def test_plot_notebook_downloads_three_results_and_makes_four_real_data_plots(self):
        code = "\n".join(code_cells(PLOT_NOTEBOOK))

        for variant in ("raw", "minhashlsh", "lshbloom"):
            self.assertIn(f'"{variant}"', code)
        self.assertIn("validation_perplexity", code)
        self.assertIn("sciq_acc_norm", code)
        self.assertIn("cumulative_train_tokens", code)
        self.assertIn("cumulative_train_gpu_hours", code)
        self.assertIn("efficiency-curves.png", code)
        self.assertIn("wandb.Image", code)
        self.assertIn("artifact.add_dir", code)
        self.assertIn("validate_shared_experiment_results", code)

    def test_new_notebooks_use_english_text_and_chart_labels(self):
        han = re.compile(r"[\u3400-\u9fff]")
        for path in (TRAIN_NOTEBOOK, PLOT_NOTEBOOK):
            notebook_text = path.read_text(encoding="utf-8")
            self.assertIsNone(han.search(notebook_text), path.name)

    def test_generators_are_deterministic(self):
        before = {path: path.read_bytes() for path in (TRAIN_NOTEBOOK, PLOT_NOTEBOOK)}
        for generator in (TRAIN_GENERATOR, PLOT_GENERATOR):
            result = subprocess.run(
                [sys.executable, str(generator)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
