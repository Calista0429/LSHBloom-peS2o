import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_GENERATOR = ROOT / "scripts" / "build_efficiency_training_notebook.py"
PLOT_GENERATOR = ROOT / "scripts" / "build_efficiency_plot_notebook.py"
TRAIN_NOTEBOOK = ROOT / "notebooks" / "qwen_pes2o_efficiency_training.ipynb"
PLOT_NOTEBOOK = ROOT / "notebooks" / "qwen_pes2o_efficiency_curves.ipynb"


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
        self.assertIn('eval_strategy="steps"', code)
        self.assertIn('eval_steps=CONFIG["curve_interval_steps"]', code)
        self.assertIn('save_steps=CONFIG["curve_interval_steps"]', code)
        self.assertIn("cumulative_train_gpu_hours", code)
        self.assertIn('CONFIG["curve_validation_sequences"]', code)
        self.assertIn("simple_evaluate(", code)
        self.assertIn('CONFIG["sciq_examples"]', code)
        self.assertIn("merge_curve_measurements", code)
        self.assertIn("curve-results.json", code)
        self.assertIn("save_only_model=True", code)
        self.assertIn(
            'data=[[row[column] for column in CURVE_COLUMNS] for row in curve_rows]',
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

    def test_generators_are_deterministic(self):
        before = {
            path: path.read_bytes()
            for path in (TRAIN_NOTEBOOK, PLOT_NOTEBOOK)
        }
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
