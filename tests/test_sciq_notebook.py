import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "qwen" / "qwen_pes2o_sciq_evaluation.ipynb"
GENERATOR_PATH = ROOT / "scripts" / "notebooks" / "build_sciq_notebook.py"


class SciQNotebookTests(unittest.TestCase):
    def load_notebook(self):
        return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    def combined_code(self):
        return "\n".join(
            "".join(cell["source"])
            for cell in self.load_notebook()["cells"]
            if cell["cell_type"] == "code"
        )

    def test_notebook_and_generator_exist(self):
        self.assertTrue(GENERATOR_PATH.is_file())
        self.assertTrue(NOTEBOOK_PATH.is_file())

    def test_notebook_is_valid_colab_v4_json(self):
        notebook = self.load_notebook()

        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["nbformat_minor"], 5)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")

    def test_every_code_cell_compiles(self):
        for index, cell in enumerate(self.load_notebook()["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{index}", "exec")

    def test_notebook_embeds_tested_result_helpers(self):
        source = (ROOT / "src" / "lshbloom_pes2o" / "sciq.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(source, self.combined_code())

    def test_notebook_has_exact_sciq_only_configuration(self):
        code = self.combined_code()
        required = (
            '"lm-eval[hf]==0.4.13"',
            '"task": "sciq"',
            '"num_fewshot": 0',
            '"batch_size": 8',
            '"smoke_limit": 10',
            '"full_limit": None',
            '"dtype": "float16"',
            "apply_chat_template=False",
            'tasks=[CONFIG["task"]]',
            "simple_evaluate(",
        )
        for item in required:
            self.assertIn(item, code)

        for excluded_task in (
            "mmlu",
            "hellaswag",
            "piqa",
            "winogrande",
            "boolq",
            "arc_easy",
            "arc_challenge",
            "openbookqa",
        ):
            self.assertNotIn(f'"{excluded_task}"', code)

    def test_notebook_uses_exact_three_checkpoint_prefixes(self):
        code = self.combined_code()

        for variant in ("raw", "minhashlsh", "lshbloom"):
            self.assertIn(
                "s3://calista-bucket/pes2o/v2/experiments/pilot-5000/"
                f"checkpoints/{variant}/final/",
                code,
            )

    def test_notebook_reads_secrets_without_printing_values(self):
        code = self.combined_code()
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "WANDB_API_KEY",
        ):
            self.assertIn(f'userdata.get("{name}")', code)
            self.assertIsNone(
                re.search(rf"print\([^\n]*\b{name}\b", code),
                f"secret {name} must not be printed",
            )

    def test_notebook_preflights_and_cleans_each_checkpoint(self):
        code = self.combined_code()

        for required in (
            '"config.json"',
            '"model.safetensors"',
            '"tokenizer_config.json"',
            '"tokenizer.json"',
            "list_objects_v2",
            "download_checkpoint(",
            "shutil.rmtree(local_dir",
            "torch.cuda.empty_cache()",
        ):
            self.assertIn(required, code)

    def test_notebook_saves_results_and_uploads_one_wandb_artifact(self):
        code = self.combined_code()

        for required in (
            'RESULTS_DIR / f"{variant}-{stage}.json"',
            'RESULTS_DIR / "comparison.csv"',
            'RESULTS_DIR / "summary.json"',
            'RESULTS_DIR / "failures.json"',
            "wandb.Table(dataframe=comparison_frame)",
            "wandb.Artifact(",
            "artifact.add_dir(str(RESULTS_DIR))",
            "wandb.finish()",
        ):
            self.assertIn(required, code)

    def test_generator_is_deterministic(self):
        before = NOTEBOOK_PATH.read_bytes()
        result = subprocess.run(
            [sys.executable, str(GENERATOR_PATH)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(NOTEBOOK_PATH.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
