import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "qwen" / "qwen_pes2o_continued_pretraining.ipynb"
GENERATOR_PATH = ROOT / "scripts" / "notebooks" / "build_training_notebook.py"


class TrainingNotebookTests(unittest.TestCase):
    def load_notebook(self):
        return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    def combined_code(self):
        notebook = self.load_notebook()
        return "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
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
        notebook = self.load_notebook()
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{index}", "exec")

    def test_notebook_embeds_both_tested_helpers(self):
        code = self.combined_code()
        training_core = (ROOT / "src" / "lshbloom_pes2o" / "training.py").read_text()
        evaluation_core = (
            ROOT / "src" / "lshbloom_pes2o" / "perplexity.py"
        ).read_text()

        self.assertIn(training_core, code)
        self.assertIn(evaluation_core, code)

    def test_notebook_fixes_equal_training_budget_and_hyperparameters(self):
        code = self.combined_code()

        for required in (
            'VARIANT = "raw"',
            'ALLOWED_VARIANTS = {"raw", "minhashlsh", "lshbloom"}',
            '"sequence_length": 2048',
            '"sequence_count": 12_207',
            '"train_input_tokens": 24_999_936',
            '"per_device_train_batch_size": 1',
            '"gradient_accumulation_steps": 8',
            '"learning_rate": 5e-5',
            '"warmup_ratio": 0.03',
            '"weight_decay": 0.1',
            '"fp16": True',
            '"seed": 42',
            '"save_steps": 250',
            '"logging_steps": 10',
        ):
            self.assertIn(required, code)

    def test_notebook_uses_colab_secrets_wandb_and_s3_checksums(self):
        code = self.combined_code()

        for required in (
            'userdata.get("AWS_ACCESS_KEY_ID")',
            'userdata.get("AWS_SECRET_ACCESS_KEY")',
            'userdata.get("WANDB_API_KEY")',
            '"lshbloom-pes2o"',
            '"qwen2.5-0.5b-pes2o-dedup-25m"',
            "s3://calista-bucket/pes2o/v2/experiments/pilot-5000/",
            'manifest["variants"][VARIANT]["sha256"]',
            'raise RuntimeError("Dataset SHA-256 mismatch")',
            "checkpoints/{VARIANT}",
        ):
            self.assertIn(required, code)

    def test_notebook_smoke_tests_then_evaluates_fixed_validation_sample(self):
        code = self.combined_code()

        for required in (
            '"s2orc_documents": 320',
            '"s2ag_documents": 680',
            "validation-00000-of-00002.json.gz",
            "validation-00001-of-00002.json.gz",
            'wandb.log({"smoke_test/passed": 1',
            "trainer.train()",
            "combine_source_metrics(source_results.values())",
            '"eval/overall_perplexity"',
        ):
            self.assertIn(required, code)

    def test_fp16_training_keeps_model_parameters_in_fp32(self):
        code = self.combined_code()

        self.assertIn("torch_dtype=torch.float32", code)
        self.assertNotIn("torch_dtype=torch.float16", code)
        self.assertIn('with torch.autocast("cuda", dtype=torch.float16):', code)

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
