import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "build_plamo_training_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "plamo2_1b_pes2o_continued_pretraining.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


class PlamoTrainingNotebookTests(unittest.TestCase):
    def test_generator_creates_compilable_english_colab_notebook(self):
        result = subprocess.run(
            [sys.executable, str(GENERATOR)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertEqual(notebook["metadata"]["colab"]["gpuType"], "V100")
        for index, source in enumerate(code_cells()):
            compile(source, f"plamo-cell-{index}", "exec")
        self.assertIsNone(re.search(r"[\u3400-\u9fff]", NOTEBOOK.read_text()))

    def test_uses_plamo_official_runtime_requirements_and_remote_code(self):
        code = "\n".join(code_cells())

        for required in (
            '"model_id": "pfnet/plamo-2-1b"',
            '"model_revision": "92c75fd6eea9018bcb9c33ee8921589febe071fa"',
            '"required_gpu_substring": "V100"',
            'trust_remote_code=True',
            '"torch_version": "2.5.1"',
            '"transformers_version": "4.57.1"',
            '"mamba_ssm_version": "2.2.4"',
            '"causal_conv1d_version": "1.4.0"',
            'import mamba_ssm',
            'import causal_conv1d',
        ):
            self.assertIn(required, code)

    def test_keeps_the_equal_twenty_five_million_token_protocol(self):
        code = "\n".join(code_cells())

        for required in (
            'VARIANT = "raw"',
            'ALLOWED_VARIANTS = {"raw", "minhashlsh", "lshbloom"}',
            '"sequence_length": 2048',
            '"sequence_count": 12_207',
            '"train_input_tokens": 24_999_936',
            '"gradient_accumulation_steps": 8',
            '"learning_rate": 5e-5',
            '"warmup_ratio": 0.03',
            '"seed": 42',
            'optim="adafactor"',
        ):
            self.assertIn(required, code)

        self.assertNotIn(
            'variant_info["token_count"] < CONFIG["train_input_tokens"]', code
        )
        self.assertIn("pack_jsonl_gz_to_memmap(", code)

    def test_preflights_and_persists_before_full_validation(self):
        code = "\n".join(code_cells())

        for required in (
            '"smoke_test/passed": 1',
            "smoke_output.loss.backward()",
            "trainer.train()",
            '"s2orc_documents": 320',
            '"s2ag_documents": 680',
            "VALIDATION_REVISION",
            "combine_source_metrics(source_results.values())",
            'plamo-2-1b-25m/checkpoints/{VARIANT}',
            "del trainer",
            "s3.put_object(",
            "s3.delete_object(",
        ):
            self.assertIn(required, code)

        self.assertLess(
            code.index('f"{checkpoint_prefix}/final/{relative}"'),
            code.index("records_by_source = collect_source_records"),
        )

    def test_generator_is_deterministic(self):
        before = NOTEBOOK.read_bytes()
        result = subprocess.run(
            [sys.executable, str(GENERATOR)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(NOTEBOOK.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
