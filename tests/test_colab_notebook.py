import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "qwen" / "qwen_pes2o_validation_perplexity.ipynb"
CORE = ROOT / "src" / "lshbloom_pes2o" / "perplexity.py"


class ColabNotebookTests(unittest.TestCase):
    def load_notebook(self):
        return json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def code_cells(self, notebook):
        return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    def test_notebook_is_valid_v4_json_with_gpu_metadata(self):
        notebook = self.load_notebook()

        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertEqual(notebook["metadata"]["kernelspec"]["language"], "python")

    def test_every_code_cell_compiles(self):
        notebook = self.load_notebook()

        for index, cell in enumerate(self.code_cells(notebook)):
            compile("".join(cell["source"]), f"cell-{index}", "exec")

    def test_notebook_embeds_the_tested_core_module(self):
        notebook = self.load_notebook()
        core_cells = [
            cell
            for cell in self.code_cells(notebook)
            if "core-library" in cell.get("metadata", {}).get("tags", [])
        ]

        self.assertEqual(len(core_cells), 1)
        self.assertEqual("".join(core_cells[0]["source"]), CORE.read_text(encoding="utf-8"))

    def test_notebook_contains_required_evaluation_and_wandb_settings(self):
        notebook = self.load_notebook()
        source = "\n".join("".join(cell["source"]) for cell in self.code_cells(notebook))

        required_fragments = [
            '"model_id": "Qwen/Qwen2.5-0.5B"',
            '"s2orc_documents": 320',
            '"s2ag_documents": 680',
            "validation-00000-of-00002.json.gz",
            "validation-00001-of-00002.json.gz",
            "torch.float16",
            "torch.inference_mode()",
            "wandb.login()",
            "wandb.Artifact",
            "wandb.finish()",
            "combine_source_metrics",
            "write_result_json",
            "if limit is not None and actual_counts[source] != limit",
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

        self.assertNotIn("from src", source)
        self.assertNotIn("import src", source)

    def test_notebook_has_install_instructions_and_result_explanation(self):
        notebook = self.load_notebook()
        all_source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
        )

        self.assertIn("pip install", all_source)
        self.assertIn("This notebook evaluates; it does not train", all_source)
        self.assertIn("/content/results/", all_source)


if __name__ == "__main__":
    unittest.main()
