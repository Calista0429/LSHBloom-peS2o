import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_expected_efficiency_plot.py"


class ExpectedEfficiencyPlotTests(unittest.TestCase):
    def test_generates_a_valid_clearly_labeled_four_panel_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "expected.svg"
            result = subprocess.run(
                [sys.executable, str(GENERATOR), "--output", str(output)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            root = ET.parse(output).getroot()
            text = " ".join(element.text or "" for element in root.iter())
            paths = [element for element in root.iter() if element.tag.endswith("path")]

        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("SIMULATED DATA", text)
        self.assertIn("Validation perplexity vs. training tokens", text)
        self.assertIn("SciQ normalized accuracy vs. GPU hours", text)
        self.assertGreaterEqual(len(paths), 12)


if __name__ == "__main__":
    unittest.main()
