from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "lshbloom_pes2o"


def test_core_code_uses_a_named_src_package():
    expected = {
        "__init__.py",
        "dedup.py",
        "efficiency.py",
        "perplexity.py",
        "sciq.py",
        "training.py",
    }
    assert {path.name for path in PACKAGE.glob("*.py")} == expected
    assert not (ROOT / "src" / "__init__.py").exists()
