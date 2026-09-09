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


def test_experiment_entry_points_are_grouped_by_responsibility():
    notebook_builders = {
        "build_colab_notebook.py",
        "build_efficiency_plot_notebook.py",
        "build_efficiency_training_notebook.py",
        "build_plamo_training_notebook.py",
        "build_sciq_notebook.py",
        "build_training_notebook.py",
    }
    qwen_notebooks = {
        "qwen_pes2o_continued_pretraining.ipynb",
        "qwen_pes2o_efficiency_curves.ipynb",
        "qwen_pes2o_efficiency_training.ipynb",
        "qwen_pes2o_sciq_evaluation.ipynb",
        "qwen_pes2o_validation_perplexity.ipynb",
    }
    assert (ROOT / "scripts" / "data" / "prepare_pes2o_variants.py").is_file()
    assert {
        path.name for path in (ROOT / "scripts" / "notebooks").glob("*.py")
    } == notebook_builders
    assert {
        path.name for path in (ROOT / "notebooks" / "qwen").glob("*.ipynb")
    } == qwen_notebooks
    assert {
        path.name for path in (ROOT / "notebooks" / "plamo").glob("*.ipynb")
    } == {"plamo2_1b_pes2o_continued_pretraining.ipynb"}
    assert not list((ROOT / "scripts").glob("build_*_notebook.py"))
    assert not list((ROOT / "notebooks").glob("*.ipynb"))
