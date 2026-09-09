# Public Repository Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize LSHBloom-peS2o into a readable, installable, locally validated research repository and publish the reviewed result on GitHub.

**Architecture:** Core behavior moves into the named `lshbloom_pes2o` package under a standard `src` layout. Executable helpers are grouped into data, notebook, and analysis scripts; generated artifacts are grouped by model or report. Local Ruff, Pytest, deterministic-generation, artifact, and credential checks replace hosted CI.

**Tech Stack:** Python 3.10+, datasketch 2.0, NumPy, Ruff, Pytest, Jupyter Notebook JSON, Git, GitHub CLI.

**Spec:** `docs/superpowers/specs/2026-09-09-public-repository-reorganization-design.md`

## Global Constraints

- Preserve all deduplication parameters, model-training settings, S3 object paths, W&B run references, and reported aggregate metric values.
- Do not add `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, GitHub Actions, or other hosted CI.
- Do not commit datasets, checkpoints, private S3 result JSON, credentials, Notebook outputs, or execution counts.
- Keep Colab-specific CUDA dependencies pinned inside generated notebooks.
- Keep all code, comments, chart labels, and repository documentation in English.
- Change GitHub visibility only after every local verification step succeeds.

---

### Task 1: Establish the Named Python Package

**Files:**
- Create: `pyproject.toml`
- Create: `tests/test_repository_layout.py`
- Rename: `src/__init__.py` to `src/lshbloom_pes2o/__init__.py`
- Rename: `src/pes2o_dedup.py` to `src/lshbloom_pes2o/dedup.py`
- Rename: `src/pes2o_perplexity.py` to `src/lshbloom_pes2o/perplexity.py`
- Rename: `src/pes2o_training.py` to `src/lshbloom_pes2o/training.py`
- Rename: `src/sciq_evaluation.py` to `src/lshbloom_pes2o/sciq.py`
- Rename: `src/efficiency_curves.py` to `src/lshbloom_pes2o/efficiency.py`
- Modify: `tests/test_pes2o_dedup.py`
- Modify: `tests/test_pes2o_perplexity.py`
- Modify: `tests/test_pes2o_training.py`
- Modify: `tests/test_sciq_evaluation.py`
- Modify: `tests/test_efficiency_curves.py`

**Interfaces:**
- Consumes: Existing public functions and classes from the five `src/*.py` modules.
- Produces: Equivalent imports under `lshbloom_pes2o.{dedup,perplexity,training,sciq,efficiency}` and an installable editable package.

- [ ] **Step 1: Record the pre-migration test baseline**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all existing tests pass before files move.

- [ ] **Step 2: Add a failing package-layout test**

Create `tests/test_repository_layout.py` with:

```python
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
```

- [ ] **Step 3: Verify the layout test fails**

Run:

```bash
pytest tests/test_repository_layout.py -q
```

Expected: failure because `src/lshbloom_pes2o` does not exist.

- [ ] **Step 4: Move the core modules and update internal imports**

Use Git-aware moves. Change `from src.pes2o_perplexity import validate_record` in `dedup.py` to:

```python
from lshbloom_pes2o.perplexity import validate_record
```

Update the five core test modules and patch targets from `src.<module>` to the corresponding `lshbloom_pes2o.<module>` import.

- [ ] **Step 5: Add project and quality-tool configuration**

Create `pyproject.toml` with setuptools package discovery from `src`, project name `lshbloom-pes2o`, version `0.1.0`, Python requirement `>=3.10`, runtime dependencies `datasketch[bloom]==2.0.0` and `numpy>=1.26`, an `analysis` extra containing `matplotlib>=3.8`, and a `dev` extra containing `pytest>=8` and `ruff>=0.12`. Configure Pytest to use `tests`, Ruff for Python 3.10 and an 88-character line length, and Ruff rules `E4`, `E7`, `E9`, `F`, and `I`.

- [ ] **Step 6: Install and verify the migrated package**

Run:

```bash
python3 -m pip install -e '.[dev,analysis]'
pytest tests/test_repository_layout.py tests/test_pes2o_dedup.py tests/test_pes2o_perplexity.py tests/test_pes2o_training.py tests/test_sciq_evaluation.py tests/test_efficiency_curves.py -q
```

Expected: the layout test and all core behavior tests pass.

- [ ] **Step 7: Commit the package migration**

```bash
git add pyproject.toml src tests
git commit -m "refactor: package core experiment utilities"
```

---

### Task 2: Group Scripts and Generated Notebooks

**Files:**
- Rename: `scripts/prepare_pes2o_variants.py` to `scripts/data/prepare_pes2o_variants.py`
- Rename: six `scripts/build_*_notebook.py` files to `scripts/notebooks/`
- Rename: `notebooks/plamo2_1b_pes2o_continued_pretraining.ipynb` to `notebooks/plamo/`
- Rename: five Qwen notebooks to `notebooks/qwen/`
- Modify: all moved notebook generators
- Modify: `tests/test_colab_notebook.py`
- Modify: `tests/test_efficiency_notebooks.py`
- Modify: `tests/test_pes2o_dedup.py`
- Modify: `tests/test_plamo_training_notebook.py`
- Modify: `tests/test_sciq_notebook.py`
- Modify: `tests/test_training_notebook.py`

**Interfaces:**
- Consumes: Named package paths from Task 1.
- Produces: Repository-root executable scripts and deterministic notebooks in model-specific directories.

- [ ] **Step 1: Extend the layout test for script and notebook groups**

Add assertions for `scripts/data/prepare_pes2o_variants.py`, six files under `scripts/notebooks`, five files under `notebooks/qwen`, and one file under `notebooks/plamo`. Assert that the former flat paths do not exist.

- [ ] **Step 2: Verify the new assertions fail**

Run `pytest tests/test_repository_layout.py -q` and expect failures for the flat layout.

- [ ] **Step 3: Move scripts and notebooks**

Use `git mv`. In every moved script, change the repository root expression from `Path(__file__).resolve().parents[1]` to `Path(__file__).resolve().parents[2]`. Point embedded core-source reads to `src/lshbloom_pes2o/*.py` and output paths to `notebooks/qwen/` or `notebooks/plamo/`.

- [ ] **Step 4: Update data CLI and test paths**

Change the data script import to:

```python
from lshbloom_pes2o.dedup import DedupConfig, prepare_variants
```

Update subprocess targets, generator constants, Notebook constants, and source-file comparisons in the affected tests.

- [ ] **Step 5: Regenerate all six notebooks**

Run each script under `scripts/notebooks/` with Python. Verify the generated notebook JSON compiles and contains no execution counts or outputs.

- [ ] **Step 6: Run script and notebook tests**

Run:

```bash
pytest tests/test_colab_notebook.py tests/test_efficiency_notebooks.py tests/test_pes2o_dedup.py tests/test_plamo_training_notebook.py tests/test_sciq_notebook.py tests/test_training_notebook.py -q
```

Expected: all tests pass with the grouped paths.

- [ ] **Step 7: Commit the grouped experiment entry points**

```bash
git add scripts notebooks tests
git commit -m "refactor: organize experiment scripts and notebooks"
```

---

### Task 3: Separate Analysis Code and Report Artifacts

**Files:**
- Rename: `scripts/generate_expected_efficiency_plot.py` to `scripts/analysis/generate_expected_efficiency_plot.py`
- Rename: `reports/qwen_results/plot_qwen_results.py` to `scripts/analysis/plot_qwen_results.py`
- Rename: `reports/qwen_results/` to `reports/qwen/`
- Rename: `figures/expected-efficiency-curves.svg` to `docs/assets/expected-efficiency-curves.svg`
- Modify: `scripts/analysis/plot_qwen_results.py`
- Modify: `scripts/analysis/generate_expected_efficiency_plot.py`
- Modify: `tests/test_expected_efficiency_plot.py`
- Modify: `tests/test_repository_layout.py`

**Interfaces:**
- Consumes: Validated local result JSON files under ignored `reports/qwen/source/*.json`.
- Produces: Versioned aggregate CSV, PNG, PDF, and English report files under `reports/qwen/`.

- [ ] **Step 1: Extend the layout test for analysis and report locations**

Assert the two analysis scripts, `reports/qwen/README.md`, and `docs/assets/expected-efficiency-curves.svg` exist. Assert `figures`, `reports/qwen_results`, and the old script paths do not exist.

- [ ] **Step 2: Verify the new assertions fail**

Run `pytest tests/test_repository_layout.py -q` and expect failures for the old analysis layout.

- [ ] **Step 3: Move analysis code, reports, and assets**

Use Git-aware moves. Set the result plotter paths to:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "qwen"
SOURCE_DIR = REPORT_DIR / "source"
```

Set the simulated figure generator default output to `docs/assets/expected-efficiency-curves.svg`.

- [ ] **Step 4: Update tests and ignored source paths**

Update the expected-figure test to the new script and asset paths. Change `.gitignore` from `reports/qwen_results/source/*.json` to `reports/qwen/source/*.json` and retain all private source JSON as ignored local files.

- [ ] **Step 5: Regenerate and compare report artifacts**

Run both analysis scripts. Compare the three aggregate CSV files against the pre-move versions and confirm metric rows are unchanged. Check both PNG and PDF files with `file` and visually inspect the PNG charts.

- [ ] **Step 6: Run analysis tests and commit**

Run `pytest tests/test_expected_efficiency_plot.py tests/test_repository_layout.py -q`, then:

```bash
git add .gitignore docs reports scripts tests
git commit -m "refactor: separate analysis code and reports"
```

---

### Task 4: Rewrite Public-Facing Documentation

**Files:**
- Modify: `README.md`
- Modify: `reports/qwen/README.md`
- Modify: `reports/qwen/source/README.md`
- Delete: `requirements-dedup.txt`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Final paths and commands from Tasks 1–3.
- Produces: A concise public project entry point and complete local reproduction commands.

- [ ] **Step 1: Add documentation checks to the layout test**

Assert `README.md` links to `reports/qwen/README.md`, `notebooks/qwen`, `notebooks/plamo`, and `docs/assets/expected-efficiency-curves.svg`. Assert `requirements-dedup.txt`, `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, and `.github/workflows` do not exist.

- [ ] **Step 2: Rewrite the root README**

Use this order: project purpose, headline measured result, methodology, installation, repository map, data preparation, Qwen experiments, PLaMo experiment, result interpretation, local quality checks, and limitations. Keep exact S3 paths only where required for reproduction and retain the PyTorch-version caveat for the LSHBloom curve.

- [ ] **Step 3: Update report and source instructions**

Update all moved links and place the exact five fixed/efficiency JSON filenames required by the plotter in `reports/qwen/source/README.md`. Keep JSON values ignored by Git.

- [ ] **Step 4: Expand `.gitignore` and remove the old requirement file**

Ignore `.venv`, Python caches, Ruff/Pytest caches, coverage files, `.env*`, local data, checkpoints, model weights, memory maps, macOS metadata, and private source JSON. Delete `requirements-dedup.txt` because `pyproject.toml` is the dependency source.

- [ ] **Step 5: Validate links and commands**

Run a local script that extracts relative Markdown links from both README files and fails for missing repository paths. Run every documented local command that does not require cloud credentials or a GPU.

- [ ] **Step 6: Commit public documentation**

```bash
git add .gitignore README.md reports tests pyproject.toml
git commit -m "docs: prepare public project documentation"
```

---

### Task 5: Apply and Verify Local Quality Standards

**Files:**
- Modify: Python files changed by `ruff format`
- Modify: import ordering reported by `ruff check --fix`

**Interfaces:**
- Consumes: The complete reorganized tree.
- Produces: A clean, locally reproducible release candidate.

- [ ] **Step 1: Format and lint maintained Python code**

Run:

```bash
ruff format src scripts tests
ruff check --fix src scripts tests
ruff format --check src scripts tests
ruff check src scripts tests
```

Expected: both final check commands exit successfully.

- [ ] **Step 2: Run the complete test suite**

Run `pytest -q` and expect every test to pass.

- [ ] **Step 3: Verify deterministic generated artifacts**

Regenerate all Notebook and SVG artifacts, then run:

```bash
git diff --exit-code -- notebooks docs/assets/expected-efficiency-curves.svg
```

Expected: no difference.

- [ ] **Step 4: Verify reports and repository hygiene**

Confirm aggregate CSV hashes or parsed rows match the pre-migration snapshot. Confirm every Notebook has zero outputs and zero non-null execution counts. Run `git diff --check` and confirm `git status --short` contains only intended quality changes.

- [ ] **Step 5: Run credential scans**

Scan the working tree and every revision returned by `git rev-list --all` for AWS access-key IDs, Hugging Face tokens, GitHub tokens, and private-key blocks. Print only matching revision and filename metadata, never secret values. Expected: zero matches.

- [ ] **Step 6: Commit formatting changes if needed**

If Ruff changed tracked files, commit only those changes:

```bash
git add src scripts tests
git commit -m "style: apply repository formatting standards"
```

---

### Task 6: Merge and Publish the Repository

**Files:**
- No source-file changes expected.

**Interfaces:**
- Consumes: Clean, verified release candidate from Task 5.
- Produces: The same commit on GitHub `main` with `PUBLIC` visibility.

- [ ] **Step 1: Push the reviewed feature branch**

Run `git push origin codex/qwen-pes2o-perplexity` and verify the remote branch points at the local HEAD.

- [ ] **Step 2: Fast-forward main**

In the clean `/Users/huangbaoxi/Code/LSHBloom-peS2o` clone, fetch the feature branch, check out `main`, pull `origin/main` with `--ff-only`, merge the feature branch with `--ff-only`, and push `main`.

- [ ] **Step 3: Verify main before visibility change**

Check that local `main`, `origin/main`, and the verified release commit have the same SHA. Run `gh repo view Calista0429/LSHBloom-peS2o --json visibility,defaultBranchRef` and confirm the default branch is `main` and visibility is still `PRIVATE`.

- [ ] **Step 4: Make the repository public**

Run:

```bash
gh repo edit Calista0429/LSHBloom-peS2o --visibility public --accept-visibility-change-consequences
```

- [ ] **Step 5: Verify the public release**

Run `gh repo view Calista0429/LSHBloom-peS2o --json visibility,isPrivate,url,defaultBranchRef` and require `visibility` to be `PUBLIC`, `isPrivate` to be `false`, and the default branch to be `main`. Open the public repository URL and verify the README and result image resolve without authentication.
