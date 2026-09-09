# Public Repository Reorganization Design

## Purpose

Prepare `Calista0429/LSHBloom-peS2o` for public visibility and make the research code easier to understand, run, test, and extend. The reorganization must preserve the implemented deduplication algorithms, experiment configurations, saved aggregate results, and generated Colab behavior.

## Scope

The change will:

- replace the generic `src` package with an installable `lshbloom_pes2o` package;
- group data preparation, notebook generation, and analysis scripts by responsibility;
- group notebooks by model family;
- separate generated reports from their plotting code;
- add project metadata, dependency groups, Ruff configuration, and Pytest configuration through `pyproject.toml`;
- rewrite the root README around installation, repository structure, experiment reproduction, results, and limitations;
- scan the working tree and full Git history for credentials before public release;
- merge the reviewed branch into `main`, push it, and change the GitHub repository visibility to public only after all checks pass.

The change will not:

- alter LSHBloom or MinHashLSH parameters;
- rerun training or change reported metrics;
- upload datasets, checkpoints, private S3 result JSON, or credentials;
- add `LICENSE`, `CONTRIBUTING.md`, or `SECURITY.md`;
- add GitHub Actions or other hosted continuous-integration workflows;
- rewrite old Git history unless the final credential scan finds a real secret.

Without a `LICENSE` file, public visibility makes the source readable but does not grant a general license to copy, modify, or redistribute it.

## Target Layout

```text
LSHBloom-peS2o/
├── docs/
│   ├── assets/
│   │   └── expected-efficiency-curves.svg
│   └── superpowers/
│       ├── plans/
│       └── specs/
├── notebooks/
│   ├── plamo/
│   │   └── plamo2_1b_pes2o_continued_pretraining.ipynb
│   └── qwen/
│       ├── qwen_pes2o_continued_pretraining.ipynb
│       ├── qwen_pes2o_efficiency_curves.ipynb
│       ├── qwen_pes2o_efficiency_training.ipynb
│       ├── qwen_pes2o_sciq_evaluation.ipynb
│       └── qwen_pes2o_validation_perplexity.ipynb
├── reports/
│   └── qwen/
│       ├── README.md
│       ├── *.csv
│       ├── *.pdf
│       ├── *.png
│       └── source/
├── scripts/
│   ├── analysis/
│   │   ├── generate_expected_efficiency_plot.py
│   │   └── plot_qwen_results.py
│   ├── data/
│   │   └── prepare_pes2o_variants.py
│   └── notebooks/
│       └── build_*.py
├── src/
│   └── lshbloom_pes2o/
│       ├── __init__.py
│       ├── dedup.py
│       ├── efficiency.py
│       ├── perplexity.py
│       ├── sciq.py
│       └── training.py
├── tests/
│   └── test_*.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Package Boundaries

`lshbloom_pes2o.dedup` owns document shingling, MinHash construction, duplicate decisions, variant generation, and manifests. `perplexity` owns peS2o validation loading, packing, and metric aggregation. `training` owns fixed-token packing and memory-mapped training datasets. `sciq` owns extraction and comparison of SciQ metrics. `efficiency` owns training-plan accounting, experiment identity validation, and curve tables.

The package will expose only stable, useful entry points from `lshbloom_pes2o.__init__`; tests and scripts will import the responsible module directly when they need lower-level helpers. No module will depend on a script or Notebook.

## Scripts and Notebooks

Scripts remain executable from the repository root. Each script will derive the root path from its own location, use the installed package interface, and provide a working `--help` path where applicable.

Notebook generators remain the source of truth for generated notebooks. Their output paths will move with the notebooks, and determinism tests will compare generated content with the checked-in `.ipynb` files. Generated notebooks will remain free of execution counts and outputs. README links and Colab instructions will be updated to the new paths.

Large notebook generators will not be refactored internally during this migration. Moving and validating them is sufficient for this repository-quality pass; splitting their embedded code is a separate behavioral refactor with higher regression risk.

## Dependencies and Tooling

`pyproject.toml` will require Python 3.10 or newer and define:

- runtime dependencies needed by the importable package, including NumPy and `datasketch[bloom]`;
- optional development dependencies for Pytest and Ruff;
- Pytest test discovery;
- Ruff lint and formatting rules with a focused rule set compatible with the existing code;
- a console entry point for peS2o variant preparation if it can reuse the existing CLI without changing behavior.

Colab-specific training dependencies will remain pinned inside the generated notebooks because their GPU and CUDA compatibility differs from lightweight local development. The old `requirements-dedup.txt` will be removed after its dependency is represented in `pyproject.toml`.

## Local Quality Checks

The README will document one local release-check sequence that installs development dependencies, runs `ruff format --check`, runs `ruff check`, executes the complete test suite, regenerates every Notebook and the simulated SVG, and fails if tracked artifacts change. These checks will not require AWS, W&B, Hugging Face credentials, GPUs, or external experiment data.

## Documentation

The root README will lead with the research question and measured outcome, then provide installation, a short architecture map, data preparation, Colab experiment instructions, result interpretation, repository layout, and known limitations. Detailed result tables and figures remain in `reports/qwen/README.md`. Historical design and plan documents remain under `docs/superpowers` as development records.

Every command and path in the README will be checked against the final tree. References to private credentials will describe Colab secret names only and will never include values.

## Public-Release Safety

Before changing visibility, the release check will confirm:

- no tracked model files, datasets, memory maps, `.env` files, credentials, or private result JSON;
- no executed Notebook output or embedded token values;
- no AWS access-key IDs, Hugging Face tokens, GitHub tokens, or private-key blocks anywhere in Git history;
- `.gitignore` covers local environments, caches, data, checkpoints, result source JSON, and common editor or operating-system files;
- the branch is clean and all local checks pass.

If a real secret is found, the repository will remain private until the secret is revoked and removed from history. Mere secret-variable names such as `AWS_SECRET_ACCESS_KEY` are documentation and are permitted.

## Migration and Compatibility

Moves will use Git-aware renames where practical. All internal imports, subprocess test paths, generator output paths, documentation links, and report asset links will change in the same migration. Aggregate result values will be compared before and after the move. Existing S3 object paths and W&B runs will not change.

This repository currently has no external public API because it is private. The new package path becomes the supported interface after publication; compatibility shims for imports from `src.*` will not be added.

## Verification and Release

Success requires:

- package installation in a clean environment;
- Ruff formatting and lint checks passing;
- all existing tests passing after path updates;
- deterministic Notebook and SVG regeneration producing no Git diff;
- the Qwen report CSV values and image dimensions remaining unchanged unless a path-only metadata change requires regeneration;
- a clean credential scan of the working tree and Git history;
- the latest result commit present on `main`;
- GitHub reporting `PUBLIC` visibility after the final `gh repo edit --visibility public` operation.
