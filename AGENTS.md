# Repository Guidelines

## Project Structure & Module Organization

Core Python packages live under `src/`. `src/rlnas_od/` contains the earlier RL architecture-search scaffold, while `src/rl_finetune/` contains layer-wise fine-tuning utilities. Executable training, evaluation, dataset-preparation, and summarization entry points belong in `scripts/`. Unit tests are in `tests/`; use matching names such as `tests/test_search_space.py` for `src/rlnas_od/search_space.py`. Research methodology and reproducible results live in `docs/`. Treat `runs/`, downloaded datasets, model weights, virtual environments, and local dataset YAML files as generated or machine-specific artifacts; do not commit them.

## Build, Test, and Development Commands

Create the supported Python 3.9+ environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the complete unit suite with `python -m pytest tests`. Use `python -m pytest tests/test_final_evaluation.py -q` for focused iteration. Reproduce the final dataset workflow with `bash scripts/run_final_new_dataset_benchmarks.sh train`; this requires prepared local datasets and detector weights. Individual experiment scripts expose their options through `python scripts/<name>.py --help`. Apple Silicon runs generally use `--device mps`; TorchVision detection workflows may be more reliable on CPU.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` for functions, variables, modules, and CLI flags, and `PascalCase` for classes. Add type hints where they clarify data flow, and keep experiment configuration explicit through CLI arguments. Shell scripts should use strict mode and quote expansions; preserve compatibility with macOS Bash 3.2. No formatter or linter is configured, so keep imports organized and changes consistent with nearby code.

## Testing Guidelines

Tests use `unittest.TestCase` and are collected by pytest. Name files `test_*.py`, classes `*Test`, and methods `test_<behavior>`. Add deterministic unit coverage for search spaces, checkpoint handling, aggregation, and protocol safeguards. Dataset- or weight-dependent changes should also include a documented smoke command and summarized output; never train on the final test split.

## Commit & Pull Request Guidelines

History is currently minimal; follow its concise, imperative subject style (for example, `Add deterministic dataset split validation`). Keep each commit focused. Pull requests should explain the research motivation, list commands run, identify dataset/model assumptions, and report metric changes when experiment behavior changes. Link relevant issues and include plots or summary tables for result-facing changes.

## Data and Configuration Safety

Do not commit credentials, absolute local paths, datasets, checkpoints, or large run directories. Document required dataset layout in `docs/datasets.md` and prefer reproducible YAML/config examples with portable paths.
