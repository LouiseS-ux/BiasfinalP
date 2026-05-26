# Coding Standards

## Project Overview

This project is a five-stage automated pipeline that detects implicit gender bias in LLM outputs using a fine-tuned RoBERTa classifier with SHAP token-level explainability. Results surface in a Streamlit dashboard that lets a user compare bias scores across models side by side.

## Project Structure

```
bias-auditing-tool/
├── data/
│   ├── raw/                  # source datasets (WinoBias, StereoSet)
│   ├── probe_bank.json       # Stage 1 output
│   ├── completions.json      # Stage 2 output
│   ├── predictions.json      # Stage 3 output
│   ├── summary_stats.json    # Stage 3 output (aggregated bias %)
│   └── shap_values.json      # Stage 4 output
├── models/
│   └── roberta_finetuned/    # saved model checkpoint
├── src/
│   ├── stage1_probe_bank.py
│   ├── stage2_query_llm.py
│   ├── stage3_classifier.py
│   ├── stage4_shap.py
│   └── dashboard/
│       ├── app.py             # Streamlit entry point
│       └── components/        # page sections
├── tests/
│   ├── unit/
│   └── integration/
├── notebooks/                 # exploratory analysis only
├── config.yaml                # model names, paths, thresholds
├── pyproject.toml
├── .pre-commit-config.yaml
├── .env.example
└── coding_standards.md
```

Each stage writes its outputs to disk so the pipeline can be paused, inspected, and resumed at any point.

## Environment Setup

Use a virtual environment. Never install into the system Python:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -e ".[dev]"
```

All dependencies including dev tools are declared in `pyproject.toml`. Do not install packages ad hoc without adding them there.

### VS Code Setup

Install the recommended extensions (add these to `.vscode/extensions.json`):

- `ms-python.python` — Python language support
- `ms-python.vscode-pylance` — type checking and IntelliSense
- `charliermarsh.ruff` — linting and formatting, runs visually in the editor

Add this to `.vscode/settings.json` so linting and formatting run on save:

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "charliermarsh.ruff",
  "editor.codeActionsOnSave": {
    "source.fixAll.ruff": "explicit",
    "source.organizeImports.ruff": "explicit"
  },
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

## Linting and Formatting

This project uses **Ruff** for both linting and formatting. Ruff replaces Flake8, isort, and most Pylint rules in a single fast tool.

### Running manually

```bash
ruff check .               # lint
ruff check . --fix         # lint and auto-fix
ruff format .              # format
ruff format . --check      # check formatting without changing files
```

### Configuration (in `pyproject.toml`)

```toml
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "ANN"]
ignore = ["ANN101", "ANN102"]

[tool.ruff.lint.isort]
known-first-party = ["src"]
```

Rules enabled:
- `E` / `F` — standard pycodestyle and pyflakes errors
- `I` — import sorting (replaces isort)
- `UP` — pyupgrade (modernise syntax)
- `B` — bugbear (common bugs and design issues)
- `SIM` — simplify (unnecessary complexity)
- `ANN` — type annotation enforcement

### Pre-commit hooks

Linting and formatting run automatically on every commit. Set this up once after cloning:

```bash
pip install pre-commit
pre-commit install
```

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

If a commit is blocked by a lint error, fix it. Do not skip hooks with `--no-verify` except in a genuine emergency.

## Type Annotations

All functions and methods must be fully type-annotated. Use built-in generics (Python 3.10+):

```python
# Good
def score_response(response: str, prompt: str) -> float:
    ...

def collect_responses(providers: list[str]) -> dict[str, str]:
    ...

# Bad — missing annotations
def score_response(response, prompt):
    ...
```

Use `from __future__ import annotations` at the top of files when forward references are needed, rather than quoting type names.

## Data Models

Use `dataclasses` for simple data containers and `Pydantic` for anything that needs validation or serialisation. The JSON schemas defined in the technical spec are the source of truth — Pydantic models must match them exactly:

```python
from dataclasses import dataclass
from pydantic import BaseModel

# Simple internal data
@dataclass
class CompletionRecord:
    probe_id: str
    gender: str
    model: str
    prompt: str
    completion: str
    timestamp: str
    tokens_used: int

# Validated / serialised data (matches predictions.json schema)
class Prediction(BaseModel):
    probe_id: str
    gender: str
    model: str
    label: str          # "biased" | "not_biased"
    confidence: float
    completion: str
```

Never use bare dicts to pass structured data between modules.

## Stage Responsibilities

### `src/stage1_probe_bank.py`

Builds the gender-paired prompt dataset from WinoBias and StereoSet source files in `data/raw/`. Writes `data/probe_bank.json`. Assigns stable `probe_id` values using the format `wb_001`, `ss_042` etc. Done when the file has at least 200 rows with no duplicate IDs.

### `src/stage2_query_llm.py`

Feeds each probe to each configured LLM and collects completions into `data/completions.json`. Must be resume-safe: on restart, read the existing file and skip probe/model pairs already collected. Uses a configurable delay between API calls (default 0.5s from `config.yaml`) rather than full async fanout, so rate limits are respected. Implements exponential backoff on 429 errors. Logs token usage per call for cost tracking.

### `src/stage3_classifier.py`

Runs in two modes via a `--mode` flag:

- `--mode train` — fine-tunes `roberta-base` on a labelled bias dataset, saves the best checkpoint (by eval F1) to `models/roberta_finetuned/`, logs metrics to console and `metrics.json`. Target F1 is 0.75 or above on the held-out test set.
- `--mode inference` — loads the saved checkpoint, runs over all completions, writes `data/predictions.json` and `data/summary_stats.json`.

Use the HuggingFace Trainer API. Training config lives in `config.yaml` under the `classifier:` key, not hardcoded.

### `src/stage4_shap.py`

Applies SHAP to the fine-tuned classifier to produce token-level attribution scores. Reads `data/completions.json` and the saved model checkpoint. Writes `data/shap_values.json`. Run on predicted-biased completions first. Cap token list length at `max_seq_length` to match classifier truncation. SHAP on transformer models is slow on CPU (2 to 5 seconds per completion) so limit initial runs to high-confidence predictions.

### `src/dashboard/app.py`

Streamlit entry point. Read-only — loads from `data/*.json` only, no database, no writes. Use `st.cache_data` on all file loads so data only reloads when files change on disk. All charts use Plotly, not Matplotlib.

## Configuration

All tuneable values live in `config.yaml`, not in code. This includes model names, file paths, thresholds, batch sizes, and learning rates. Load config at the entry point of each stage and pass values down as arguments:

```python
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

model_name = config["classifier"]["model_name"]
```

Never hardcode a path, model name, or threshold directly in stage code.

## ML Conventions

### Model checkpoints

Save checkpoints to `models/roberta_finetuned/` only. Never commit model weights to Git — add `models/` to `.gitignore`. Document how to download or reproduce the checkpoint in the README.

### Metrics

Log train/eval metrics to both the console and a `metrics.json` file at the end of each training run. Include at minimum: F1, precision, recall, epoch count, and the dataset split sizes.

### Reproducibility

Set random seeds at the top of any training script:

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

### HuggingFace Trainer

Use `load_best_model_at_end=True` and set `metric_for_best_model="eval_f1"` so the saved checkpoint is always the best-performing one, not just the last epoch.

## Imports

Import order is enforced by Ruff. The correct order is:

1. Standard library
2. Third-party packages
3. First-party (`src.*`)
4. Relative (`.`)

Always use absolute imports from `src.*` when importing across modules. Use relative imports only within the same package.

```python
# Good
from src.models import Prediction
from src.utils.retry import with_retry

# Bad
from ...models import Prediction
```

## Environment Variables and Secrets

API keys and secrets are loaded from environment variables only. Use `python-dotenv` in development:

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]   # raises KeyError if missing — intentional
```

Never use `os.environ.get("KEY", "fallback")` for secrets. A missing key should fail loudly, not silently use a fallback. The `.env.example` file documents all required variables. Add `.env` to `.gitignore`.

## Error Handling

Be explicit. Never catch bare `Exception` unless you immediately re-raise or log with full context:

```python
# Good
try:
    response = client.complete(prompt)
except RateLimitError as e:
    logger.warning("Rate limited by %s, retrying: %s", provider, e)
    raise

# Bad
try:
    response = client.complete(prompt)
except Exception:
    pass
```

Provider errors should be caught at the client level and mapped to a consistent internal exception type so stage code does not need to know which provider it is talking to.

## Testing

Every module must have tests. Use `pytest`:

```bash
pytest                          # run all tests
pytest tests/unit/              # unit only
pytest -v -k "test_classifier"  # filtered
```

Test what a function does, not how it does it:

- **Stage scripts** — mock file I/O and API calls; test that outputs match the expected JSON schema
- **Classifier** — test inference on a small fixed input; do not re-run training in tests
- **Dashboard components** — test data loading and transformation logic separately from Streamlit rendering
- **Utils** — edge cases and error paths

Fixtures live in `conftest.py` at the appropriate level. Mock data lives in `tests/fixtures/` as `.json` files matching the real data contracts. Never hardcode large strings inline in tests.

## Logging

Use the standard `logging` module. Configure it once at the entry point of each stage script, never in shared library code:

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Stage 2: querying %s with probe %s", model, probe_id)
```

Never use `print` for anything except throwaway scripts.

## Notebooks

Jupyter notebooks in `notebooks/` are for exploration and visualisation only. No production logic lives in a notebook. If something useful is prototyped in a notebook, move it to `src/` with proper structure and tests before it is used anywhere else.
