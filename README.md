# Analytics Copilot (Text-to-SQL) – Mistral-7B QLoRA

## Overview

This repository contains the scaffolding for an **Analytics Copilot** that converts natural-language questions into SQL queries over structured data (e.g., warehouse tables). The core goal is to fine-tune a **Mistral-7B** model using **QLoRA** for efficient, high-quality **text-to-SQL** generation, and to expose it via a **Streamlit** UI.

The project is currently in the **initial setup** phase:
- Basic Python project structure (src/ layout).
- Dataset smoke test for **b-mc2/sql-create-context** using Hugging Face Datasets.
- Minimal test suite using `pytest`.
- Persistent project context in `context.md`.

For the evolving high-level plan, decisions, and change history, see:
- [`context.md`](./context.md)

---

## Architecture (placeholder)

> This section will be expanded as the project matures.

Planned high-level components:

- **Modeling**
  - Base model: Mistral-7B
  - Finetuning: QLoRA (parameter-efficient)
  - Libraries: `transformers`, `peft`, `accelerate`, etc.

- **Data & Training**
  - Primary dataset: **b-mc2/sql-create-context** (Hugging Face: `b-mc2/sql-create-context`)
  - Preprocessing: prompt construction, schema serialization, and handling of `CREATE TABLE` context.
  - Training scripts & notebooks in `scripts/` and `notebooks/`.

- **Inference & Serving**
  - Text-to-SQL generation pipeline (prompting, decoding, validation).
  - Safe SQL execution layer (read-only queries, limits).
  - Streamlit UI under `app/` for interactive usage.

- **Evaluation**
  - Accuracy metrics on WikiSQL and optionally Spider dev.
  - Latency measurements and quality reports.

---

## Dataset (b-mc2/sql-create-context) + Smoke Loader

### Primary Training Dataset: b-mc2/sql-create-context

- **Name:** `b-mc2/sql-create-context`
- **Source:** [Hugging Face Datasets](https://huggingface.co/datasets/b-mc2/sql-create-context)
- **Contents:** Natural language questions, the corresponding `CREATE TABLE` DDL context, and gold SQL query answers (well-suited for text-to-SQL with schema awareness).

This repo includes a **smoke script** to verify dataset access locally.

> Note: The **WikiSQL** Hugging Face dataset (`Salesforce/wikisql`) is implemented as a **script-based dataset** (`wikisql.py`). Starting with `datasets>=4`, script-based datasets like this are no longer supported by default, so loading WikiSQL will fail unless you explicitly **pin `datasets<4.0.0`** or use an older version of `datasets`. This project instead uses `b-mc2/sql-create-context`, which is backed by parquet data files and compatible with `datasets 4.x`.

### Prerequisites

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. (Optional but recommended) Configure your Hugging Face credentials if required for any private resources:

- Copy `.env.example` to `.env` and fill in values as needed.
- Or run `huggingface-cli login` if you plan to use a HF Hub token.

### Running the Smoke Loader

To verify that the `b-mc2/sql-create-context` dataset can be loaded:

```bash
python scripts/smoke_load_dataset.py
```

Expected behavior:

- Logs basic information while loading the dataset.
- Prints the sizes of all available splits (e.g., `train`, `validation`, `test` if present).
- Shows one example from the `train` split.
- If an error occurs (e.g., missing `datasets` library, no network, or invalid dataset name), a clear error message will be logged.

---

## Dataset Preparation

The raw `b-mc2/sql-create-context` dataset is converted into Alpaca-style
instruction-tuning JSONL files using the build script:

```bash
# Full preprocessing run (uses the full dataset)
python scripts/build_dataset.py

# Quick dev run on a subset of rows (e.g., 2000 examples)
python scripts/build_dataset.py --max_rows 2000
```

By default, the script writes:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`

The `data/` directory is **not** tracked in version control; it is intended to
be generated locally as needed. See [`docs/dataset.md`](./docs/dataset.md) for
details on the raw dataset, the train/val split strategy, and the output format.

---

## Training (QLoRA)

We provide two main paths for QLoRA fine-tuning:

1. A detailed, Colab-friendly notebook:
   - `notebooks/finetune_mistral7b_qlora_text2sql.ipynb`
2. A reproducible CLI script:
   - `scripts/train_qlora.py`

Basic usage:

```bash
# Dry run: load config + dataset, format a small batch, and exit
python scripts/train_qlora.py --dry_run

# Smoke run: validate dataset + config; model loading is skipped on CPU-only environments
python scripts/train_qlora.py --smoke

# Full training example (requires a GPU with sufficient VRAM)
python scripts/train_qlora.py \
  --train_path data/processed/train.jsonl \
  --val_path data/processed/val.jsonl \
  --output_dir outputs/ \
  --max_steps 500 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --max_seq_length 2048
```

See [`docs/training.md`](./docs/training.md) for details on hyperparameters,
QLoRA/LoRA configuration, and troubleshooting (OOM, sequence length, etc.).

---

## Evaluation

The repository includes a lightweight but robust evaluation pipeline for
text-to-SQL:

- **Internal evaluation** on the processed `b-mc2/sql-create-context` val set.
- **Secondary external validation** on the Spider dev split.

See [`docs/evaluation.md`](./docs/evaluation.md) for full details. Below are the
most common commands.

### Internal Evaluation (b-mc2/sql-create-context val)

Mock mode (no model required, exercises metrics/reporting):

```bash
python scripts/evaluate_internal.py --mock \
  --val_path data/processed/val.jsonl \
  --out_dir reports/
```

With a trained QLoRA adapter (GPU recommended):

```bash
python scripts/evaluate_internal.py \
  --val_path data/processed/val.jsonl \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --adapter_dir /path/to/outputs/adapters \
  --device auto \
  --max_examples 200 \
  --out_dir reports/
```

Outputs:

- `reports/eval_internal.json` – metrics, config, and sample predictions.
- `reports/eval_internal.md` – human-readable summary.

### External Validation (Spider dev)

Mock mode (offline fixtures only, no internet required):

```bash
python scripts/evaluate_spider_external.py --mock \
  --out_dir reports/
```

Full Spider dev evaluation with a trained model (requires internet + HF Datasets):

```bash
python scripts/evaluate_spider_external.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --adapter_dir /path/to/outputs/adapters \
  --device auto \
  --spider_source xlangai/spider \
  --schema_source richardr1126/spider-schema \
  --spider_split validation \
  --max_examples 200 \
  --out_dir reports/
```

Outputs:

- `reports/eval_spider.json` – metrics, config, and sample predictions.
- `reports/eval_spider.md` – human-readable summary, including notes on
  differences from official Spider evaluation.

---

## External Validation (Spider dev)

After training on `b-mc2/sql-create-context`, we run a secondary evaluation
harness on the **Spider** dev set (e.g., `xlangai/spider`) to measure
generalization to harder, multi-table, cross-domain text-to-SQL tasks.

Spider evaluation uses a **lightweight EM-style** metric suite:

- Exact Match and No-values Exact Match on normalized SQL.
- SQL parse success using `sqlglot`.
- Schema adherence checks against serialized schemas from
  `richardr1126/spider-schema` (licensed under **CC BY-SA 4.0**).

Spider and its schema helper are used **only for evaluation**, not for
training.

For details, see [`docs/external_validation.md`](./docs/external_validation.md).

---

## Local Development & QA

For a quick local quality check before pushing changes, you can run:

```bash
# 1) Syntax validation across src/, scripts/, and app/
python scripts/check_syntax.py

# 2) Linting (requires ruff to be installed, e.g. `pip install ruff`)
ruff check .

# 3) Test suite (offline-friendly)
pytest -q
```

These commands are also wired into the CI workflow (`.github/workflows/ci.yml`).

## Demo (placeholder)

> The Streamlit UI will be documented here when implemented.

Planned content for this section:

- How to start the Streamlit app (under `app/`).
- Sample configuration for connecting to a demo database or local SQLite DB.
- Usage examples:
  - Asking natural-language questions.
  - Viewing generated SQL and query results.
  - Editing and re-running SQL.

---

## Repo Structure

Current high-level layout:

```text
.
├── app/                        # Streamlit app (to be implemented)
├── docs/                       # Documentation, design notes, evaluation reports
│   ├── dataset.md
│   ├── training.md
│   ├── evaluation.md
│   └── external_validation.md
├── notebooks/                  # Jupyter/Colab notebooks for experimentation
├── scripts/                    # CLI scripts (dataset, training, evaluation)
│   ├── build_dataset.py
│   ├── smoke_load_dataset.py
│   ├── train_qlora.py
│   ├── evaluate_internal.py
│   └── evaluate_spider_external.py
├── src/
│   └── text2sql/               # Core Python package
│       ├── __init__.py
│       ├── data_prep.py
│       ├── infer.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── formatting.py
│       └── eval/
│           ├── __init__.py
│           ├── normalize.py
│           ├── schema.py
│           ├── metrics.py
│           └── spider.py
├── tests/
│   ├── fixtures/
│   │   ├── sql_create_context_sample.jsonl
│   │   ├── eval_internal_sample.jsonl
│   │   ├── spider_sample.jsonl
│   │   └── spider_schema_sample.jsonl
│   ├── test_repo_smoke.py
│   ├── test_build_dataset_offline.py
│   ├── test_data_prep.py
│   ├── test_prompt_formatting.py
│   ├── test_normalize_sql.py
│   ├── test_schema_adherence.py
│   ├── test_metrics_aggregate.py
│   └── test_prompt_building_spider.py
├── .env.example                # Example environment file
├── .gitignore
├── context.md                  # Persistent project context & decisions
├── LICENSE
├── README.md
└── requirements.txt
```

As the project progresses, this structure will be refined and additional modules,
scripts, and documentation will be added.