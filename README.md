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

## Evaluation (placeholder)

> Evaluation scripts and methodology will be documented here later.

Planned content for this section:

- How to run evaluation on:
  - WikiSQL test split.
  - (Optional) Spider dev set.
- Metrics:
  - Logical form accuracy (exact SQL match).
  - Execution accuracy (matching query results).
  - Latency benchmarks (p50/p95).
- How to generate evaluation reports under `docs/` or `outputs/`.

---

## External Validation (Spider dev) – planned

After training on `b-mc2/sql-create-context`, we plan to add a secondary
evaluation harness on the **Spider** dev set (e.g., `xlangai/spider`) to
measure generalization to harder, multi-table, cross-domain text-to-SQL tasks.

For the high-level plan, see [`docs/external_validation.md`](./docs/external_validation.md).
_code
-new-</-
---

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
├── app/                    # Streamlit app (to be implemented)
├── docs/                   # Documentation, design notes, evaluation reports
├── notebooks/              # Jupyter/Colab notebooks for experimentation
├── scripts/                # CLI scripts (e.g., dataset loading, training, eval)
│   └── smoke_load_dataset.py
├── src/
│   └── text2sql/           # Core Python package
│       ├── __init__.py
│       └── utils/          # Utility modules (to be implemented)
│           └── __init__.py
├── tests/
│   └── test_repo_smoke.py  # Basic smoke test (imports the package)
├── .env.example            # Example environment file
├── .gitignore
├── context.md              # Persistent project context & decisions
├── LICENSE
├── README.md
└── requirements.txt
```

As the project progresses, this structure will be refined and additional modules, scripts, and documentation will be added.