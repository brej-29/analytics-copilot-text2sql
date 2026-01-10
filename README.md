# Analytics Copilot (Text-to-SQL) – Mistral-7B QLoRA

## Overview

This repository contains the scaffolding for an **Analytics Copilot** that converts natural-language questions into SQL queries over structured data (e.g., warehouse tables). The core goal is to fine-tune a **Mistral-7B** model using **QLoRA** for efficient, high-quality **text-to-SQL** generation, and to expose it via a **Streamlit** UI.

The project is currently in the **initial setup** phase:
- Basic Python project structure (src/ layout).
- Dataset smoke test for **WikiSQL** using Hugging Face Datasets.
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
  - Dataset: WikiSQL (Hugging Face: `Salesforce/wikisql`)
  - Preprocessing: prompt construction, schema serialization.
  - Training scripts & notebooks in `scripts/` and `notebooks/`.

- **Inference & Serving**
  - Text-to-SQL generation pipeline (prompting, decoding, validation).
  - Safe SQL execution layer (read-only queries, limits).
  - Streamlit UI under `app/` for interactive usage.

- **Evaluation**
  - Accuracy metrics on WikiSQL and optionally Spider dev.
  - Latency measurements and quality reports.

---

## Dataset (WikiSQL) + Smoke Loader

### Primary Training Dataset: WikiSQL

- **Name:** `Salesforce/wikisql`
- **Source:** [Hugging Face Datasets](https://huggingface.co/datasets/Salesforce/wikisql)
- **Contents:** Natural language questions, associated SQL queries, and table schemas.

This repo includes a **smoke script** to verify dataset access locally.

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

To verify that the WikiSQL dataset can be loaded:

```bash
python scripts/smoke_load_dataset.py
```

Expected behavior:

- Logs basic information while loading the dataset.
- Prints the sizes of the `train`, `validation`, and `test` splits.
- Shows one example from the training split.
- If an error occurs (e.g., missing `datasets` library or no network), a clear error message will be logged.

---

## Training (placeholder)

> Detailed training instructions will be added once the training pipeline is implemented.

Planned content for this section:

- How to run QLoRA fine-tuning on Mistral-7B using WikiSQL.
- Recommended hyperparameters and hardware setup.
- Checkpointing, resuming, and logging (e.g., Weights & Biases or HF Hub).
- Exporting and pushing the trained model/adapters to Hugging Face Hub.

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