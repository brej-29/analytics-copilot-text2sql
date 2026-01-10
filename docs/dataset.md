# Dataset Preparation for Analytics Copilot (Text-to-SQL)

This document describes how we use the **b-mc2/sql-create-context** dataset for
instruction-tuning the Analytics Copilot (Text-to-SQL) model, and how to run
the preprocessing pipeline.

---

## 1. Dataset Source

- **Name:** `b-mc2/sql-create-context`
- **Source:** Hugging Face Datasets  
  https://huggingface.co/datasets/b-mc2/sql-create-context
- **Access Method:** `datasets.load_dataset("b-mc2/sql-create-context")`

This dataset is backed by **parquet data files**, making it compatible with
`datasets 4.x` (unlike script-based datasets such as `Salesforce/wikisql`).

---

## 2. Raw Fields

Each example contains at least the following fields:

- `question` – natural language question the user asks.
- `context` – schema context, typically one or more `CREATE TABLE` statements.
- `answer` – target SQL query that correctly answers the question given the schema.

Example (conceptual):

```json
{
  "question": "How many heads of the departments are older than 56 ?",
  "context": "CREATE TABLE head (age INTEGER)",
  "answer": "SELECT COUNT(*) FROM head WHERE age > 56"
}
```

The dataset ships with a **single `train` split**; we create our own
deterministic validation split.

---

## 3. Train / Validation Split Strategy

Because the dataset only provides a `train` split, we create a reproducible
train/validation split ourselves.

- Start from the raw `train` split (or a local JSONL file in the same format).
- Use `datasets.Dataset.train_test_split()` with:
  - **test_size = val_ratio** (default: `0.08`, i.e. 8% validation).
  - **seed = 42** for determinism.
- Rename:
  - `split["train"]` → final **train** split.
  - `split["test"]` → final **val** split.

This ensures:

- Deterministic splits with a fixed seed.
- Reproducible experiments across machines and runs.

---

## 4. Output Format (Instruction-Tuning JSONL)

The preprocessing pipeline converts raw records into **Alpaca-style**
instruction-tuning examples, written to:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`

Each line is a standalone JSON object with the following keys:

- `id` – unique string identifier (e.g., `sqlcc-train-000001`).
- `instruction` – static instruction describing the text-to-SQL task.
- `input` – formatted text that includes the schema and the question.
- `output` – normalized SQL query.
- `source` – dataset source, fixed to `"b-mc2/sql-create-context"`.
- `meta` – metadata object with build information.

Example record:

```json
{
  "id": "sqlcc-train-000001",
  "instruction": "Write a SQL query that answers the user's question using ONLY the tables and columns provided in the schema.",
  "input": "### Schema:\nCREATE TABLE head (age INTEGER)\n\n### Question:\nHow many heads of the departments are older than 56 ?",
  "output": "SELECT COUNT(*) FROM head WHERE age > 56",
  "source": "b-mc2/sql-create-context",
  "meta": {
    "original_split": "train",
    "row": 0,
    "split": "train",
    "val_ratio": 0.08,
    "seed": 42,
    "from_local_input": false
  }
}
```

### 4.1 Input Formatting

The `input` field is constructed as:

```text
### Schema:
<CREATE TABLE ...>

### Question:
<question text>
```

This makes it explicit which text is schema and which is the natural-language
question, and is designed to reduce schema hallucinations.

### 4.2 SQL Normalization

We apply a light normalization step to the SQL:

- Strip leading/trailing whitespace.
- Collapse runs of whitespace (spaces, tabs, newlines) into a single space.

This keeps formatting consistent without changing the semantics of the query.

---

## 5. Preprocessing Script

The main entry point for preprocessing is:

**File:** `scripts/build_dataset.py`

### 5.1 Command-Line Interface

Basic usage (full dataset):

```bash
python scripts/build_dataset.py
```

Common options:

- `--out_dir`:
  - Default: `data/processed`
  - Destination directory for `train.jsonl` and `val.jsonl`.

- `--val_ratio`:
  - Default: `0.08`
  - Fraction of examples to use for validation.

- `--seed`:
  - Default: `42`
  - Random seed for deterministic splitting.

- `--max_rows`:
  - Optional integer.
  - If provided, limits the number of rows used from the input dataset.
  - Useful for quick dev runs.

- `--overwrite`:
  - Flag.
  - Overwrite existing output files if they exist.

- `--input_jsonl`:
  - Optional path to a local JSONL file containing raw records with keys:
    - `question`, `context`, `answer`.
  - When provided, the script does **not** download from Hugging Face and uses
    this file instead (useful for offline/dev testing).

Example commands:

```bash
# Full preprocessing with default settings
python scripts/build_dataset.py

# Quick dev run on a subset (e.g., 2000 rows)
python scripts/build_dataset.py --max_rows 2000

# Offline/dev mode using a local fixture
python scripts/build_dataset.py \
  --input_jsonl tests/fixtures/sql_create_context_sample.jsonl \
  --out_dir /tmp/sqlcc_dev \
  --val_ratio 0.4 \
  --overwrite
```

The script logs progress (dataset loading, splitting, writing files) and prints
a final summary with row counts and output paths.

---

## 6. Notes on Version Control

- The `data/` directory is **not** committed to version control.
- All processed files (`data/processed/*.jsonl`) are generated locally and can
  be safely recreated using `scripts/build_dataset.py` at any time.

This keeps the repository lightweight while maintaining full reproducibility of
the training data pipeline.