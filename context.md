# Analytics Copilot (Text-to-SQL) – Mistral-7B QLoRA

## 1) Project Summary (what we are building)

We are building an **Analytics Copilot** that allows business and data users to query structured data (e.g., data warehouse tables) using **natural language**, which is translated into **SQL** and executed against a database. The core of the system will be a **Mistral-7B-based model fine-tuned with QLoRA** for text-to-SQL generation, combined with a lightweight retrieval and schema-understanding layer, plus a Streamlit UI for interactive exploration.

Key capabilities (target state):
- Accept natural-language questions about tabular data.
- Generate syntactically valid and semantically correct SQL against a known schema.
- Execute queries safely (read-only, resource-limited) and visualize results.
- Provide explanations of the generated SQL for transparency and debugging.
- Support iterative refinement: user can edit SQL or ask follow-up questions.

This repo will contain:
- Data loading and preprocessing pipelines.
- Training and evaluation scripts/notebooks for text-to-SQL with QLoRA on Mistral-7B.
- A simple evaluation harness and metrics reporting.
- A Streamlit-based demo app showcasing the Analytics Copilot.

---

## 2) Final Deliverables (HF model, Streamlit app, repo, metrics)

**Model & Artifacts**
- A trained text-to-SQL model based on **Mistral-7B** fine-tuned via **QLoRA** on the **b-mc2/sql-create-context** dataset (with optional evaluation on WikiSQL and Spider).
- Model uploaded to **Hugging Face Hub** (public or private), including:
  - Model weights and adapter (QLoRA) weights.
  - Model card documenting training data, evaluation metrics, and usage instructions.

**Application**
- **Streamlit UI** for the Analytics Copilot that:
  - Lets users configure DB connection or select a demo schema.
  - Accepts natural-language questions and displays:
    - Generated SQL.
    - Query results (table, possibly charts).
    - Optional explanation/rationale.

**Repository**
- Production-grade, well-structured repo with:
  - Reproducible environment (requirements.txt / Docker later).
  - Scripts for dataset download, preprocessing, training, evaluation, and inference.
  - Tests (unit + smoke) and basic CI hooks (later).
  - Documentation in `README.md` and `docs/`.

**Metrics & Reports**
- Evaluation report(s) including:
  - Text-to-SQL accuracy metrics on WikiSQL (and optionally Spider dev).
  - Latency measurements for inference (end-to-end from NL query to DB result).
  - Resource + training-time summary (GPU hours, batch size, etc.).

---

## 3) Success Metrics (latency target, training time target, quality metrics)

**Latency Targets (inference)**
- **Cold-start latency** (first query after model load):
  - Target: &lt; 8 seconds on a single GPU with 7B model + QLoRA adapter.
- **Steady-state latency** (subsequent queries):
  - Target: **p50 &lt; 1.5s**, **p95 &lt; 3s** per query (text → SQL only, excluding DB execution).

**Training Efficiency**
- **Training time target**:
  - Full QLoRA fine-tuning on WikiSQL should complete in **≤ 8 GPU-hours** on a single modern GPU (e.g., A10/A100/L4 class) with mixed-precision and reasonable hyperparameters.
- Clear documentation of:
  - Hardware used.
  - Epochs, batch size, LR schedule, and total tokens seen.

**Quality Metrics**
- On **WikiSQL** test split:
  - **Logical form accuracy** (exact match of SQL) ≥ 75%.
  - **Execution accuracy** (correct result when executed) ≥ 85%.
- On **Spider dev** (optional stretch goal):
  - Report standard text-to-SQL metrics (exact set TBC later).
- Qualitative success:
  - Generated SQL is usually **readable**, follows schema, and fails safely when unsure.

---

## 4) Dataset Plan

**Training Dataset**
- **b-mc2/sql-create-context**:
  - Source: Hugging Face Datasets → `"b-mc2/sql-create-context"`.
  - Description: Natural-language questions paired with the corresponding `CREATE TABLE` DDL context and gold SQL query answers, making it well-suited for text-to-SQL.
  - Usage in this project:
    - Primary training dataset for the text-to-SQL model.
    - May apply light preprocessing:
      - Normalize or canonicalize SQL.
      - Standardize how `CREATE TABLE` context and schema information are injected into prompts.
      - Filter out pathological or broken examplesary.

**Evaluation Dataset (optional later)**
- **Spider dev**:
  - Source: Standard Spider dataset (Hugging Face or official distribution).
  - Description: Complex, multi-table text-to-SQL benchmark.
  - Planned usage:
    - Optional out-of-domain evaluation to see how well the model generalizes beyond WikiSQL.
    - Might require a separate evaluation harness and schema-serialization strategy.

**General Dataset Strategy**
- Keep dataset handling **reproducible**:
  - Versioned dataset scripts.
  - Clear documentation of any filters and preprocessing.
- Use Hugging Face Datasets where possible for:
  - Easy download/caching.
  - Integration with training pipelines (map/filter/shuffle, streaming if needed).

---

## 5) Decisions Log (dated bullet points)

- **2026-01-10** – Chose **WikiSQL (Salesforce/wikisql)** on Hugging Face as the primary training dataset; Spider dev considered as optional evaluation dataset.
- **2026-01-10** – Adopted **Mistral-7B + QLoRA** as the base modeling approach for the Analytics Copilot (Text-to-SQL).
- **2026-01-10** – Selected a **`src/`-based layout** (`src/text2sql`) and Python tooling centered on `requirements.txt` (instead of pyproject.toml) for simpler initial setup.
- **2026-01-10** – Decided to build a **Streamlit** app as the primary UI for the Analytics Copilot demo.
- **2026-01-10** – Introduced a **dataset smoke test script** (`scripts/smoke_load_dataset.py`) to verify access to WikiSQL via Hugging Face Datasets early in the project.
- **2026-01-10** – Switched the primary training dataset from `Salesforce/wikisql` (script-based, incompatible with `datasets>=4`) to **`b-mc2/sql-create-context`**, which is backed by parquet data files and provides natural-language questions, `CREATE TABLE` context, and SQL answers ideal for text-to-SQL.
- **2026-01-10** – Chose to rely on the `CREATE TABLE` context in `b-mc2/sql-create-context` as a primary mechanism to reduce schema hallucinations during text-to-SQL generation, by always conditioning the model on the explicit schema.
- **2026-01-10** – Decided to create our own deterministic validation split (default 8% of the data, seed=42) from the single `train` split shipped with `b-mc2/sql-create-context`, to enable reproducible model selection and early-stopping.
- **2026-01-10** – Selected **`mistralai/Mistral-7B-Instruct-v0.1`** as the base model for fine-tuning, using **QLoRA (4-bit) + LoRA adapters** implemented via **Unsloth + bitsandbytes** for efficient training on a single GPU.
- **2026-01-10** – Planned a **secondary external validation** step on **Spider dev** (e.g., `xlangai/spider`) after primary training on `b-mc2/sql-create-context`, to measure cross-domain, multi-table generalization.
- **2026-01-10** – Implemented a dedicated evaluation pipeline (internal + Spider dev) using normalized SQL metrics, schema adherence checks, and lightweight external validation based on `xlangai/spider` and `richardr1126/spider-schema` (Spider used only for evaluation, not training).

---

## 6) Change Log (append-only; every future task must add an entry)

- **2026-01-10** – Initial project scaffolding created:
  - Added repo structure (app/, notebooks/, scripts/, src/text2sql/, tests/, docs/).
  - Added `context.md` to serve as the persistent project context.
  - Added `requirements.txt`, `.gitignore`, `.env.example`, and `README.md` skeleton.
  - Implemented `scripts/smoke_load_dataset.py` for WikiSQL dataset access smoke testing.
  - Added basic pytest smoke test to verify that the `text2sql` package imports successfully.
- **2026-01-10** – Updated dataset plan and smoke loader to use the parquet-backed **`b-mc2/sql-create-context`** dataset (compatible with `datasets>=4`) instead of the script-based `Salesforce/wikisql`, and documented this decision in the project context.
- **2026-01-10** – Added a dataset preprocessing pipeline (`scripts/build_dataset.py`) that converts `b-mc2/sql-create-context` into Alpaca-style instruction-tuning JSONL files under `data/processed/` (train/val splits), along with reusable formatting utilities in `text2sql.data_prep`.
- **2026-01-10** – Added QLoRA training scaffolding: a detailed Colab-friendly notebook (`notebooks/finetune_mistral7b_qlora_text2sql.ipynb`), a reproducible training script (`scripts/train_qlora.py`), training utilities under `src/text2sql/training/`, and documentation for training (`docs/training.md`) plus planned external validation on Spider dev (`docs/external_validation.md`).
- **2026-01-10** – Task 4: Added an evaluation pipeline with internal metrics on `b-mc2/sql-create-context` (Exact Match, No-values EM, SQL parse success, schema adherence) and a secondary external validation harness on Spider dev using `xlangai/spider` and `richardr1126/spider-schema`, along with reports under `reports/` and supporting documentation (`docs/evaluation.md`).