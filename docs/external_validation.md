# External Validation on Spider Dev

This document describes the **secondary external validation** workflow
for the Analytics Copilot (Text-to-SQL) model using the **Spider** dataset.

The implementation of this evaluation pipeline is completed as part of
**Task 4** and wired into the repository via dedicated scripts and tests.

---

## 1. Why Spider?

Our primary training dataset is **b-mc2/sql-create-context**, which focuses on:

- Single (or simple) schemas provided via `CREATE TABLE` context.
- Natural-language questions.
- Gold SQL answers.

While this is excellent for **schema-aware, single-schema text-to-SQL**, it
does not fully test:

- Complex, multi-table joins.
- Cross-domain generalization across many databases.
- More diverse SQL patterns and schema structures.

The **Spider** dataset is a standard benchmark that addresses these gaps:

- Multi-table schemas across many databases.
- Cross-domain questions and schemas.
- Emphasis on compositional generalization.

Evaluating on Spider helps us understand **how well the model generalizes
beyond the training distribution**.

---

## 2. Dataset Sources and Licensing

We use Hugging Face-hosted Spider variants:

- Examples: `xlangai/spider` (dev/validation split)
- Schema helper: `richardr1126/spider-schema`

This choice keeps the evaluation flow consistent with the rest of the project,
which already relies on Hugging Face Datasets for loading and caching.

> **License:** The `richardr1126/spider-schema` dataset is distributed under
> **CC BY-SA 4.0**. In this project, Spider and its schema helper are used
> **only for evaluation**, not for training.

---

## 3. High-Level Evaluation Plan

The high-level steps for external validation on Spider dev are:

1. **Load Spider dev**
   - Use `datasets.load_dataset("xlangai/spider", split="validation")` (or a
     compatible dev split).
   - Keep only rows with `db_id`, `question`, and `query` populated.

2. **Schema Serialization**
   - Load the schema helper dataset `richardr1126/spider-schema` and build a
     mapping `{db_id -> create_table_context}` using
     `text2sql.eval.spider.build_schema_map`.
   - For each Spider example, retrieve `create_table_context` by `db_id` and
     treat it as a textual schema context.

3. **Prompt Construction**
   - For each example, construct the input section:

     ```text
     ### Schema:
     <create_table_context>

     ### Question:
     <Spider question>
     ```

   - Use the same instruction text as internal training/evaluation:

     > Write a SQL query that answers the user's question using ONLY the tables
     > and columns provided in the schema.

   - Wrap instruction + input into a full prompt using
     `text2sql.eval.spider.build_spider_prompt`, which internally reuses the
     training formatter.

4. **Model Inference**
   - Load the fine-tuned Mistral-7B-Instruct model with QLoRA adapters (or a
     merged model) via `text2sql.infer.load_model_for_inference`.
   - Generate SQL for each prompt using `text2sql.infer.generate_sql`.
   - Post-process generated text into clean SQL with
     `text2sql.training.formatting.ensure_sql_only`.

5. **Metrics**
   - Compute lightweight logical-form metrics using
     `text2sql.eval.metrics.aggregate_metrics`:
     - **Exact Match (normalized SQL)**.
     - **No-values Exact Match** (string and numeric literals replaced).
     - **SQL parse success rate** using `sqlglot`.
     - **Schema adherence** (references confined to the serialized schema).
   - These are intentionally lightweight and do **not** attempt to reproduce
     the full official Spider evaluation protocol.

6. **Reporting**
   - Summarize:
     - Overall metrics.
     - Representative examples (successes and failures).
   - Write machine-readable JSON and human-readable Markdown reports under
     `reports/` (see `docs/evaluation.md` for details).

---

## 4. Implementation Notes (Task 4)

Task 4 implemented this evaluation pipeline with the following components:

- A dedicated evaluation script under `scripts/`:
  - `scripts/evaluate_spider_external.py`
- Utility functions under `src/text2sql/eval/`:
  - `spider.py` for schema mapping and prompt construction.
  - `normalize.py`, `schema.py`, and `metrics.py` shared between internal and
    external evaluation.
- Tests that:
  - Use small Spider-like fixtures in `tests/fixtures/` to validate prompt
    construction and metrics.
  - Do not require access to the full Spider dataset or a database engine.

---

## 5. Relation to Core Training

The Spider external validation is explicitly **secondary**:

- Primary training and validation are performed on:
  - `b-mc2/sql-create-context` (train/val splits).
- Spider is used to:
  - Measure **out-of-domain** and **cross-domain** performance.
  - Highlight gaps between training distribution and broader text-to-SQL
    tasks.

This separation keeps the primary training pipeline simple and focused, while
still giving us a rigorous external check on generalization capabilities.