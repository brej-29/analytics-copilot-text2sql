# External Validation on Spider Dev

This document describes the **secondary external validation** workflow
for the Analytics Copilot (Text-to-SQL) model using the **Spider** dataset.

The implementation is delivered as part of **Task 4**, and is intended to
complement (not replace) the primary internal evaluation on
`b-mc2/sql-create-context`.

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

- **Text-to-SQL pairs:** `xlangai/spider` (dev/validation split).
- **Schemas per database:** `richardr1126/spider-schema`.

Spider is licensed under **CC BY-SA 4.0**. In this project, Spider and the
associated schema dataset are used **only for evaluation**, not for training.
This keeps the primary training data (b-mc2/sql-create-context) cleanly
separated from the external benchmark.

---

## 3. Implemented Evaluation Plan (Task 4)

The implemented external validation flow on Spider dev is:

1. **Load Spider dev**
   - Use `datasets.load_dataset("xlangai/spider", "spider", split="validation")`.
   - Fields used:
     - `db_id` – database identifier.
     - `question` – natural language question.
     - `query` – gold SQL.

2. **Load and serialize schemas**
   - Load `richardr1126/spider-schema` via `datasets.load_dataset`.
   - For each `db_id`, obtain a compact schema text field (column detected
     heuristically at runtime), which looks roughly like:

     ```text
     table1 : col1 (type) , col2 (type) | table2 : ...
     ```

   - Convert this into pseudo-DDL:

     ```sql
     CREATE TABLE table1 (
       col1 type,
       col2 type,
       ...
     );
     CREATE TABLE table2 (
       ...
     );
     ```

   - This conversion is implemented as `_schema_text_to_create_table` in
     `scripts/evaluate_spider_external.py`.

3. **Prompt construction**
   - For each Spider example:
     - Serialize the schema for its `db_id` as pseudo `CREATE TABLE` DDL.
     - Use the same `build_input_text` helper as internal training to build:

       ```text
       ### Schema:
       <CREATE TABLE ...>

       ### Question:
       <Spider question>
       ```

     - Wrap this `input` with the same instruction template and
       `build_prompt` function used for training on `b-mc2/sql-create-context`.

4. **Model inference**
   - Load the fine-tuned Mistral-7B model and LoRA adapters via
     `src/text2sql/infer.py`:
     - `load_model_for_inference(base_model, adapter_dir, device)`
     - `generate_sql(prompt, max_new_tokens, temperature, top_p)`
   - Generate SQL for each question; post-process with `ensure_sql_only` to
     extract the SQL span.

5. **Metrics**
   - Report:
     - **Exact Match (EM)** on normalized SQL.
     - **No-values EM** (literals masked before comparison).
     - **SQL parse success rate** using `sqlglot`.
   - Execution-based metrics are intentionally deferred to keep the pipeline
     lightweight and dependency-free on database engines.

6. **Reporting**
   - The script `scripts/evaluate_spider_external.py` writes:
     - `reports/eval_spider.json` – metrics + config.
     - `reports/eval_spider.md` – Markdown summary + 10 example cases.
   - These artifacts are meant to be linked from higher-level reports and
     notebooks.

---

## 4. Implementation Notes (Task 4)

The implementation in Task 4 takes the following concrete form:

- Dedicated evaluation script:
  - `scripts/evaluate_spider_external.py`
- Reused / shared utilities:
  - Prompt construction, instruction text, and input formatting from
    `text2sql.data_prep` and `text2sql.training.formatting`.
  - Generic evaluation helpers (normalization, metrics) from
    `text2sql.eval.*`.
- Tests:
  - Small Spider-like fixtures in `tests/fixtures/` to validate schema
    serialization and prompt construction.
  - Tests do **not** require full Spider or any database engine; they are
    purely offline.

Execution-accuracy metrics can be added later by introducing an execution
backend (e.g., SQLite with Spider databases), but this is intentionally out of
scope for the current implementation.

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