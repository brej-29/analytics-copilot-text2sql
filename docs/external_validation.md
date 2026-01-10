# External Validation on Spider Dev

This document describes the implemented **secondary external validation**
workflow for the Analytics Copilot (Text-to-SQL) model using the **Spider**
dataset.

External validation is implemented in **Task 4** via dedicated evaluation
scripts and utilities. It provides a lightweight but informative check of
cross-domain generalization beyond our primary training distribution.

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

We use the following Hugging Face datasets:

- **Spider dev**:
  - Dataset: `xlangai/spider`
  - Split: `validation`
  - Fields used:
    - `db_id` – database identifier.
    - `question` – natural language question.
    - `query` – gold SQL query.

- **Spider schema**:
  - Dataset: `richardr1126/spider-schema`
  - Fields used:
    - `db_id`
    - `Schema (values (type))` – compact, pipe-delimited schema representation
      for each database.

Licensing:

- Spider is licensed under **CC BY-SA 4.0**.
- We use Spider **only for evaluation**, **not** for training.
- This evaluation is intended for research/portfolio purposes rather than
  official leaderboard submissions.

---

## 3. Evaluation Pipeline

The external validation pipeline is implemented by:

- `scripts/evaluate_spider_external.py`
- Shared utilities under `src/text2sql/eval/`:
  - SQL normalization, schema parsing, and metric aggregation.
- Inference wrapper:
  - `src/text2sql/infer.py`

High-level flow:

1. **Load Spider dev examples**
   - From `xlangai/spider` (`validation` split by default).
   - Each example provides `(db_id, question, query)`.

2. **Load Spider schemas**
   - From `richardr1126/spider-schema`.
   - For each `db_id`, read the `Schema (values (type))` string.
   - Parse this compact representation into table/column structures.
   - Convert to simplified `CREATE TABLE` DDL:

     ```sql
     CREATE TABLE table1 (col1 TEXT, col2 NUMERIC, ...);
     CREATE TABLE table2 (...);
     ```

   - If parsing fails, fall back to embedding the raw schema text.

3. **Prompt Construction**
   - For each example:
     - Locate the schema for its `db_id`.
     - Build the input text:

       ```text
       ### Schema:
       <CREATE TABLE ...>

       ### Question:
       <Spider question>
       ```

     - Wrap with the same instruction template used for internal training:

       ```text
       ### Instruction:
       Write a SQL query that answers the user's question using ONLY the tables and columns provided in the schema.

       ### Input:
       <schema + question as above>

       ### Response:
       ```

   - This keeps prompting consistent between internal and external evaluations.

4. **Model Inference**
   - Load the fine-tuned Mistral-7B-Instruct model with QLoRA adapters using
     `load_model_for_inference(...)` from `src/text2sql/infer.py`.
   - For each prompt, generate a SQL query with configurable decoding
     parameters (temperature, top-p, max_new_tokens).

5. **Metrics**
   - Compute metrics using `src/text2sql/eval/metrics.py`:
     - **Exact Match (normalized)**:
       - Uses SQL normalization (strip, collapse whitespace, remove trailing
         semicolons).
     - **No-values Exact Match**:
       - Additionally abstracts away literal values (numbers and strings).
     - **SQL Parse Success Rate**:
       - Fraction of predictions that `sqlglot` can parse.

   - We intentionally **do not** implement full official Spider metrics
     (component matching, execution accuracy) to keep the evaluation pipeline
     lightweight and easy to run.

6. **Reporting**
   - Write:
     - `reports/eval_spider.json`
     - `reports/eval_spider.md`
   - Reports include:
     - Configuration summary (datasets, split, model, adapters, decoding
       settings).
     - Aggregated metrics.
     - Example predictions with:
       - `db_id`
       - Question
       - Schema snippet
       - Gold SQL
       - Predicted SQL

---

## 4. Mock Mode and Offline Testing

To support offline development and CI, the Spider evaluation script provides a
`--mock` mode:

- Uses small local fixtures under `tests/fixtures/`:
  - `spider_sample.jsonl`
  - `spider_schema_sample.jsonl`
- Uses **gold SQL as predictions** to validate:
  - Prompt construction.
  - Schema parsing/serialization.
  - Metric aggregation.
  - Report generation.
- Does **not** import or call Hugging Face `load_dataset`, and does not require
  internet access or a GPU.

This mock mode is exercised by `tests/test_prompt_building_spider.py`.

---

## 5. Relation to Core Training

The Spider external validation is explicitly **secondary**:

- Primary training and validation are performed on:
  - `b-mc2/sql-create-context` (train/val splits).
- Spider is used exclusively to:
  - Measure **out-of-domain** and **cross-domain** performance.
  - Highlight gaps between the training distribution and broader text-to-SQL
    tasks.

This separation keeps the primary training pipeline simple and focused, while
still giving us a rigorous external check on generalization capabilities.

For implementation details of the evaluation scripts and metrics, see
[`docs/evaluation.md`](./evaluation.md).