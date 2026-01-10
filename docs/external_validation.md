# External Validation on Spider Dev (Planned)

This document outlines the planned **secondary external validation** workflow
for the Analytics Copilot (Text-to-SQL) model using the **Spider** dataset.

The actual implementation will be tackled in a later task (Task 4). This file
serves as scaffolding so that the evaluation plan is embedded in the project
narrative from the beginning.

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

## 2. Planned Dataset Source

We plan to use a Hugging Face-hosted Spider variant, such as:

- `xlangai/spider` (dev split)

This choice keeps the evaluation flow consistent with the rest of the project,
which already relies on Hugging Face Datasets for loading and caching.

---

## 3. High-Level Evaluation Plan

The high-level steps for external validation on Spider dev are:

1. **Load Spider dev**
   - Use `datasets.load_dataset("xlangai/spider", split="validation")` or the
     equivalent dev split.
   - Inspect fields: questions, database ids, schemas, and SQL queries.

2. **Schema Serialization**
   - Build a schema-serialization strategy that converts Spider’s multi-table
     schemas into a textual context suitable for the model.
   - Likely format (TBD):
     ```text
     ### Schema:
     CREATE TABLE table1 (...);
     CREATE TABLE table2 (...);
     ...

     ### Question:
     <Spider question>
     ```
   - This should align with the prompt style used during training on
     `b-mc2/sql-create-context`.

3. **Prompt Construction**
   - Reuse or extend the same formatting utilities used for training:
     - `build_prompt(...)`-style function for instruction + input.
     - Ensure that the model receives a consistent prompt structure across
       training and evaluation.

4. **Model Inference**
   - Load the fine-tuned Mistral-7B-Instruct model with the QLoRA adapters.
   - Run the model on Spider dev questions with the appropriate schema
     context.
   - Decode generated SQL queries.

5. **Metrics**
   - We plan to report (at minimum):
     - **Logical form accuracy** (exact match between generated and gold SQL).
     - **Execution accuracy** (whether executing the generated SQL matches
       the gold answer).
   - Where possible, reuse or adapt existing Spider evaluation scripts to
     ensure comparability with prior work.

6. **Reporting**
   - Summarize:
     - Overall metrics.
     - Per-database performance.
     - Qualitative examples (successes and failures).
   - Integrate key results into `docs/` and the main `README`.

---

## 4. Implementation Notes (for Task 4)

When we implement this evaluation pipeline (Task 4), we expect to:

- Add a dedicated evaluation script under `scripts/`
  (e.g., `scripts/eval_spider.py`).
- Add utility functions under `src/text2sql/` for:
  - Schema serialization specific to Spider.
  - Prompt construction for multi-table schemas.
  - Metric computation (exact match, execution accuracy).
- Add tests that:
  - Use small Spider-like fixtures to validate serialization and metrics.
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