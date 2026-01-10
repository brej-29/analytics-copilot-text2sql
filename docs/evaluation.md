# Evaluation and External Validation

This document describes the evaluation pipeline for the Analytics Copilot
(Text-to-SQL) project, including:

- Internal evaluation on our processed validation set derived from
  **b-mc2/sql-create-context**.
- Secondary external validation on the **Spider** dev set using
  Hugging Face datasets (`xlangai/spider` and `richardr1126/spider-schema`).

The goals are to provide reproducible, lightweight metrics and human-readable
reports suitable for both local development and portfolio-style summaries.

---

## 1. Core Evaluation Utilities

All shared evaluation logic lives under:

- `src/text2sql/eval/`

Modules:

- `normalize.py`
  - `normalize_sql(sql: str) -> str`
    - Strip leading/trailing whitespace.
    - Remove trailing semicolons.
    - Collapse internal whitespace (spaces, tabs, newlines) into a single space.
  - `normalize_sql_no_values(sql: str) -> str`
    - Apply `normalize_sql`.
    - Replace string literals with a placeholder (`'__str__'` / `"__str__"`).
    - Replace integer and floating-point numeric literals with `__num__`.

- `schema.py`
  - `parse_create_table_context(context: str) -> {tables: set[str], columns_by_table: dict[str, set[str]]}`
    - Best-effort parsing of `CREATE TABLE` DDL using `sqlglot`.
  - `referenced_identifiers(sql: str) -> {tables: set[str], columns: set[str]}`
    - Extract referenced table and column names from a SQL query.
  - `schema_adherence(sql: str, context: str) -> bool`
    - True if all referenced tables and columns appear in the provided schema.
    - Returns False if either the query or the schema cannot be parsed.
  - `is_parsable_sql(sql: str) -> bool`
    - Helper used to compute SQL parse success rate.

- `metrics.py`
  - `exact_match(pred, gold) -> bool`
    - Normalized exact string match.
  - `aggregate_metrics(preds, golds, contexts=None, flags=None) -> dict`
    - Aggregates:
      - `num_examples`
      - `exact_match`
      - `no_values_exact_match`
      - `parse_success_rate`
      - `schema_adherence_rate` (when requested via flags)

These utilities are lightweight and are used by both evaluation scripts and
unit tests (no network or GPU required for the tests).

---

## 2. Inference Wrapper

The inference entry point lives in:

- `src/text2sql/infer.py`

Functions:

- `load_model_for_inference(base_model: str, adapter_dir: str | None, device: str = "auto")`
  - Loads a base model (e.g., `mistralai/Mistral-7B-Instruct-v0.1`) using
    `transformers.AutoModelForCausalLM`.
  - If `adapter_dir` is provided, loads LoRA adapters (QLoRA) via `peft.PeftModel`.
  - Uses 4-bit quantization (`BitsAndBytesConfig`) on CUDA when available.
  - Supports devices:
    - `auto` (prefer GPU if available).
    - `cuda` (with CPU fallback and a warning if CUDA is unavailable).
    - `cpu`.

- `generate_sql(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str`
  - Runs `.generate(...)` on the loaded model and decodes only the continuation
    (everything after the prompt).
  - Uses the same normalization (`ensure_sql_only`) as in training to strip
    code fences and normalize whitespace.
  - Returns a single SQL string.

This wrapper is used by both internal and Spider evaluation scripts. In `--mock`
mode the evaluation scripts do **not** import or load the model, keeping local
tests lightweight.

---

## 3. Internal Evaluation (b-mc2/sql-create-context Val)

### 3.1 Script: `scripts/evaluate_internal.py`

CLI arguments:

- `--val_path` (default: `data/processed/val.jsonl`)
  - Alpaca-style validation file produced by `scripts/build_dataset.py`.
- `--base_model` (default: `mistralai/Mistral-7B-Instruct-v0.1`)
- `--adapter_dir`
  - Directory with LoRA adapters from QLoRA training.
  - Required unless `--mock` is used.
- `--device` (`auto` / `cuda` / `cpu`, default: `auto`)
- `--max_examples` (default: `200`)
  - Upper bound on the number of validation examples evaluated.
- `--out_dir` (default: `reports/`)
- Generation parameters:
  - `--max_new_tokens` (default: `256`)
  - `--temperature` (default: `0.0` – greedy)
  - `--top_p` (default: `0.9`)
- `--mock`
  - When set, **no model is loaded**; the script uses the gold SQL as the
    prediction for each example to validate the metric pipeline and report
    generation.

### 3.2 Metrics

For the internal validation set we report:

- **Exact Match (normalized)**:
  - `normalize_sql(pred) == normalize_sql(gold)`
  - Normalization strips whitespace, removes trailing semicolons, and collapses
    internal whitespace.

- **No-values Exact Match**:
  - `normalize_sql_no_values(pred) == normalize_sql_no_values(gold)`
  - Ignores differences in literal values (numbers and quoted strings).

- **SQL Parse Success Rate**:
  - Fraction of predictions for which `sqlglot` successfully parses the SQL.

- **Schema Adherence Rate**:
  - Fraction of predictions for which:
    - All referenced table names appear in the `CREATE TABLE` schema.
    - All referenced column names appear among the schema's columns.
  - The schema is extracted from the Alpaca-style `input` field:
    - `### Schema: ...`
    - `### Question: ...`

### 3.3 Outputs

The script writes:

- `reports/eval_internal.json`
  - Contains:
    - `config`: CLI arguments and resolved values.
    - `metrics`: aggregated metrics as described above.
    - `examples`: up to 10 example rows (question, schema snippet, gold/pred).

- `reports/eval_internal.md`
  - Human-readable report with:
    - Configuration summary.
    - Metrics section.
    - Example predictions, each showing:
      - Question.
      - Schema snippet.
      - Gold SQL.
      - Predicted SQL.
      - Exact-match indicator.

### 3.4 How to Run

Mock run (no model, no GPU; suitable for quick checks and CI):

```bash
python scripts/evaluate_internal.py \
  --val_path tests/fixtures/eval_internal_sample.jsonl \
  --mock \
  --out_dir reports
```

Real evaluation with adapters (requires GPU and internet to download the base model):

```bash
python scripts/evaluate_internal.py \
  --val_path data/processed/val.jsonl \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --adapter_dir /path/to/qlora/adapters \
  --device cuda \
  --max_examples 200 \
  --out_dir reports \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 0.9
```

---

## 4. External Validation on Spider Dev

### 4.1 Datasets and License

External validation uses two Hugging Face datasets:

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
    - `Schema (values (type))` – compact, pipe-delimited schema description.

Licensing:

- Spider is licensed under **CC BY-SA 4.0**.
- We use Spider **only for evaluation**, not for training.

### 4.2 Script: `scripts/evaluate_spider_external.py`

CLI arguments:

- `--base_model` (default: `mistralai/Mistral-7B-Instruct-v0.1`)
- `--adapter_dir`
  - Directory with LoRA adapters from QLoRA training.
  - Required unless `--mock` is used.
- `--device` (`auto` / `cuda` / `cpu`, default: `auto`)
- `--spider_split` (default: `"validation"`)
- `--spider_source` (default: `"xlangai/spider"`)
- `--schema_source` (default: `"richardr1126/spider-schema"`)
- `--max_examples` (default: `200`)
- `--out_dir` (default: `reports/`)
- Generation parameters:
  - `--max_new_tokens` (default: `256`)
  - `--temperature` (default: `0.0`)
  - `--top_p` (default: `0.9`)
- `--mock`
  - When set, uses **only local fixtures** under `tests/fixtures/`:
    - `spider_sample.jsonl`
    - `spider_schema_sample.jsonl`
  - Gold SQL is used as the prediction for each example to validate the
    evaluation flow and prompt construction.
  - No internet access is required in this mode.

### 4.3 Schema Handling

The Spider schema dataset provides a compact schema description:

```text
|phone_market|phone : Name (text) , Phone_ID (number) , ... | market : ... | ...
```

The script:

1. Parses the `Schema (values (type))` string into table-level structures.
2. Converts these into simplified `CREATE TABLE` statements with generic
   types (`TEXT` / `NUMERIC`), e.g.:

   ```sql
   CREATE TABLE phone (Name TEXT, Phone_ID NUMERIC, ...);
   CREATE TABLE market (...);
   ```

3. Uses this DDL as the schema context when building prompts.

If parsing fails, the raw schema string is used as a fallback context.

### 4.4 Prompt Construction

For each Spider example:

1. Look up the schema for the example’s `db_id`.
2. Build an input text using the same format as internal training:

   ```text
   ### Schema:
   <CREATE TABLE ...>

   ### Question:
   <Spider question>
   ```

3. Wrap it with the same instruction template used for training:

   ```text
   ### Instruction:
   Write a SQL query that answers the user's question using ONLY the tables and columns provided in the schema.

   ### Input:
   <schema + question as above>

   ### Response:
   ```

This ensures consistent prompting between internal and external evaluations.

### 4.5 Metrics

For Spider evaluation we report:

- **Exact Match (normalized)**:
  - Same definition as internal.

- **No-values Exact Match**:
  - Same definition as internal; ignores literal differences.

- **SQL Parse Success Rate**:
  - Fraction of predictions that can be parsed by `sqlglot`.

We deliberately **do not** attempt to reproduce the full official Spider
evaluation (component matching, execution accuracy) to keep the pipeline
lightweight and easy to run in varied environments.

### 4.6 Outputs

The script writes:

- `reports/eval_spider.json`
  - Contains:
    - `config`: CLI arguments and resolved values.
    - `metrics`: aggregated metrics.
    - `examples`: up to 10 examples, each including:
      - `db_id`
      - `question`
      - `schema` (DDL)
      - `prompt`
      - `gold_sql`
      - `pred_sql`

- `reports/eval_spider.md`
  - Narrative report with:
    - Configuration summary.
    - Metrics section.
    - Example predictions with schema snippets, gold, and predicted SQL.
    - Note about the difference between this lightweight evaluation and the
      official Spider metrics.

### 4.7 How to Run

Mock run (offline, using local fixtures only):

```bash
python scripts/evaluate_spider_external.py \
  --mock \
  --max_examples 4 \
  --out_dir reports
```

Real evaluation with adapters (requires internet and GPU for practical speed):

```bash
python scripts/evaluate_spider_external.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --adapter_dir /path/to/qlora/adapters \
  --device cuda \
  --spider_source xlangai/spider \
  --schema_source richardr1126/spider-schema \
  --spider_split validation \
  --max_examples 200 \
  --out_dir reports \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 0.9
```

---

## 5. Local Testing and CI Considerations

To keep tests offline and lightweight:

- All **unit tests** operate on small local fixtures and synthetic examples.
- No tests require internet or GPU access.
- The evaluation scripts can be exercised in `--mock` mode, which:
  - Skips model loading.
  - Uses gold SQL as predictions.
  - Uses local fixtures for Spider instead of downloading from Hugging Face.

Key tests:

- `tests/test_normalize_sql.py`
  - Verifies SQL normalization and no-values behavior.

- `tests/test_schema_adherence.py`
  - Checks schema parsing and adherence logic on simple synthetic schemas.

- `tests/test_metrics_aggregate.py`
  - Ensures aggregated metrics (EM, no-values EM, parse success, schema adherence)
    behave as expected.

- `tests/test_prompt_building_spider.py`
  - Runs `scripts/evaluate_spider_external.py --mock` against fixtures and checks
    that prompts include the expected schema/question structure and are written
    to the JSON report.

---

## 6. Summary

The evaluation pipeline now provides:

- A reproducible **internal validation** harness aligned with the training
  data and prompt format.
- A lightweight **secondary external validation** on Spider dev to measure
  cross-domain generalization.
- Shared normalization, schema, and metric utilities under `src/text2sql/eval/`.
- Mock modes and small fixtures that keep local tests fast and fully offline.

These components together support both rigorous quantitative evaluation and
readable reports suitable for documentation, demos, and portfolio use.