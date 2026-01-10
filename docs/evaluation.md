# Evaluation for Analytics Copilot (Text-to-SQL)

This document describes the evaluation pipeline for the Analytics Copilot
(Text-to-SQL) model, including:

- **Internal validation** on the held-out split of
  `b-mc2/sql-create-context`.
- **External validation** on the **Spider dev** split using
  `xlangai/spider` + `richardr1126/spider-schema`.

Both evaluation scripts are designed to be:

- **Reproducible** – CLI-driven, with JSON + Markdown artifacts.
- **Offline-friendly** – `--mock` mode uses small local fixtures instead of
  real models / remote datasets.
- **Consistent** – Reuse the same instruction and prompt format as used
  during training.

---

## 1. Metrics

Evaluation uses string- and parser-based metrics only. No database engines are
invoked, so metrics are easy to compute in any environment.

### 1.1 Normalization

We define two normalized views of SQL:

- `normalize_sql(sql)`:
  - Strips leading/trailing whitespace.
  - Collapses runs of whitespace into a single space.
  - Removes trailing semicolons.

- `normalize_sql_no_values(sql)`:
  - Applies `normalize_sql`.
  - Replaces string and numeric literals with a generic `<value>` placeholder.

These helpers live in:

- `src/text2sql/eval/normalize.py`

### 1.2 Core metrics

Implemented in:

- `src/text2sql/eval/metrics.py`

Metrics:

- **Exact Match (EM)** – `exact_match(pred, gold)`:
  - Compares `normalize_sql(pred)` vs `normalize_sql(gold)`.

- **No-values EM** – `normalize_sql_no_values(pred) == normalize_sql_no_values(gold)`:
  - Ignores differences in literal values (numbers, strings).

- **SQL parse success rate**:
  - Fraction of predictions that `sqlglot.parse_one` can parse without errors.

- **Schema adherence rate** (internal only):
  - Fraction of predictions that reference only tables and columns present in
    the provided schema context (see below).

---

## 2. Schema Parsing and Adherence (Internal)

To measure whether generated SQL respects the provided schema, we implement a
lightweight schema parser and adherence checker in:

- `src/text2sql/eval/schema.py`

### 2.1 Parsing CREATE TABLE context

Internal examples (from `b-mc2/sql-create-context`) include a `context` field
with one or more `CREATE TABLE` statements.

`parse_create_table_context(context: str)` returns:

```python
{
    "tables": {"head", "department", ...},
    "columns_by_table": {
        "head": {"age", "name", "born_state", ...},
        "department": {"department_id", "name", ...},
        ...
    },
}
```

All identifiers are lowercased for robust matching.

### 2.2 Extracting referenced identifiers

`referenced_identifiers(sql: str)` performs a best-effort analysis using
`sqlglot`:

- Finds all `Table` nodes (table names).
- Finds all `Column` nodes (column names).

It returns:

```python
{
    "tables": {"head", "department"},
    "columns": {"age", "name", "department_id"},
}
```

This is mainly useful for debugging and exploratory analysis.

### 2.3 Schema adherence

`schema_adherence(sql: str, context: str) -> bool` checks:

- Every referenced table appears in `tables`.
- Every referenced column appears in `columns_by_table` for some table.

Details:

- If the SQL cannot be parsed, adherence is **False**.
- If the context cannot be parsed and yields an empty schema, adherence is
  treated as **True** (we have no schema to check against).
- For qualified columns (`t.age`):
  - Resolve `t` via table aliases when possible.
  - Check that `age` is in the column set for the resolved table.
- For unqualified columns (`age`):
  - Check that `age` appears in *some* table’s column set.

This is intentionally conservative and is meant as a **sanity check**, not a
formal database-level verification.

---

## 3. Internal Evaluation Script

**File:** `scripts/evaluate_internal.py`

This script evaluates the model on the processed validation split produced by
`scripts/build_dataset.py`:

- `data/processed/val.jsonl`

Each record is an Alpaca-style example:

- `id`, `instruction`, `input`, `output`, `source`, `meta`

The `input` field encodes schema + question as:

```text
### Schema:
<CREATE TABLE ...>

### Question:
<question text>
```

### 3.1 CLI usage

Mock mode (no model required, uses local fixture if `val.jsonl` is missing):

```bash
python scripts/evaluate_internal.py --mock
```

Typical model-backed run (GPU recommended):

```bash
python scripts/evaluate_internal.py \
  --val_path data/processed/val.jsonl \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --adapter_dir outputs/adapters \
  --device auto \
  --max_examples 200 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 0.95
```

Key arguments:

- `--val_path`:
  - Path to the processed validation JSONL.
- `--base_model`:
  - Base model identifier (or local path).
- `--adapter_dir`:
  - Path to LoRA adapters (required unless `--mock`).
- `--device`:
  - `auto`, `cpu`, or `cuda`.
- `--max_examples`:
  - Cap on how many validation records to evaluate.
- `--max_new_tokens`, `--temperature`, `--top_p`:
  - Generation hyperparameters.
- `--mock`:
  - Run without loading any model; predictions echo the gold SQL.

### 3.2 Outputs

The script writes two artifacts under the `--out_dir` directory (default:
`reports/`):

1. **JSON summary**

   - `reports/eval_internal.json`

   Structure (simplified):

   ```json
   {
     "metrics": {
       "num_examples": 200,
       "em": 0.87,
       "no_values_em": 0.93,
       "parse_success_rate": 0.98,
       "schema_adherence_rate": 0.95
     },
     "config": {
       "val_path": "data/processed/val.jsonl",
       "base_model": "mistralai/Mistral-7B-Instruct-v0.1",
       "adapter_dir": "outputs/adapters",
       "device": "cuda",
       "max_examples": 200,
       "max_new_tokens": 256,
       "temperature": 0.0,
       "top_p": 0.95,
       "mock": false
     }
   }
   ```

2. **Markdown report**

   - `reports/eval_internal.md`

   Contains:

   - A summary of metrics.
   - 10 example rows with:
     - Question.
     - Schema snippet.
     - Gold SQL.
     - Predicted SQL.
     - Per-example EM / no-values EM / parse success / schema adherence.

---

## 4. External Evaluation on Spider dev

External evaluation uses:

- `xlangai/spider` (Spider text-to-SQL data).
- `richardr1126/spider-schema` (schema summaries per `db_id`).

Spider is licensed under **CC BY-SA 4.0** and is used here **only for
evaluation**, not for training.

### 4.1 Schema serialization for Spider

The `richardr1126/spider-schema` dataset provides, for each database (`db_id`),
a compact textual description of the schema. A typical row contains:

- `db_id`
- A schema text field (column auto-detected at runtime), e.g.:

  ```text
  perpetrator : Perpetrator_ID (number) , People_ID (number) , ... |
  people : People_ID (number) , Name (text) , ...
  ```

We convert this into pseudo-DDL:

```sql
CREATE TABLE perpetrator (
  Perpetrator_ID number,
  People_ID number,
  ...
);
CREATE TABLE people (
  People_ID number,
  Name text,
  ...
);
```

This is done by `_schema_text_to_create_table` in
`scripts/evaluate_spider_external.py`. The exact types are not critical; the
model primarily needs table and column names.

### 4.2 Prompt construction for Spider

For each Spider example:

- Fields (from `xlangai/spider`):
  - `db_id`
  - `question`
  - `query` (gold SQL)

- We look up the schema string for `db_id` from `richardr1126/spider-schema`.
- Convert to pseudo-DDL as above.
- Build `input` using the same helper as internal training:

  ```python
  input_text = build_input_text(context=schema_ddl, question=question)
  ```

- Build the final prompt:

  ```python
  prompt = build_prompt(
      instruction=INSTRUCTION_TEXT,
      input=input_text,
  )
  ```

This ensures internal and external evaluation use the **same instruction and
prompt template**.

---

## 5. Spider Evaluation Script

**File:** `scripts/evaluate_spider_external.py`

### 5.1 CLI usage

Mock mode (no model, fixtures only):

```bash
python scripts/evaluate_spider_external.py --mock
```

Real evaluation (GPU recommended):

```bash
python scripts/evaluate_spider_external.py \
  --spider_source xlangai/spider \
  --spider_subset spider \
  --spider_split validation \
  --schema_source richardr1126/spider-schema \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --adapter_dir outputs/adapters \
  --device auto \
  --max_examples 200 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 0.95
```

Key arguments:

- `--spider_source`:
  - HF dataset id for Spider (default: `xlangai/spider`).
- `--spider_subset`:
  - HF subset/config (default: `spider`).
- `--spider_split`:
  - Split to evaluate (default: `validation`).
- `--schema_source`:
  - HF dataset id for Spider schemas (default: `richardr1126/spider-schema`).
- `--base_model`, `--adapter_dir`, `--device`, `--max_examples`,
  `--max_new_tokens`, `--temperature`, `--top_p`, `--mock`:
  - As in internal evaluation.

In `--mock` mode the script does **not** touch Hugging Face Hub. Instead it
uses small local fixtures:

- `tests/fixtures/spider_sample.jsonl`
- `tests/fixtures/spider_schema_sample.jsonl`

### 5.2 Outputs

Artifacts under `--out_dir` (default: `reports/`):

1. **JSON summary**

   - `reports/eval_spider.json`

   Structure (simplified):

   ```json
   {
     "metrics": {
       "num_examples": 200,
       "em": 0.21,
       "no_values_em": 0.34,
       "parse_success_rate": 0.92
     },
     "config": {
       "spider_source": "xlangai/spider",
       "spider_subset": "spider",
       "spider_split": "validation",
       "schema_source": "richardr1126/spider-schema",
       "base_model": "mistralai/Mistral-7B-Instruct-v0.1",
       "adapter_dir": "outputs/adapters",
       "device": "cuda",
       "max_examples": 200,
       "mock": false
     }
   }
   ```

2. **Markdown report**

   - `reports/eval_spider.md`

   Contains:

   - Summary metrics.
   - 10 example rows with:
     - `db_id`
     - Question.
     - Schema snippet (pseudo-DDL).
     - Gold SQL.
     - Predicted SQL.
     - Per-example EM / no-values EM / parse success.

---

## 6. Inference Helper

Both evaluation scripts rely on a shared inference helper module:

- `src/text2sql/infer.py`

It exposes two functions:

- `load_model_for_inference(base_model: str, adapter_dir: Optional[str], device: str)`

  - Loads a `transformers` Causal LM and tokenizer.
  - Optionally attaches LoRA adapters from `adapter_dir`.
  - Uses 4-bit loading on CUDA when `bitsandbytes` is available; otherwise
    falls back to float16 on GPU or float32 on CPU.
  - Returns `(model, tokenizer, resolved_device)` and also caches them in
    module-level globals.

- `generate_sql(prompt: str, max_new_tokens: int, temperature: float, top_p: float)`

  - Uses the globally cached model/tokenizer to generate text from the prompt.
  - Applies `ensure_sql_only` to strip markdown fences and isolate the SQL.

This separation keeps the evaluation scripts small and makes it easy to reuse
inference logic in other tools (e.g., a notebook or a UI backend).

---

## 7. Notes and Limitations

- Metrics are **string-based**; they do *not* execute queries against real
  databases.
- Spider’s official leaderboard uses more sophisticated evaluation (including
  execution accuracy); numbers from this pipeline are therefore **not directly
  comparable** to leaderboard scores.
- Schema adherence is currently implemented only for internal evaluation on
  `b-mc2/sql-create-context`.
- The Spider datasets used here are licensed under **CC BY-SA 4.0** and are
  employed solely for evaluation, not for training.