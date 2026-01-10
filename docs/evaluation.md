# Evaluation for Analytics Copilot (Text-to-SQL)

This document describes the evaluation pipeline for the Analytics Copilot
(Text-to-SQL) model, including:

- **Internal evaluation** on the preprocessed `b-mc2/sql-create-context` val set.
- **Secondary external validation** on the Spider dev set using lightweight,
  portfolio-friendly metrics.

The goal is to provide reproducible, scriptable evaluation that can run both
in local development environments (including mock/offline modes) and in
GPU-backed Colab sessions with trained adapters.

---

## 1. Internal Evaluation (b-mc2/sql-create-context val)

### 1.1 Dataset

Internal evaluation uses the Alpaca-style validation file produced by the
dataset builder:

- `data/processed/val.jsonl`

Each line is a JSON object with at least:

- `instruction`
- `input` – formatted schema + question, e.g.:

  ```text
  ### Schema:
  <CREATE TABLE ...>

  ### Question:
  <natural language question>
  ```

- `output` – normalized gold SQL query.

### 1.2 Metrics

The internal evaluation script computes:

- **Exact Match (EM)** – comparison on *normalized* SQL:
  - Strips leading/trailing whitespace.
  - Removes trailing semicolons.
  - Collapses runs of whitespace into a single space.
  - Implemented via `text2sql.eval.normalize.normalize_sql`.

- **No-values Exact Match**
  - Builds on `normalize_sql` and additionally replaces:
    - Single-quoted string literals with a placeholder (`'__STR__'`).
    - Numeric literals (integers/decimals, optionally negative) with a
      placeholder (`__NUM__`).
  - Useful to detect structural matches even when literal values differ.

- **SQL parse success rate**
  - Fraction of predictions that can be parsed by `sqlglot.parse_one`.
  - Provides a lightweight proxy for syntactic validity of generated SQL.

- **Schema adherence rate**
  - Uses the `CREATE TABLE` context from each example and parses it with
    `sqlglot` to recover:
    - Known tables.
    - Known columns per table.
  - Parses the predicted SQL and extracts referenced table and column names.
  - A prediction is schema-adherent if **all** referenced tables/columns
    appear in the context.
  - Implemented via:
    - `text2sql.eval.schema.parse_create_table_context`
    - `text2sql.eval.schema.referenced_identifiers`
    - `text2sql.eval.schema.schema_adherence`

All metrics are aggregated via:

- `text2sql.eval.metrics.aggregate_metrics`

which returns:

- `n_examples`
- `exact_match` – `{count, rate}`
- `no_values_em` – `{count, rate}`
- `parse_success` – `{count, rate}`
- `schema_adherence` – `{count, rate}`

### 1.3 How to Run Internal Evaluation

#### 1.3.1 Mock Mode (no model required)

Mock mode is designed for quick local checks and CI:

```bash
python scripts/evaluate_internal.py --mock \
  --val_path data/processed/val.jsonl \
  --out_dir reports/
```

Behavior:

- Uses the gold SQL (`output`) as the prediction.
- Exercises normalization, parsing, schema adherence, and reporting code.
- Produces:

  - `reports/eval_internal.json`
  - `reports/eval_internal.md`

#### 1.3.2 Real Evaluation with Adapters (GPU recommended)

After fine-tuning with QLoRA (see `docs/training.md`), you can evaluate the
model using the trained adapters:

```bash
python scripts/evaluate_internal.py \
  --val_path data/processed/val.jsonl \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --adapter_dir /path/to/outputs/adapters \
  --device auto \
  --max_examples 200 \
  --temperature 0.0 \
  --top_p 0.9 \
  --max_new_tokens 256 \
  --out_dir reports/
```

Notes:

- `--device auto` prefers GPU when available and falls back to CPU otherwise
  (with a warning).
- By default, when running on CUDA the inference loader will try to load the
  base model in **4-bit (bitsandbytes)** for faster and more memory-efficient
  evaluation. You can explicitly control this with:
  - `--load_in_4bit` / `--no_load_in_4bit`
  - `--dtype` (default `auto`, which maps to `float16` on CUDA and `float32` on CPU)
- `--max_examples` allows you to subsample the validation set for quick runs.
- If you have a **merged model directory**, you can pass it as `--base_model`
  and omit `--adapter_dir`.

---

## 2. External Validation on Spider Dev

### 2.1 Datasets and Licensing

External validation uses two Hugging Face datasets:

1. **Spider examples**

   - Dataset: `xlangai/spider`
   - Split: `validation` (configured via `--spider_split`)
   - Provides:
     - `db_id`
     - `question`
     - `query` (gold SQL)

2. **Spider schema helper**

   - Dataset: `richardr1126/spider-schema`
   - Provides:
     - `db_id`
     - `create_table_context` – a serialized schema context with `CREATE TABLE`
       information for all tables in the database.

> **License:** `xlangai/spider` is derived from the original Spider benchmark,
> and `richardr1126/spider-schema` is licensed under **CC BY-SA 4.0**. In this
> project, Spider is used **only for evaluation**, **not** for training.

### 2.2 Prompt Construction

For each Spider example:

1. Look up `db_id` in the schema helper dataset to retrieve
   `create_table_context`.
2. Build the schema + question input using the **same** format as internal
   evaluation:

   ```text
   ### Schema:
   <create_table_context>

   ### Question:
   <Spider question>
   ```

3. Use the same instruction text as training:

   > "Write a SQL query that answers the user's question using ONLY the tables
   > and columns provided in the schema."

4. Wrap instruction + input into a full prompt using the training formatter:

   - Implemented in `text2sql.eval.spider.build_spider_prompt`, which internally
     reuses:
     - `text2sql.data_prep.INSTRUCTION_TEXT`
     - `text2sql.data_prep.build_input_text`
     - `text2sql.training.formatting.build_prompt`

### 2.3 Metrics

Spider evaluation uses the **same metric suite** as internal evaluation:

- **Exact Match (normalized SQL)**
- **No-values Exact Match**
- **SQL parse success rate**
- **Schema adherence rate**

This provides a **lightweight generalization check** on Spider dev, but it is
**not a full reproduction** of official Spider evaluation. In particular:

- Official Spider metrics include detailed component matching (SELECT, WHERE,
  GROUP BY, etc.).
- Execution-based evaluation is often used to measure semantic equivalence via
  query results.

Here we focus on structural/logical-form approximations that are easy to run
without database execution, suitable for a portfolio-style baseline.

### 2.4 How to Run Spider Evaluation

#### 2.4.1 Mock Mode (offline, fixtures only)

Mock mode uses small offline fixtures under `tests/fixtures/` and **does not
require internet**:

```bash
python scripts/evaluate_spider_external.py --mock \
  --out_dir reports/
```

Behavior:

- Loads:
  - `tests/fixtures/spider_sample.jsonl`
  - `tests/fixtures/spider_schema_sample.jsonl`
- Uses gold SQL as predictions.
- Produces:

  - `reports/eval_spider.json`
  - `reports/eval_spider.md`

This is ideal for local smoke tests of the Spider pipeline.

#### 2.4.2 Real Evaluation with Adapters (GPU recommended)

With network access and a trained model, you can run full Spider dev evaluation:

```bash
python scripts/evaluate_spider_external.py \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --adapter_dir /path/to/outputs/adapters \
  --device auto \
  --spider_source xlangai/spider \
  --schema_source richardr1126/spider-schema \
  --spider_split validation \
  --max_examples 200 \
  --temperature 0.0 \
  --top_p 0.9 \
  --max_new_tokens 256 \
  --out_dir reports/
```

Notes:

- By default, when running on CUDA the inference loader will try to load the
  base model in **4-bit (bitsandbytes)** for faster and more memory-efficient
  evaluation. You can explicitly control this with:
  - `--load_in_4bit` / `--no_load_in_4bit`
  - `--dtype` (default `auto`, which maps to `float16` on CUDA and `float32` on CPU)
- `--max_examples` allows a lighter-weight subset run (e.g., 50–200 examples).
- When `--mock` is not set, the script downloads datasets via
  `datasets.load_dataset`, so internet access is required.

---

## 3. Inference Wrapper

Both evaluation scripts rely on a shared inference helper:

- `src/text2sql/infer.py`

Key functions:

- `load_model_for_inference(base_model, adapter_dir=None, device='auto', load_in_4bit=None, bnb_compute_dtype='float16', dtype='auto')`
  - Loads a base HF model or local directory.
  - Optionally applies LoRA adapters from `adapter_dir`.
  - Resolves device via:
    - `"auto"` → GPU if available, otherwise CPU (with a warning).
    - `"cuda"` / `"cpu"` for explicit control.
  - When running on CUDA and `load_in_4bit` is not explicitly set, the loader
    defaults to 4-bit (NF4) quantization using bitsandbytes. This significantly
    reduces memory usage and speeds up evaluation on Colab-style GPUs.

- `generate_sql(prompt, max_new_tokens, temperature, top_p) -> str`
  - Uses the loaded model/tokenizer to generate text.
  - Evaluation scripts post-process the raw text via
    `text2sql.training.formatting.ensure_sql_only` before metric computation.

This separation keeps the evaluation scripts thin and allows reuse of the
inference pipeline in other tools (e.g., a Streamlit demo or interactive
notebooks).

---

## 4. Local Testing Strategy (No Internet Required)

To keep the test suite lightweight and offline-friendly:

- Fixtures under `tests/fixtures/` provide small synthetic datasets:
  - `eval_internal_sample.jsonl` – mini val-style examples.
  - `spider_sample.jsonl` and `spider_schema_sample.jsonl` – Spider-like
    examples and schemas.
- Unit tests cover:
  - SQL normalization (`test_normalize_sql.py`).
  - Schema parsing and adherence (`test_schema_adherence.py`).
  - Metric aggregation (`test_metrics_aggregate.py`).
  - Spider prompt construction (`test_prompt_building_spider.py`).

CI or local developers can run:

```bash
pytest -q
```

without requiring internet access or GPU hardware. For full model-based
evaluation, see the commands in sections 1.3.2 and 2.4.2 above.

If you see TensorFlow CUDA warnings in Colab logs (e.g. about missing
`libcudart`), they can generally be ignored for this project. The evaluation
scripts also set `TF_CPP_MIN_LOG_LEVEL=3` to suppress most TensorFlow log
noise; you can optionally uninstall TensorFlow entirely if you are not using
it elsewhere in your notebook.