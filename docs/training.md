# QLoRA Training for Analytics Copilot (Text-to-SQL)

This document explains how to fine-tune **Mistral-7B-Instruct-v0.1** on the
processed Text-to-SQL dataset using **QLoRA** (4-bit) with **Unsloth** and
**bitsandbytes**.

We provide:

- A **Colab-friendly notebook**:
  - `notebooks/finetune_mistral7b_qlora_text2sql.ipynb`
- A **reproducible CLI script**:
  - `scripts/train_qlora.py`

Both rely on the preprocessed instruction-tuning data produced by
`scripts/build_dataset.py`.

---

## 1. Prerequisites

### 1.1 Data preparation

Before training, generate the processed JSONL files:

```bash
python scripts/build_dataset.py
```

This creates:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`

Each line is an Alpaca-style record with keys:

- `id`
- `instruction`
- `input`
- `output`
- `source`
- `meta`

For details, see [`docs/dataset.md`](./dataset.md).

### 1.2 Environment & dependencies

The training stack uses:

- `torch` (GPU strongly recommended)
- `transformers`
- `accelerate`
- `peft`
- `trl`
- `unsloth`
- `bitsandbytes`

These are included in `requirements.txt`. Install them with:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Note: QLoRA training is intended for **GPU environments**. A 7B model with
> 4-bit quantization typically requires a GPU with **≥ 16 GB VRAM** (depending
> on sequence length and batch size). The CLI script supports CPU-only
> **`--dry_run`** and **`--smoke`** modes for configuration checks.

---

## 2. Notebook Workflow (Recommended for Exploration)

**File:** `notebooks/finetune_mistral7b_qlora_text2sql.ipynb`

The notebook is organized into the following sections:

1. **Overview**
   - Explains the Text-to-SQL task, QLoRA, and why providing explicit
     `CREATE TABLE` schema context reduces hallucinations.

2. **Environment Setup**
   - Installs `unsloth`, `trl`, `bitsandbytes`, etc. (for Colab).
   - Verifies GPU availability and prints device info.
   - Sets random seeds for reproducibility.

3. **Load Processed Dataset**
   - Reads `data/processed/train.jsonl` and `data/processed/val.jsonl`.
   - Inspects a few records (instruction, input, output).
   - Optionally subsamples for a **fast dev run** (e.g., 512 examples).

4. **Prompt Formatting**
   - Uses the same prompt pattern as the CLI script:
     - `build_prompt(instruction, input)` constructs:
       ```text
       ### Instruction:
       <instruction>

       ### Input:
       <schema + question>

       ### Response:
       ```
     - The final training text is `prompt + sql_output`, where the SQL
       output is cleaned via `ensure_sql_only`.
   - Shows 2–3 formatted examples end-to-end.

5. **Model Loading & QLoRA Setup**
   - Loads `mistralai/Mistral-7B-Instruct-v0.1` using Unsloth’s
     `FastLanguageModel.from_pretrained` in 4-bit mode.
   - Explains key QLoRA hyperparameters:
     - `r` (rank): how many low-rank dimensions to add.
     - `alpha`: scaling factor for the LoRA updates.
     - `dropout`: regularization on the LoRA adapters.
     - `target_modules`: which projection layers receive LoRA adapters.

6. **Training Configuration & SFTTrainer**
   - Uses TRL’s `SFTTrainer` (with `SFTConfig`) to run supervised
     finetuning on the formatted dataset.
   - Key settings:
     - `max_steps`
     - `per_device_train_batch_size`
     - `gradient_accumulation_steps`
     - `learning_rate`
     - `warmup_steps`
     - `max_seq_length`
   - Includes a **fast dev run** mode:
     - Limits to ~512 training examples.
     - Uses `max_steps=20` to quickly validate the setup.

7. **Saving Artifacts**
   - Saves LoRA adapters under `outputs/adapters/`.
   - Saves training configuration and metrics to:
     - `outputs/run_meta.json`
     - `outputs/metrics.json`

8. **Quick Sanity Inference**
   - Loads the trained adapters.
   - Runs a few example schema + question pairs through the model.
   - Prints generated SQL for manual inspection.

9. **Next Steps**
   - Describes planned work:
     - External validation on **Spider dev**.
     - Pushing adapters/model to Hugging Face Hub.
     - Integrating with a Streamlit-based Analytics Copilot UI.

---

## 3. CLI Training Script

**File:** `scripts/train_qlora.py`

The script is designed for reproducible runs on a GPU machine. It uses the same
prompt formatting as the notebook and can be integrated into automation or
scheduled jobs.

### 3.1 Basic usage

Dry run (format a batch and exit, no model required):

```bash
python scripts/train_qlora.py --dry_run
```

Smoke test (validate dataset + config; skip model loading on CPU-only):

```bash
python scripts/train_qlora.py --smoke
```

Full training example (GPU required):

```bash
python scripts/train_qlora.py \
  --train_path data/processed/train.jsonl \
  --val_path data/processed/val.jsonl \
  --base_model mistralai/Mistral-7B-Instruct-v0.1 \
  --output_dir outputs/ \
  --max_steps 500 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_steps 50 \
  --weight_decay 0.0 \
  --max_seq_length 2048 \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --seed 42
```

### 3.2 Arguments

Key flags (see `--help` for the full list):

- `--train_path`, `--val_path`:
  - Paths to the processed Alpaca-style JSONL files.

- `--base_model`:
  - Base model name; default: `mistralai/Mistral-7B-Instruct-v0.1`.

- `--output_dir`:
  - Directory for all outputs (adapters, metrics, meta).

- `--max_steps`, `--per_device_train_batch_size`,
  `--gradient_accumulation_steps`:
  - Control how much training is performed and the effective batch size.

- `--learning_rate`, `--warmup_steps`, `--weight_decay`:
  - Standard optimizer hyperparameters.

- `--max_seq_length`:
  - Sequence length used for tokenization. Larger values increase VRAM usage.

- `--lora_r`, `--lora_alpha`, `--lora_dropout`:
  - LoRA hyperparameters that define the adapter capacity and regularization.

- `--seed`:
  - Random seed for reproducibility.

- `--dry_run`:
  - Loads and formats datasets, prints a few sample prompts, and exits. No
    model is loaded.

- `--smoke`:
  - Validates dataset and configuration. On CPU-only systems, skips model
    loading; on GPU, can be extended to attempt a lightweight model load.

### 3.3 Outputs

After a **full training** run (not `--dry_run` / `--smoke`), you should see:

- `outputs/adapters/`:
  - LoRA adapter weights and tokenizer config.

- `outputs/run_meta.json`:
  - Contains:
    - Base model name.
    - Dataset paths.
    - Training hyperparameters (`TrainingConfig`).
    - Dataset sizes.
    - Git commit (if available).
    - Run mode (`train`, `smoke`, or `dry_run`).

- `outputs/metrics.json`:
  - Contains training and evaluation metrics if available from the trainer.

For `--dry_run` and `--smoke`, the script still writes lightweight
`run_meta.json` and `metrics.json` with explanatory notes.

---

## 4. Colab Usage

To use the notebook in Google Colab:

1. Upload or clone the repository into Colab.
2. Open `notebooks/finetune_mistral7b_qlora_text2sql.ipynb`.
3. Run the environment setup cell:
   - Installs `unsloth`, `trl`, `bitsandbytes`, etc.
4. Mount Google Drive (optional) to persist `outputs/` and `data/`.
5. Run the preprocessing step or copy `data/processed/*.jsonl` into the
   environment.
6. Enable the **fast dev run** section first:
   - Verifies that everything works with a very small subset of data.
7. Once satisfied, disable fast dev mode and run a full training regime.

---

## 5. Troubleshooting & Tips

### 5.1 Out-of-memory (OOM) errors

If you encounter CUDA OOM errors:

- **Reduce effective batch size**:
  - Lower `--per_device_train_batch_size`.
  - Increase `--gradient_accumulation_steps` to maintain the same effective
    batch size.

- **Shorten sequence length**:
  - Reduce `--max_seq_length` (e.g., from 2048 → 1024 or 768).

- **Close other GPU processes**:
  - Free up GPU memory used by other notebooks or services.

### 5.2 Slow training

- Consider:
  - Reducing `max_steps` for exploratory runs.
  - Using mixed precision (bf16/fp16) where supported.
  - Ensuring data loading is not a bottleneck.

### 5.3 CPU-only environments

- Training itself requires a GPU.
- However, you can still:
  - Run `python scripts/train_qlora.py --dry_run` to validate dataset
    formatting and prompt construction.
  - Run `python scripts/train_qlora.py --smoke` to validate config and
    pipeline wiring.
  - Use some notebook cells for data inspection and prompt experimentation.

---

## 6. External Validation (Spider dev)

After primary training on `b-mc2/sql-create-context`, we run a **secondary
external validation** on the **Spider dev** split (via `xlangai/spider` with
schemas from `richardr1126/spider-schema`). This provides a harder, more
cross-domain benchmark (multi-table joins, many databases, compositional
generalization).

The external validation pipeline is implemented in:

- `scripts/evaluate_spider_external.py`

and documented in:

- [`docs/external_validation.md`](./external_validation.md)
- [`docs/evaluation.md`](./evaluation.md)

Spider is licensed under **CC BY-SA 4.0** and is used **only for evaluation**
in this project, not for training.