# Deploying the Text-to-SQL Adapter on Hugging Face

This guide walks through publishing a trained Text-to-SQL LoRA/QLoRA adapter
to the Hugging Face Hub and deploying it via **Inference Endpoints** using
**Multi-LoRA** in Text Generation Inference (TGI).

The recommended pattern is:

- Host the **base model once** in an Inference Endpoint.
- Attach one or more **LoRA adapters** via the `LORA_ADAPTERS` environment
  variable.
- Select the adapter at request time using `adapter_id`.

---

## 1. Create a Hugging Face token

1. Go to <https://huggingface.co/settings/tokens>.
2. Create a new token with at least:
   - **Write** access to model repos for publishing.
   - **Read** access for inference.

You can authenticate in two ways:

```bash
huggingface-cli login
```

or set the environment variable:

```bash
export HF_TOKEN="hf_your_token_here"
```

The `scripts/publish_to_hub.py` script uses this token both to validate
authentication and to push adapter artifacts.

---

## 2. Create a model repository (optional)

You can either:

- Create a repo in the UI at <https://huggingface.co/new>, or
- Let `publish_to_hub.py` create it for you automatically.

If you prefer CLI:

```bash
huggingface-cli repo create your-username/analytics-copilot-text2sql-mistral7b-qlora \
  --type model
```

`publish_to_hub.py` will also call `create_repo(..., exist_ok=True)` so it is
safe to rerun.

---

## 3. Publish the adapter with `scripts/publish_to_hub.py`

Assuming you have run training and have an adapter saved under:

- `outputs/adapters/`
  - `adapter_config.json`
  - `adapter_model.safetensors` (or `adapter_model.bin`)

you can publish it using:

```bash
python scripts/publish_to_hub.py \
  --repo_id your-username/analytics-copilot-text2sql-mistral7b-qlora \
  --adapter_dir outputs/adapters
```

The script will:

1. Validate **adapter contents**:
   - Require `adapter_config.json`.
   - Require `adapter_model.safetensors` **or** `adapter_model.bin`.
2. Validate **Hugging Face authentication** via `HfApi().whoami()`.
3. Create or reuse the Hub repo (`repo_type="model"`).
4. Generate a **minimal README.md** (if none exists) that includes:
   - Base model name (from `adapter_config.json.base_model_name_or_path`).
   - Task description: Text-to-SQL (schema + question → SQL).
   - Evaluation commands (internal + Spider external).
   - Deployment notes for Inference Endpoints + Multi-LoRA.
5. Upload the entire adapter directory to the model repo.

### 3.1 Skipping or tightening README generation

You can opt out of auto README generation:

```bash
python scripts/publish_to_hub.py \
  --repo_id your-username/analytics-copilot-text2sql-mistral7b-qlora \
  --adapter_dir outputs/adapters \
  --skip_readme
```

- If `--skip_readme` is set:
  - The script **does not** create or modify README.md.
  - Any existing README.md in `adapter_dir` is left as-is.

You can also enforce strict behavior when README generation fails:

```bash
python scripts/publish_to_hub.py \
  --repo_id your-username/analytics-copilot-text2sql-mistral7b-qlora \
  --adapter_dir outputs/adapters \
  --strict_readme
```

- If `--strict_readme` is set:
  - Any error while generating README.md will cause the script to **exit non-zero**.
- If it is **not** set (default):
  - README errors are logged and the adapter upload still proceeds.

### 3.2 Including evaluation metrics in the README

If you have run evaluation scripts and saved metrics to a JSON report
(e.g. `reports/eval_internal.json` or `reports/eval_spider.json`), you can
embed a summary into the README:

```bash
python scripts/publish_to_hub.py \
  --repo_id your-username/analytics-copilot-text2sql-mistral7b-qlora \
  --adapter_dir outputs/adapters \
  --include_metrics reports/eval_internal.json
```

The script expects either:

- A raw metrics dict, or
- An object with a top-level `metrics` key (as produced by the evaluation
  scripts).

---

## 4. Create an Inference Endpoint with the base model

Next, create an Inference Endpoint for the **base model** only, for example:

- Base model: `mistralai/Mistral-7B-Instruct-v0.1`
- Task: Text Generation
- Accelerator: a GPU instance (e.g. A10G, A100) suitable for Mistral-7B
- Implementation: **Text Generation Inference (TGI)**

You can create the endpoint via the HF UI:

1. Go to <https://huggingface.co/inference-endpoints>.
2. Click **Create Endpoint**.
3. Choose the base model (e.g. `mistralai/Mistral-7B-Instruct-v0.1`).
4. Configure hardware and autoscaling.
5. In **Advanced configuration** / environment variables, you will set
   `LORA_ADAPTERS` as described in the next section.

---

## 5. Attach the adapter via `LORA_ADAPTERS` (Multi-LoRA)

TGI supports loading multiple LoRA adapters using the `LORA_ADAPTERS`
environment variable, which contains a JSON array describing each adapter.

For a single Text-to-SQL adapter, set:

```bash
LORA_ADAPTERS='[
  {"id": "text2sql-qlora", "source": "your-username/analytics-copilot-text2sql-mistral7b-qlora"}
]'
```

- `id`: Logical adapter identifier you will reference as `adapter_id` in
  requests (e.g. `"text2sql-qlora"`).
- `source`: Hugging Face Hub model repo containing the adapter (the same
  `repo_id` you passed to `publish_to_hub.py`).

You can set this in the endpoint’s **Environment variables** section in
the UI, or via the Inference Endpoints API.

After saving the configuration and deploying the endpoint, TGI will:

- Load the base Mistral-7B model.
- Load the LoRA adapter weights from the specified repo.
- Register the adapter under `id="text2sql-qlora"`.

---

## 6. Sending requests with `adapter_id`

To use the Text-to-SQL adapter at inference time, include `adapter_id` in
your request parameters.

### 6.1 Raw HTTP request example

Assuming your endpoint URL is:

```text
https://your-endpoint-1234.us-east-1.aws.endpoints.huggingface.cloud
```

you can send a POST request:

```bash
curl -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  https://your-endpoint-1234.us-east-1.aws.endpoints.huggingface.cloud \
  -d '{
    "inputs": "### Schema:\n<DDL here>\n\n### Question:\n<NL question>",
    "parameters": {
      "adapter_id": "text2sql-qlora",
      "max_new_tokens": 256,
      "temperature": 0.0
    }
  }'
```

Key points:

- `inputs` should contain the **schema + question** prompt in the same
  format used for training/evaluation.
- `parameters.adapter_id` selects the **LoRA adapter** to apply.
- Other generation parameters (`max_new_tokens`, `temperature`, etc.)
  can be tuned as usual.

### 6.2 Using `huggingface_hub.InferenceClient`

You can also call the endpoint from Python:

```python
from huggingface_hub import InferenceClient

ENDPOINT_URL = "https://your-endpoint-1234.us-east-1.aws.endpoints.huggingface.cloud"

client = InferenceClient(
    base_url=ENDPOINT_URL,
    api_key="hf_your_token_here",
)

schema = """CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER,
  amount NUMERIC,
  created_at TIMESTAMP
);"""

question = "Total order amount per customer for the last 7 days."

prompt = f"""### Schema:
{schema}

### Question:
{question}

Return only the SQL query."""

response = client.post(
    json={
        "inputs": prompt,
        "parameters": {
            "adapter_id": "text2sql-qlora",
            "max_new_tokens": 256,
            "temperature": 0.0,
        },
    }
)

print(response)
```

Depending on how your endpoint is configured, you may also be able to use
`client.text_generation` or `client.chat_completion` with provider-specific
options for `adapter_id`. The raw `post` call shown above works with the
standard TGI JSON API.

---

## 7. Summary

- Use `scripts/publish_to_hub.py` to:
  - Validate adapter files (`adapter_config.json`, adapter weights).
  - Optionally generate a minimal README.md.
  - Push the adapter to a Hub model repo.
- Deploy the **base model** once via an Inference Endpoint (TGI).
- Attach the adapter by setting `LORA_ADAPTERS` with the adapter repo id.
- At inference time, select the adapter using `adapter_id` in your request,
  and pass the schema + question prompt in the same format as training.

This pattern keeps deployment efficient (one base model, multiple adapters)
and makes it easy to switch between different specialized adapters without
reprovisioning new endpoints.