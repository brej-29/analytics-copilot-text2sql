# Deploying the Streamlit App to Streamlit Community Cloud

This guide walks through deploying the Analytics Copilot Streamlit UI to
**Streamlit Community Cloud** using this repository as the source of truth.

The Streamlit app is UI-only and lives at:

- `app/streamlit_app.py`

It talks to remote inference backends (Hugging Face Inference + optional OpenAI
fallback) and does not require a GPU on the Streamlit side.

---

## 1. Prerequisites

Before creating the app on Streamlit Community Cloud:

1. Push this repository to GitHub (public or private, depending on your plan).
2. Ensure the app runs locally:

   ```bash
   # Create a virtualenv and install dependencies (see README for details)
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

   # Optional but recommended: check runtime config
   python scripts/check_runtime_config.py

   # Start the Streamlit app
   streamlit run app/streamlit_app.py
   ```

3. Confirm the UI loads and you can see the input fields (schema + question).

For more background on Streamlit secrets and `secrets.toml`, see the official
Streamlit docs:

- Secrets management on Community Cloud  
  (see [`Secrets management` in the Streamlit docs][streamlit-secrets-docs])
- Local `secrets.toml` file format  
  (see [`secrets.toml` reference][streamlit-secrets-toml-docs])

---

## 2. Create the app on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud.
2. Click **New app** and choose **From existing repo**.
3. Select the GitHub repository that contains this project.
4. Choose the branch you want to deploy from (typically `main` or a feature branch).
5. Set the **Main file path** (entrypoint) to:

   ```text
   app/streamlit_app.py
   ```

6. Click **Deploy**. The app will build and start, but it will show
   configuration errors until secrets are provided.

---

## 3. Configure secrets (HF + OpenAI) in Streamlit Cloud

The Streamlit app expects configuration via **Streamlit secrets** (with
environment variables as a secondary fallback). On Community Cloud, secrets are
managed entirely through the web UI.

### 3.1 Local secrets file (for reference)

Locally, secrets are stored in:

```text
.streamlit/secrets.toml
```

This file is **ignored by git** (see `.gitignore`) and should never be
committed. The repository provides an example template:

```text
.streamlit/secrets.toml.example
```

You can copy it locally:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml` and fill in the values for your environment.
The same keys and values are what you will paste into the Streamlit Cloud
**Secrets** UI.

### 3.2 Preferred configuration: Hugging Face Inference Endpoint + adapter

Recommended for production-like usage:

```toml
HF_TOKEN        = "hf_your_access_token_here"

# Dedicated Inference Endpoint / TGI URL
HF_ENDPOINT_URL = "https://your-endpoint-1234.us-east-1.aws.endpoints.huggingface.cloud"

# Adapter identifier configured in your endpoint's LORA_ADAPTERS
HF_ADAPTER_ID   = "text2sql-qlora"
```

Notes:

- `HF_TOKEN` should be a token with permission to call your Inference Endpoint.
- `HF_ENDPOINT_URL` points at a dedicated Text Generation Inference (TGI)
  endpoint (often with Multi-LoRA configured).
- `HF_ADAPTER_ID` must match the adapter `id` you configured in the endpoint
  `LORA_ADAPTERS` setting.

The app also understands `HF_INFERENCE_BASE_URL` as an alias for
`HF_ENDPOINT_URL`. If both are set, `HF_ENDPOINT_URL` takes precedence.

### 3.3 Fallback configuration: HF router with a merged model

If you prefer to call a provider-managed merged model via the HF router (no
adapters), you can configure:

```toml
HF_TOKEN    = "hf_your_access_token_here"
HF_MODEL_ID = "your-username/your-merged-text2sql-model"
HF_PROVIDER = "auto"  # optional provider hint
```

Notes:

- Do **not** point `HF_MODEL_ID` at a pure adapter repo; most providers will
  respond with `model_not_supported` in that case.
- For adapter-based inference, use the dedicated endpoint pattern above
  (`HF_ENDPOINT_URL` + `HF_ADAPTER_ID`).

### 3.4 Optional OpenAI fallback configuration

When HF inference fails (for example, endpoint paused or network issues),
the app can automatically fall back to a cheap OpenAI model.

To enable this, add:

```toml
OPENAI_API_KEY          = "sk_..."      # required to use the fallback
OPENAI_FALLBACK_MODEL   = "gpt-5-nano"  # default if omitted
OPENAI_FALLBACK_STRICT_JSON = "false"   # or "true" to request structured JSON
```

- If `OPENAI_API_KEY` is missing, the OpenAI fallback is disabled.
- `OPENAI_FALLBACK_MODEL` defaults to `"gpt-5-nano"` when not set, matching the
  app’s internal default.
- When `OPENAI_FALLBACK_STRICT_JSON` is a truthy value (`"true"`, `"1"`,
  `"yes"`, `"on"`), the app asks for a JSON object of the form
  `{"sql": "SELECT ..."}` and tries to parse the `sql` field.

### 3.5 Pasting secrets into Streamlit Cloud

In the Streamlit Cloud UI:

1. Open your deployed app.
2. Go to the app’s **Settings** → **Secrets** section.
3. Copy the contents of your local `.streamlit/secrets.toml` (or build it from
   the example) and paste it into the secrets editor.
4. Save the secrets and restart the app if necessary.

The format is exactly the same TOML syntax as
`.streamlit/secrets.toml.example`.

---

## 4. Verify runtime configuration

Before relying on the deployed app, you can validate your configuration locally
using the runtime config checker:

```bash
python scripts/check_runtime_config.py
```

This script:

- Looks for `.streamlit/secrets.toml` in the project root (if present).
- Checks environment variables as a fallback.
- Prints which keys are configured (masking sensitive values).
- Exits with a non-zero status if **neither**:
  - a usable Hugging Face configuration (`HF_TOKEN` plus either
    `HF_ENDPOINT_URL`/`HF_INFERENCE_BASE_URL` or `HF_MODEL_ID`), nor
  - an `OPENAI_API_KEY`
  is set.

A successful run will report that at least one provider (HF and/or OpenAI) is
configured.

---

## 5. Using the Diagnostics panel

Once the app is deployed and secrets are configured:

- Trigger a generation by providing a schema and question and clicking
  **Generate SQL**.
- Open the **Diagnostics** section in the sidebar.

The Diagnostics panel shows:

- Which provider handled the last request (`HF` vs `OpenAI fallback`).
- Whether the last HF call reported that the endpoint is paused (with a hint
  to resume it from the Hugging Face Inference Endpoint **Overview** page).
- The request duration (in milliseconds).
- The `max_new_tokens` value used for the last request.

The diagnostics are intentionally lightweight and do **not** show stack traces
or detailed exception internals, to keep the UI clean and safe for end users.

---

## 6. Summary

To deploy this app on Streamlit Community Cloud:

1. Push the repository to GitHub.
2. Create a new Streamlit app from that repo and set the main file to
   `app/streamlit_app.py`.
3. Configure secrets in the Streamlit Cloud **Secrets** editor, using the keys
   from `.streamlit/secrets.toml.example`.
4. (Optional) Run `python scripts/check_runtime_config.py` locally to validate
   configuration before pushing changes.
5. Deploy and use the Diagnostics panel to verify which provider is serving
   requests and whether your Hugging Face endpoint is healthy.

With these steps, there should be no guesswork required to get a working
deployment on Streamlit Community Cloud.

---

[streamlit-secrets-docs]: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
[streamlit-secrets-toml-docs]: https://docs.streamlit.io/develop/api-reference/connections/secrets.toml