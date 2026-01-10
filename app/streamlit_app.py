"""
Streamlit UI for Analytics Copilot (Text-to-SQL).

This app is intentionally **UI-only**:

- It does NOT load any local models or GPU resources.
- All inference is performed remotely via Hugging Face Inference
  (`huggingface_hub.InferenceClient`), making it suitable for
  Streamlit Community Cloud deployments.

Configuration is provided via Streamlit secrets (`.streamlit/secrets.toml`):

Required:
    HF_TOKEN      = "hf_xxx"           # Hugging Face access token
    HF_MODEL_ID   = "username/model"   # Served model id OR adapter/merged model

Optional:
    HF_INFERENCE_BASE_URL = "https://..."  # Dedicated Inference Endpoint / TGI URL
    HF_PROVIDER           = "auto"        # Provider hint for InferenceClient(model=...)

Priority:
1. If HF_INFERENCE_BASE_URL is non-empty, we call:
       InferenceClient(base_url=..., api_key=HF_TOKEN)
2. Otherwise we call:
       InferenceClient(model=HF_MODEL_ID, api_key=HF_TOKEN, provider=HF_PROVIDER)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import streamlit as st
from huggingface_hub import InferenceClient  # type: ignore[import]


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure lightweight logging for the Streamlit app."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


@st.cache_resource(show_spinner=False)
def _get_cached_client(
    hf_token: str,
    base_url: str,
    model_id: str,
    provider: str,
    timeout_s: int,
) -> InferenceClient:
    """
    Construct and cache a Hugging Face InferenceClient instance.

    The cache key is derived from the provided parameters, so changing any of
    them in Streamlit secrets will cause a new client to be created.
    """
    if base_url:
        logger.info("Creating InferenceClient with base_url=%s", base_url)
        return InferenceClient(base_url=base_url, api_key=hf_token, timeout=timeout_s)

    logger.info(
        "Creating InferenceClient with model=%s and provider=%s",
        model_id,
        provider,
    )
    return InferenceClient(
        model=model_id,
        api_key=hf_token,
        provider=provider,
        timeout=timeout_s,
    )


def _build_prompt(schema: str, question: str) -> Tuple[str, str]:
    """
    Build the system and user prompt content for text-to-SQL generation.

    The prompt format mirrors the training/evaluation pipeline:
        ### Schema:
        <DDL>

        ### Question:
        <natural language question>
    """
    system_prompt = (
        "You are a careful text-to-SQL assistant. "
        "Given a database schema and a question, you respond with a single SQL "
        "query that answers the question. "
        "Return ONLY the SQL query, without explanation or commentary."
    )

    user_prompt = f"""### Schema:
{schema.strip()}

### Question:
{question.strip()}

Return only the SQL query."""
    return system_prompt, user_prompt


def _create_inference_client(timeout_s: int = 45) -> InferenceClient:
    """
    Create (or retrieve) an InferenceClient based on Streamlit secrets.

    Raises Streamlit errors if required configuration is missing so the user
    sees actionable feedback in the UI.
    """
    secrets = st.secrets

    hf_token = (secrets.get("HF_TOKEN") or "").strip()
    if not hf_token:
        st.error(
            "Missing `HF_TOKEN` in Streamlit secrets. "
            "Set it in `.streamlit/secrets.toml` or Streamlit Cloud settings."
        )
        st.stop()

    base_url = (secrets.get("HF_INFERENCE_BASE_URL") or "").strip()
    model_id = (secrets.get("HF_MODEL_ID") or "").strip()
    provider = (secrets.get("HF_PROVIDER") or "auto").strip() or "auto"

    if not base_url and not model_id:
        st.error(
            "Neither `HF_INFERENCE_BASE_URL` nor `HF_MODEL_ID` is configured. "
            "Set at least one in your Streamlit secrets."
        )
        st.stop()

    return _get_cached_client(
        hf_token=hf_token,
        base_url=base_url,
        model_id=model_id,
        provider=provider,
        timeout_s=timeout_s,
    )


def _call_model(
    client: InferenceClient,
    schema: str,
    question: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int = 45,
) -> Tuple[Optional[str], str]:
    """
    Call the remote model via chat_completion and return the generated SQL.

    Returns a tuple of (sql_text_or_none, user_prompt), where sql_text_or_none
    is None if the call failed.
    """
    system_prompt, user_prompt = _build_prompt(schema=schema, question=question)

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error while calling Hugging Face Inference API.", exc_info=True)
        st.error(
            "The Hugging Face Inference endpoint did not respond successfully. "
            "This can happen if the endpoint is cold, overloaded, or misconfigured. "
            "Please try again, or check your HF model / endpoint settings."
        )
        st.caption(f"Details: {exc}")
        return None, user_prompt

    # InferenceClient may return an object or a dict; handle both gracefully.
    try:
        if hasattr(response, "choices"):
            choice = response.choices[0]
            # Newer client versions expose choice.message as dict-like.
            message = getattr(choice, "message", None) or choice["message"]
            content = message["content"]
        else:
            choice = response["choices"][0]
            message = choice["message"]
            content = message["content"]
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected response format from InferenceClient.", exc_info=True)
        st.error(
            "Received an unexpected response format from the Hugging Face "
            "Inference API. Please check your endpoint and model configuration."
        )
        st.caption(f"Details: {exc}")
        return None, user_prompt

    sql_text = (content or "").strip()
    return sql_text, user_prompt


def main() -> None:
    """Render the Streamlit UI."""
    _configure_logging()
    st.set_page_config(
        page_title="Analytics Copilot – Text-to-SQL",
        layout="centered",
    )

    st.title("Analytics Copilot – Text-to-SQL")
    st.markdown(
        "This demo converts natural-language questions into SQL using a "
        "remote model hosted on **Hugging Face Inference**. "
        "The model is not loaded inside the Streamlit app; all heavy lifting "
        "happens on a remote endpoint or serverless provider."
    )

    st.markdown("### Inputs")

    schema = st.text_area(
        "Database schema (DDL)",
        height=220,
        placeholder="CREATE TABLE orders (...);\nCREATE TABLE customers (...);",
    )

    question = st.text_area(
        "Question (natural language)",
        height=140,
        placeholder="What is the total order amount per customer for the last 7 days?",
    )

    with st.expander("Advanced generation settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Sampling temperature for the model (0.0 = greedy).",
            )
        with col2:
            max_new_tokens = st.slider(
                "Max new tokens",
                min_value=32,
                max_value=512,
                value=256,
                step=16,
                help="Maximum number of tokens to generate for the SQL query.",
            )

    generate_clicked = st.button("Generate SQL", type="primary")

    if generate_clicked:
        if not schema.strip():
            st.warning("Please provide a database schema (DDL) before generating SQL.")
            return
        if not question.strip():
            st.warning("Please provide a natural-language question.")
            return

        with st.spinner("Calling Hugging Face Inference API..."):
            client = _create_inference_client(timeout_s=45)
            sql_text, user_prompt = _call_model(
                client=client,
                schema=schema,
                question=question,
                temperature=temperature,
                max_tokens=max_new_tokens,
                timeout_s=45,
            )

        if sql_text is not None:
            st.subheader("Generated SQL")
            st.code(sql_text, language="sql")

            with st.expander("Show prompt", expanded=False):
                st.code(user_prompt, language="markdown")
        else:
            st.warning(
                "No SQL was generated due to an error when calling the remote "
                "inference endpoint. Please review the error message above."
            )


if __name__ == "__main__":
    main()