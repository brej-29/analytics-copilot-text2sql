"""
Streamlit UI for Analytics Copilot (Text-to-SQL).

This app is intentionally **UI-only**:

- It does NOT load any local models or GPU resources.
- All inference is performed remotely via Hugging Face Inference
  (`huggingface_hub.InferenceClient`), making it suitable for
  Streamlit Community Cloud deployments.

Configuration can be provided via Streamlit secrets (`.streamlit/secrets.toml`)
or environment variables. Secrets take precedence over environment variables.

Required:
    HF_TOKEN        = "hf_xxx"   # Hugging Face access token

Preferred (dedicated endpoint + adapters):
    HF_ENDPOINT_URL = "https://..."   # Dedicated Inference Endpoint / TGI URL
    HF_ADAPTER_ID   = "adapter-id"    # Adapter identifier configured in TGI

Fallback (provider/router-based; no adapters):
    HF_MODEL_ID     = "username/model"   # Provider-supported merged model
    HF_PROVIDER     = "auto"             # Provider hint for InferenceClient(model=...)

Compatibility:
    HF_INFERENCE_BASE_URL is also supported as an alias for HF_ENDPOINT_URL.

Priority:
1. If HF_ENDPOINT_URL (or HF_INFERENCE_BASE_URL) is non-empty, we call:

       InferenceClient(base_url=HF_ENDPOINT_URL, api_key=HF_TOKEN)

   and send `adapter_id=HF_ADAPTER_ID` in `text_generation` requests.

2. Otherwise, we fall back to provider-based routing via the HF router:

       InferenceClient(model=HF_MODEL_ID, api_key=HF_TOKEN, provider=HF_PROVIDER)

   Note that pure adapter repositories are **not** supported by the router; use
   a dedicated endpoint for adapter-based inference.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Mapping, NamedTuple, Optional, Tuple

import streamlit as st
from huggingface_hub import InferenceClient  # type: ignore[import]
from openai import OpenAI  # type: ignore[import]


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure lightweight logging for the Streamlit app."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


class HFConfig(NamedTuple):
    """Resolved Hugging Face configuration for the Streamlit app."""

    hf_token: str
    endpoint_url: str
    model_id: str
    provider: str
    adapter_id: Optional[str]


def _resolve_hf_config(
    secrets: Mapping[str, Any],
    environ: Mapping[str, str],
) -> HFConfig:
    """
    Resolve Hugging Face configuration from secrets and environment variables.

    Precedence:
    - Secrets take priority over environment variables.
    - HF_ENDPOINT_URL and HF_INFERENCE_BASE_URL are treated as aliases.
    """

    def _get_from_mapping(mapping: Mapping[str, Any], key: str) -> str:
        try:
            value = mapping.get(key)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            value = None
        if value is None:
            return ""
        return str(value).strip()

    hf_token = _get_from_mapping(secrets, "HF_TOKEN") or environ.get(
        "HF_TOKEN",
        "",
    ).strip()

    endpoint_url = (
        _get_from_mapping(secrets, "HF_ENDPOINT_URL")
        or _get_from_mapping(secrets, "HF_INFERENCE_BASE_URL")
        or environ.get("HF_ENDPOINT_URL", "").strip()
        or environ.get("HF_INFERENCE_BASE_URL", "").strip()
    )

    model_id = _get_from_mapping(secrets, "HF_MODEL_ID") or environ.get(
        "HF_MODEL_ID",
        "",
    ).strip()

    provider = (
        _get_from_mapping(secrets, "HF_PROVIDER")
        or environ.get("HF_PROVIDER", "").strip()
        or "auto"
    )

    adapter_id_raw = _get_from_mapping(secrets, "HF_ADAPTER_ID") or environ.get(
        "HF_ADAPTER_ID",
        "",
    ).strip()
    adapter_id = adapter_id_raw or None

    return HFConfig(
        hf_token=hf_token,
        endpoint_url=endpoint_url,
        model_id=model_id,
        provider=provider,
        adapter_id=adapter_id,
    )


def _get_openai_settings() -> Tuple[str, str]:
    """
    Resolve OpenAI fallback settings from Streamlit secrets and environment variables.

    Secrets take precedence over environment variables. The model name falls back
    to "gpt-5-nano" when not configured explicitly.
    """
    try:
        secrets: Mapping[str, Any] = st.secrets  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        secrets = {}

    def _get_from_mapping(mapping: Mapping[str, Any], key: str) -> str:
        try:
            value = mapping.get(key)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            value = None
        if value is None:
            return ""
        return str(value).strip()

    api_key = _get_from_mapping(secrets, "OPENAI_API_KEY") or os.environ.get(
        "OPENAI_API_KEY",
        "",
    ).strip()

    model_name = _get_from_mapping(secrets, "OPENAI_FALLBACK_MODEL") or os.environ.get(
        "OPENAI_FALLBACK_MODEL",
        "",
    ).strip()

    if not model_name:
        model_name = "gpt-5-nano"

    return api_key, model_name


def _is_openai_strict_json_enabled() -> bool:
    """
    Return True if structured JSON mode is enabled for the OpenAI fallback.

    Controlled by OPENAI_FALLBACK_STRICT_JSON in Streamlit secrets or environment
    variables. Defaults to False.
    """
    try:
        secrets: Mapping[str, Any] = st.secrets  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        secrets = {}

    def _get_from_mapping(mapping: Mapping[str, Any], key: str) -> str:
        try:
            value = mapping.get(key)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            value = None
        if value is None:
            return ""
        return str(value).strip()

    raw_value = _get_from_mapping(secrets, "OPENAI_FALLBACK_STRICT_JSON") or os.environ.get(
        "OPENAI_FALLBACK_STRICT_JSON",
        "",
    ).strip()

    return raw_value.lower() in {"1", "true", "yes", "on"}


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
    them in Streamlit secrets or environment variables will cause a new client
    to be created.
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


def _openai_response_text(resp: Any) -> str:
    """
    Extract text content from an OpenAI Responses API response object.

    1) Prefer the convenience `output_text` attribute when present and non-empty.
    2) Otherwise, iterate over `resp.output` and aggregate any text content blocks.

    The Responses API typically exposes text as:
        resp.output[0].content[0].text.value
    but this helper is defensive and also supports dict-like structures.
    """
    # 1) Convenience attribute, when present.
    try:
        value = getattr(resp, "output_text", None)
    except Exception:  # noqa: BLE001
        value = None

    if isinstance(value, str) and value.strip():
        return value.strip()

    # 2) Walk the output structure.
    try:
        output = getattr(resp, "output", None)
    except Exception:  # noqa: BLE001
        output = None

    if not output:
        return ""

    chunks: list[str] = []

    try:
        for item in output:
            # Handle both object-style and dict-style access to content.
            content_list: Any
            if isinstance(item, dict):
                content_list = item.get("content")
            else:
                try:
                    content_list = getattr(item, "content", None)
                except Exception:  # noqa: BLE001
                    content_list = None

            if not content_list:
                continue

            for content in content_list:
                # Retrieve the "text" field from the content object or dict.
                text_obj: Any = None
                if isinstance(content, dict):
                    text_obj = content.get("text")
                else:
                    try:
                        text_obj = getattr(content, "text", None)
                    except Exception:  # noqa: BLE001
                        text_obj = None

                if text_obj is None:
                    continue

                # Unwrap the actual string value, handling nested objects/dicts.
                text_val: Optional[str] = None
                if isinstance(text_obj, str):
                    text_val = text_obj
                else:
                    # Object-style: content.text.value
                    try:
                        inner_val = getattr(text_obj, "value", None)
                    except Exception:  # noqa: BLE001
                        inner_val = None
                    if isinstance(inner_val, str):
                        text_val = inner_val
                    elif isinstance(text_obj, dict):
                        # Dict-style: {"text": {"value": "..."}} or similar.
                        inner_val = text_obj.get("value")
                        if isinstance(inner_val, str):
                            text_val = inner_val

                if isinstance(text_val, str) and text_val:
                    chunks.append(text_val)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to extract text from OpenAI response.output.")
        return ""

    return "".join(chunks).strip()


def _call_openai_fallback(
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Call the OpenAI Responses API as a fallback when HF inference fails.

    Uses the cheapest nano model (default: gpt-5-nano) and avoids passing
    unsupported parameters such as temperature or top_p.
    """
    api_key, model_name = _get_openai_settings()
    if not api_key:
        raise RuntimeError("OpenAI fallback not configured")

    client = OpenAI(api_key=api_key)
    full_input = f"{system_prompt}\n\n{user_prompt}"

    extra_kwargs: dict[str, Any] = {}
    strict_json = _is_openai_strict_json_enabled()
    if strict_json:
        extra_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "sql_fallback",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string"},
                    },
                    "required": ["sql"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    response = client.responses.create(
        model=model_name,
        input=full_input,
        **extra_kwargs,
    )

    raw_text = _openai_response_text(response)

    if strict_json and raw_text:
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                sql_val = parsed.get("sql")
                if isinstance(sql_val, str) and sql_val.strip():
                    return sql_val.strip()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to parse structured JSON output from OpenAI fallback; "
                "falling back to raw text.",
            )

    if not raw_text:
        logger.warning("OpenAI fallback returned empty text output.")
        return ""

    return raw_text.strip()


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


def _create_inference_client(timeout_s: int = 45) -> Tuple[InferenceClient, HFConfig]:
    """
    Create (or retrieve) an InferenceClient based on Streamlit secrets/env.

    Raises Streamlit errors if required configuration is missing so the user
    sees actionable feedback in the UI.
    """
    secrets = st.secrets
    hf_config = _resolve_hf_config(secrets=secrets, environ=os.environ)

    if not hf_config.hf_token:
        st.error(
            "Missing `HF_TOKEN` in Streamlit secrets or environment. "
            "Set it in `.streamlit/secrets.toml` or as the `HF_TOKEN` "
            "environment variable."
        )
        st.stop()

    if not hf_config.endpoint_url and not hf_config.model_id:
        st.error(
            "Neither `HF_ENDPOINT_URL`/`HF_INFERENCE_BASE_URL` nor `HF_MODEL_ID` "
            "is configured. Set at least one via Streamlit secrets or "
            "environment variables."
        )
        st.stop()

    if hf_config.endpoint_url and hf_config.adapter_id is None:
        st.error(
            "HF_ENDPOINT_URL is set but `HF_ADAPTER_ID` is missing. "
            "For adapter-based inference with a dedicated endpoint, set "
            "`HF_ADAPTER_ID` to the adapter identifier configured in your "
            "Text Generation Inference (TGI) endpoint (the value used as "
            "`adapter_id` in requests)."
        )
        st.stop()

    if hf_config.endpoint_url:
        logger.info(
            "Using dedicated Hugging Face Inference Endpoint at %s",
            hf_config.endpoint_url,
        )
    else:
        logger.info(
            "Using provider-based HF Inference routing with model=%s, provider=%s",
            hf_config.model_id,
            hf_config.provider,
        )

    client = _get_cached_client(
        hf_token=hf_config.hf_token,
        base_url=hf_config.endpoint_url,
        model_id=hf_config.model_id,
        provider=hf_config.provider,
        timeout_s=timeout_s,
    )

    return client, hf_config


def _extract_sql_from_text(raw_text: str) -> str:
    """
    Extract SQL from model output text.

    Currently implemented as a simple pass-through that returns the
    stripped text. Defined as a separate helper so it can be extended
    or replaced in tests without changing call sites.
    """
    return raw_text.strip()


def _update_last_request_diagnostics(
    provider_label: str,
    endpoint_paused: bool,
    duration_ms: float,
    max_tokens: int,
) -> None:
    """
    Store lightweight diagnostics about the last inference request.

    Diagnostics are kept in Streamlit's session_state (when available) so that
    the UI can render them in a dedicated Diagnostics panel without exposing
    stack traces or raw exception objects.
    """
    try:
        state = st.session_state  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return

    try:
        state["diagnostics_last_request"] = {
            "provider": provider_label,
            "endpoint_paused": bool(endpoint_paused),
            "duration_ms": float(duration_ms),
            "max_tokens": int(max_tokens),
        }
    except Exception:  # noqa: BLE001
        # Diagnostics should never break the main app flow.
        logger.debug("Failed to update diagnostics in session_state.", exc_info=True)


def _call_model(
    client: InferenceClient,
    schema: str,
    question: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int = 45,  # noqa: ARG001
    adapter_id: Optional[str] = None,
    use_endpoint: bool = False,
) -> Tuple[Optional[str], str]:
    """
    Call the remote model via text_generation and return the generated SQL.

    Returns a tuple of (sql_text_or_none, user_prompt), where sql_text_or_none
    is None if the call failed.

    When `use_endpoint` is True and `adapter_id` is provided, the request will
    include `adapter_id` to select the appropriate LoRA adapter on a TGI
    Inference Endpoint.
    """
    system_prompt, user_prompt = _build_prompt(schema=schema, question=question)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    generation_kwargs: dict[str, Any] = {
        "prompt": full_prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
    }
    if use_endpoint and adapter_id:
        generation_kwargs["adapter_id"] = adapter_id

    provider_label = "HF"
    endpoint_paused = False
    start = time.perf_counter()

    try:
        response = client.text_generation(**generation_kwargs)
    except Exception as exc:  # noqa: BLE001
        message = str(exc) if exc is not None else ""
        lower_message = message.lower()
        if "endpoint is paused" in lower_message:
            endpoint_paused = True
            logger.warning(
                "Hugging Face Inference endpoint is paused; using OpenAI fallback.",
            )
            # See HF Inference Endpoints docs on pausing/resuming endpoints:
            # https://huggingface.co/docs/inference-endpoints/en/guides/pause_endpoint
        else:
            logger.exception(
                "Error while calling Hugging Face Inference API. "
                "Attempting OpenAI fallback if configured.",
            )

        provider_label = "OpenAI fallback"
        try:
            raw_text = _call_openai_fallback(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception:  # noqa: BLE001
            logger.exception("OpenAI fallback inference failed.")
            st.error(
                "The service is temporarily unavailable. Please try again in a moment."
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            _update_last_request_diagnostics(
                provider_label=provider_label,
                endpoint_paused=endpoint_paused,
                duration_ms=duration_ms,
                max_tokens=max_tokens,
            )
            return None, user_prompt

        if not raw_text.strip():
            sql_text = "-- Fallback provider returned empty output. Try again."
        else:
            extracted = _extract_sql_from_text(raw_text)
            if extracted and extracted.strip():
                sql_text = extracted.strip()
            else:
                sql_text = raw_text.strip()

        st.caption("Using backup inference provider")
        duration_ms = (time.perf_counter() - start) * 1000.0
        _update_last_request_diagnostics(
            provider_label=provider_label,
            endpoint_paused=endpoint_paused,
            duration_ms=duration_ms,
            max_tokens=max_tokens,
        )
        return sql_text, user_prompt

    # InferenceClient.text_generation may return a string, a dict, or a list.
    try:
        if isinstance(response, str):
            text = response
        elif isinstance(response, dict) and "generated_text" in response:
            text = str(response["generated_text"])
        elif (
            isinstance(response, list)
            and response
            and isinstance(response[0], dict)
            and "generated_text" in response[0]
        ):
            text = str(response[0]["generated_text"])
        else:
            # Fallback: best-effort string representation.
            text = str(response)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected response format from InferenceClient.", exc_info=True)
        st.error(
            "Received an unexpected response format from the Hugging Face "
            "Inference API. Please check your endpoint and model configuration."
        )
        st.caption(f"Details: {exc}")
        duration_ms = (time.perf_counter() - start) * 1000.0
        _update_last_request_diagnostics(
            provider_label=provider_label,
            endpoint_paused=endpoint_paused,
            duration_ms=duration_ms,
            max_tokens=max_tokens,
        )
        return None, user_prompt

    sql_text = (text or "").strip()
    duration_ms = (time.perf_counter() - start) * 1000.0
    _update_last_request_diagnostics(
        provider_label=provider_label,
        endpoint_paused=endpoint_paused,
        duration_ms=duration_ms,
        max_tokens=max_tokens,
    )
    return sql_text, user_prompt


def _render_diagnostics_sidebar() -> None:
    """Render a lightweight diagnostics panel in the sidebar."""
    try:
        state = st.session_state  # type: ignore[attr-defined]
        diagnostics = state.get("diagnostics_last_request")  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        diagnostics = None

    with st.sidebar.expander("Diagnostics", expanded=False):
        if not isinstance(diagnostics, dict):
            st.caption("Run a request to see diagnostics for the last call.")
            return

        provider_label = diagnostics.get("provider") or "–"
        duration_ms = diagnostics.get("duration_ms")
        max_tokens = diagnostics.get("max_tokens")
        endpoint_paused = bool(diagnostics.get("endpoint_paused"))

        st.markdown(f"**Active provider (last request):** {provider_label}")

        if duration_ms is not None:
            try:
                duration_val = float(duration_ms)
            except Exception:  # noqa: BLE001
                duration_val = None
            if duration_val is not None:
                st.markdown(f"**Request duration:** {duration_val:.0f} ms")

        if max_tokens is not None:
            try:
                max_tokens_val = int(max_tokens)
            except Exception:  # noqa: BLE001
                max_tokens_val = None
            if max_tokens_val is not None:
                st.markdown(f"**Max new tokens parameter:** {max_tokens_val}")

        if endpoint_paused:
            # Friendly hint only; do not surface raw exception text or stack traces.
            st.info("HF endpoint is paused; resume it in HF Endpoints → Overview.")


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
                min_value=0.001,
                max_value=1.0,
                value=0.01,
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
            _render_diagnostics_sidebar()
            return
        if not question.strip():
            st.warning("Please provide a natural-language question.")
            _render_diagnostics_sidebar()
            return

        with st.spinner("Calling Hugging Face Inference API..."):
            client, hf_config = _create_inference_client(timeout_s=45)
            sql_text, user_prompt = _call_model(
                client=client,
                schema=schema,
                question=question,
                temperature=temperature,
                max_tokens=max_new_tokens,
                timeout_s=45,
                adapter_id=hf_config.adapter_id,
                use_endpoint=bool(hf_config.endpoint_url),
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

    _render_diagnostics_sidebar()


if __name__ == "__main__":
    main()