from pathlib import Path
import sys
from typing import Any

import pytest


def _ensure_root_on_path() -> None:
    """Ensure that the project root is available on sys.path for imports."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_root_on_path()

from app import streamlit_app  # noqa: E402  # isort: skip


class _DummyHFClient:
    def text_generation(self, **_: Any) -> str:
        raise RuntimeError("HF inference failed")


class _DummyOpenAIResponse:
    def __init__(self, text: str) -> None:
        self.output_text = text


class _DummyOpenAIResponses:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(
        self,
        model: str,
        input: str,
        max_output_tokens: int,
        **_: Any,
    ) -> _DummyOpenAIResponse:  # noqa: ARG002
        return _DummyOpenAIResponse(self._text)


class _DummyOpenAIClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.responses = _DummyOpenAIResponses("SELECT 1;")


class _DummyStreamlit:
    def __init__(self) -> None:
        self.caption_called = False
        self.error_called = False

    def caption(self, *_: Any, **__: Any) -> None:
        self.caption_called = True

    def error(self, *_: Any, **__: Any) -> None:
        self.error_called = True


def test_openai_response_text_uses_output_when_output_text_empty() -> None:
    class _Resp:
        def __init__(self) -> None:
            self.output_text = ""
            self.output = [
                {
                    "content": [
                        {"text": "SELECT 42;"},
                    ],
                }
            ]

    resp = _Resp()
    text = streamlit_app._openai_response_text(resp)
    assert text == "SELECT 42;"


def test_hf_error_triggers_openai_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_client = _DummyHFClient()

    # Ensure OpenAI settings return an API key and model name without touching real secrets/env.
    monkeypatch.setattr(
        streamlit_app,
        "_get_openai_settings",
        lambda: ("test-api-key", "gpt-5-nano"),
    )

    # Replace OpenAI client in the app module with our dummy implementation.
    monkeypatch.setattr(streamlit_app, "OpenAI", _DummyOpenAIClient)

    # Replace Streamlit module used inside the app with a minimal stub to avoid UI dependencies.
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(streamlit_app, "st", dummy_st)

    sql_text, user_prompt = streamlit_app._call_model(
        client=dummy_client,
        schema="CREATE TABLE test (id INT);",
        question="How many rows are in test?",
        temperature=0.1,
        max_tokens=128,
        timeout_s=45,
        adapter_id=None,
        use_endpoint=True,
    )

    assert sql_text == "SELECT 1;"
    assert "How many rows" in user_prompt
    assert dummy_st.caption_called is True
    assert dummy_st.error_called is False


def test_openai_fallback_uses_raw_text_when_extractor_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_client = _DummyHFClient()

    # Ensure OpenAI settings return an API key and model name without touching real secrets/env.
    monkeypatch.setattr(
        streamlit_app,
        "_get_openai_settings",
        lambda: ("test-api-key", "gpt-5-nano"),
    )

    class _DummyOpenAIResponseRaw:
        def __init__(self, text: str) -> None:
            self.output_text = text

    class _DummyOpenAIResponsesRaw:
        def __init__(self, text: str) -> None:
            self._text = text

        def create(
            self,
            model: str,
            input: str,
            max_output_tokens: int,
            **_: Any,
        ) -> _DummyOpenAIResponseRaw:  # noqa: ARG002
            return _DummyOpenAIResponseRaw(self._text)

    class _DummyOpenAIClientRaw:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.responses = _DummyOpenAIResponsesRaw("RAW TEXT OUTPUT")

    # Replace OpenAI client in the app module with our dummy implementation.
    monkeypatch.setattr(streamlit_app, "OpenAI", _DummyOpenAIClientRaw)

    # Force the SQL extractor to return an empty string to ensure we fall back to the raw text.
    monkeypatch.setattr(streamlit_app, "_extract_sql_from_text", lambda _raw: "")

    # Replace Streamlit module used inside the app with a minimal stub to avoid UI dependencies.
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(streamlit_app, "st", dummy_st)

    sql_text, _ = streamlit_app._call_model(
        client=dummy_client,
        schema="CREATE TABLE test (id INT);",
        question="How many rows are in test?",
        temperature=0.1,
        max_tokens=128,
        timeout_s=45,
        adapter_id=None,
        use_endpoint=True,
    )

    assert sql_text == "RAW TEXT OUTPUT"
    assert dummy_st.caption_called is True
    assert dummy_st.error_called is False