from pathlib import Path
import sys
from typing import Any


def _ensure_root_on_path() -> None:
    """Ensure that the project root is available on sys.path for imports."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_root_on_path()

from app import streamlit_app  # noqa: E402  # isort: skip


class _DummyWarmClient:
    def text_generation(self, **_: Any) -> str:
        return "ok"


class _DummyPausedClient:
    def text_generation(self, **_: Any) -> str:
        raise RuntimeError("The endpoint is paused, please resume it.")


class _DummyBrokenClient:
    def text_generation(self, **_: Any) -> str:
        raise RuntimeError("connection timed out")


def test_wake_up_endpoint_reports_warm_on_success() -> None:
    is_warm, message = streamlit_app._wake_up_endpoint(_DummyWarmClient())
    assert is_warm is True
    assert "warm" in message.lower()


def test_wake_up_endpoint_detects_paused_endpoint() -> None:
    is_warm, message = streamlit_app._wake_up_endpoint(_DummyPausedClient())
    assert is_warm is False
    assert "paused" in message.lower()


def test_wake_up_endpoint_reports_generic_failure() -> None:
    is_warm, message = streamlit_app._wake_up_endpoint(_DummyBrokenClient())
    assert is_warm is False
    assert "not reachable" in message.lower()
    assert "connection timed out" in message.lower()


def test_describe_backend_mode_variants() -> None:
    unconfigured = streamlit_app.HFConfig(
        hf_token="", endpoint_url="", model_id="", provider="auto", adapter_id=None
    )
    assert "not configured" in streamlit_app._describe_backend_mode(unconfigured).lower()

    endpoint_cfg = streamlit_app.HFConfig(
        hf_token="tok",
        endpoint_url="https://example.endpoint",
        model_id="",
        provider="auto",
        adapter_id="my-adapter",
    )
    assert "dedicated endpoint" in streamlit_app._describe_backend_mode(endpoint_cfg).lower()
    assert "my-adapter" in streamlit_app._describe_backend_mode(endpoint_cfg)

    router_cfg = streamlit_app.HFConfig(
        hf_token="tok",
        endpoint_url="",
        model_id="some/model",
        provider="auto",
        adapter_id=None,
    )
    assert "router model" in streamlit_app._describe_backend_mode(router_cfg).lower()
