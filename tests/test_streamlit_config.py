from pathlib import Path
import sys

from typing import Dict, Any


def _ensure_root_on_path() -> None:
    """Ensure that the project root is available on sys.path for imports."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_root_on_path()

from app import streamlit_app  # noqa: E402  # isort: skip


def test_resolve_hf_config_prefers_secrets_over_env() -> None:
    secrets: Dict[str, Any] = {
        "HF_TOKEN": "secret-token",
        "HF_ENDPOINT_URL": "https://endpoint-from-secrets",
        "HF_ADAPTER_ID": "adapter-from-secrets",
        "HF_MODEL_ID": "model-from-secrets",
        "HF_PROVIDER": "hf-inference",
    }
    environ = {
        "HF_TOKEN": "env-token",
        "HF_ENDPOINT_URL": "https://endpoint-from-env",
        "HF_ADAPTER_ID": "adapter-from-env",
        "HF_MODEL_ID": "model-from-env",
        "HF_PROVIDER": "env-provider",
    }

    cfg = streamlit_app._resolve_hf_config(secrets=secrets, environ=environ)

    assert cfg.hf_token == "secret-token"
    assert cfg.endpoint_url == "https://endpoint-from-secrets"
    assert cfg.adapter_id == "adapter-from-secrets"
    assert cfg.model_id == "model-from-secrets"
    assert cfg.provider == "hf-inference"


def test_resolve_hf_config_falls_back_to_env_when_secrets_missing() -> None:
    secrets: Dict[str, Any] = {}
    environ = {
        "HF_TOKEN": "env-token",
        "HF_ENDPOINT_URL": "https://endpoint-from-env",
        "HF_ADAPTER_ID": "adapter-from-env",
        "HF_MODEL_ID": "model-from-env",
        "HF_PROVIDER": "env-provider",
    }

    cfg = streamlit_app._resolve_hf_config(secrets=secrets, environ=environ)

    assert cfg.hf_token == "env-token"
    assert cfg.endpoint_url == "https://endpoint-from-env"
    assert cfg.adapter_id == "adapter-from-env"
    assert cfg.model_id == "model-from-env"
    assert cfg.provider == "env-provider"


def test_resolve_hf_config_router_mode_without_endpoint() -> None:
    secrets: Dict[str, Any] = {
        "HF_TOKEN": "secret-token",
        "HF_MODEL_ID": "model-from-secrets",
    }
    environ: Dict[str, str] = {}

    cfg = streamlit_app._resolve_hf_config(secrets=secrets, environ=environ)

    assert cfg.hf_token == "secret-token"
    assert cfg.endpoint_url == ""
    assert cfg.model_id == "model-from-secrets"
    # Default provider should fall back to "auto"
    assert cfg.provider == "auto"
    # No adapter id when not configured
    assert cfg.adapter_id is None