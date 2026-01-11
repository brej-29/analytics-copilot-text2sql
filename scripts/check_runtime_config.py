from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Mapping, Tuple

"""
Runtime configuration checker for the Streamlit app.

This script inspects configuration for:

- Hugging Face Inference (primary provider)
- OpenAI fallback (secondary provider)

It looks for settings in:

- .streamlit/secrets.toml  (if present, takes precedence)
- Environment variables    (fallback when secrets are missing)

and reports:

- Which keys are configured (with masked values for sensitive fields)
- Whether at least one provider is usable

Exit codes:

- 0: At least one provider is configured (HF and/or OpenAI)
- 1: Neither HF nor OpenAI fallback is configured

The masking format for tokens/keys is similar to: `hf_****1234`
(i.e. keep a short prefix and the last 4 characters).
"""

try:
    import tomllib  # Python 3.11+
except Exception:  # noqa: BLE001
    tomllib = None


def _load_secrets_file(project_root: Path) -> Mapping[str, Any]:
    """
    Load .streamlit/secrets.toml if it exists and tomllib is available.

    Returns an empty dict when the file is missing or cannot be parsed.
    """
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if not secrets_path.is_file():
        return {}

    if tomllib is None:
        print(
            "WARNING: tomllib is not available; skipping .streamlit/secrets.toml "
            "parsing and relying on environment variables only.",
        )
        return {}

    try:
        with secrets_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: Failed to parse {secrets_path} as TOML: {exc}. "
            "Falling back to environment variables only.",
        )
        return {}

    return data or {}


def _get_from_mapping(mapping: Mapping[str, Any], key: str) -> str:
    """Return a trimmed string value from a generic mapping, or empty string."""
    try:
        value = mapping.get(key)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        value = None
    if value is None:
        return ""
    return str(value).strip()


def _mask_secret(value: str) -> str:
    """
    Return a masked representation of a secret value.

    Example:
        "hf_example_token_1234" -> "hf_****1234"
    """
    if not value:
        return "<missing>"
    trimmed = value.strip()
    if len(trimmed) <= 6:
        return "*" * len(trimmed)
    prefix = trimmed[:3]
    suffix = trimmed[-4:]
    return f"{prefix}****{suffix}"


def _resolve_value(
    key: str,
    secrets: Mapping[str, Any],
    environ: Mapping[str, str],
) -> Tuple[str, str]:
    """
    Resolve a value from secrets, then environment.

    Returns (value, source), where source is "secrets.toml", "env", or "".
    """
    from_secrets = _get_from_mapping(secrets, key)
    if from_secrets:
        return from_secrets, "secrets.toml"

    from_env = environ.get(key, "").strip()
    if from_env:
        return from_env, "env"

    return "", ""


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001
    project_root = Path(__file__).resolve().parents[1]
    secrets = _load_secrets_file(project_root)
    environ = os.environ

    print("=== Runtime configuration check ===")
    if secrets:
        print(f"Loaded .streamlit/secrets.toml from: {project_root / '.streamlit/secrets.toml'}")
    else:
        print("No usable .streamlit/secrets.toml found; using environment variables only.")
    print()

    # Resolve Hugging Face configuration.
    hf_token, hf_token_src = _resolve_value("HF_TOKEN", secrets, environ)

    endpoint_url, endpoint_src = _resolve_value("HF_ENDPOINT_URL", secrets, environ)
    if not endpoint_url:
        # Backwards-compatible alias for HF_ENDPOINT_URL.
        base_url, base_url_src = _resolve_value("HF_INFERENCE_BASE_URL", secrets, environ)
        endpoint_url = base_url
        endpoint_src = base_url_src

    model_id, model_src = _resolve_value("HF_MODEL_ID", secrets, environ)
    adapter_id, adapter_src = _resolve_value("HF_ADAPTER_ID", secrets, environ)

    # Resolve OpenAI fallback configuration.
    openai_key, openai_key_src = _resolve_value("OPENAI_API_KEY", secrets, environ)
    openai_model, openai_model_src = _resolve_value("OPENAI_FALLBACK_MODEL", secrets, environ)
    if not openai_model:
        openai_model = "gpt-5-nano"
        if not openai_key_src:
            # Use "default" to indicate the value comes from the app's default,
            # not from secrets or env.
            openai_model_src = "default"

    def _print_field(name: str, value: str, source: str, mask: bool = False) -> None:
        if not value:
            print(f"{name:30s}: MISSING")
            return
        display = _mask_secret(value) if mask else value
        origin = source or "-"
        print(f"{name:30s}: {display} (from {origin})")

    print("Hugging Face configuration:")
    _print_field("HF_TOKEN", hf_token, hf_token_src, mask=True)
    _print_field("HF_ENDPOINT_URL / HF_INFERENCE_BASE_URL", endpoint_url, endpoint_src)
    _print_field("HF_MODEL_ID", model_id, model_src)
    _print_field("HF_ADAPTER_ID", adapter_id, adapter_src)
    print()

    print("OpenAI fallback configuration:")
    _print_field("OPENAI_API_KEY", openai_key, openai_key_src, mask=True)
    _print_field("OPENAI_FALLBACK_MODEL", openai_model, openai_model_src)
    print()

    hf_configured = bool(hf_token and (endpoint_url or model_id))
    openai_configured = bool(openai_key)

    if not hf_configured and not openai_configured:
        print("ERROR: Neither Hugging Face nor OpenAI fallback is fully configured.")
        print("  - To use Hugging Face Inference, set HF_TOKEN and either:")
        print("      * HF_ENDPOINT_URL / HF_INFERENCE_BASE_URL, or")
        print("      * HF_MODEL_ID (for router/provider-based inference).")
        print("  - Or configure OPENAI_API_KEY (and optionally OPENAI_FALLBACK_MODEL)")
        print("    for the OpenAI fallback path.")
        return 1

    print("At least one provider is configured:")
    if hf_configured:
        print("  - Hugging Face Inference (primary)")
    if openai_configured:
        print("  - OpenAI fallback (secondary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))