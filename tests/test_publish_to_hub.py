from pathlib import Path
import json
import sys
from unittest import mock

import pytest


def _ensure_root_on_path() -> None:
    """Ensure that the project root is available on sys.path for script imports."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_root_on_path()

from scripts import publish_to_hub  # noqa: E402  # isort: skip


def _make_minimal_adapter_dir(tmp_path: Path) -> Path:
    """Create a minimal fake adapter directory with required files."""
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    config = {"base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.1"}
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    # Touch a fake weights file; content does not matter for validation.
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

    return adapter_dir


def test_publish_to_hub_skip_readme_does_not_create_readme(tmp_path: Path) -> None:
    """--skip_readme should avoid README generation and still call upload_folder."""
    adapter_dir = _make_minimal_adapter_dir(tmp_path)

    with mock.patch.object(publish_to_hub, "HfApi") as mock_hfapi_cls:
        api_instance = mock_hfapi_cls.return_value
        api_instance.whoami.return_value = {"name": "tester"}
        api_instance.create_repo.return_value = None
        api_instance.upload_folder.return_value = None

        rc = publish_to_hub.main(
            [
                "--repo_id",
                "user/test-adapter",
                "--adapter_dir",
                str(adapter_dir),
                "--skip_readme",
            ]
        )

    assert rc == 0
    # README.md should not be auto-created when --skip_readme is used.
    assert not (adapter_dir / "README.md").exists()
    api_instance.upload_folder.assert_called_once()


def test_publish_to_hub_creates_readme_when_missing(tmp_path: Path) -> None:
    """When README.md is missing, the script should create a minimal README."""
    adapter_dir = _make_minimal_adapter_dir(tmp_path)
    readme_path = adapter_dir / "README.md"
    assert not readme_path.exists()

    with mock.patch.object(publish_to_hub, "HfApi") as mock_hfapi_cls:
        api_instance = mock_hfapi_cls.return_value
        api_instance.whoami.return_value = {"name": "tester"}
        api_instance.create_repo.return_value = None
        api_instance.upload_folder.return_value = None

        rc = publish_to_hub.main(
            [
                "--repo_id",
                "user/test-adapter",
                "--adapter_dir",
                str(adapter_dir),
            ]
        )

    assert rc == 0
    assert readme_path.is_file()
    content = readme_path.read_text(encoding="utf-8")
    # Basic sanity checks on README contents.
    assert "Text-to-SQL" in content
    assert "Deployment with Hugging Face Inference Endpoints" in content
    assert "LORA_ADAPTERS" in content


def test_validate_adapter_dir_missing_config_raises(tmp_path: Path) -> None:
    """Adapter validation should fail fast if adapter_config.json is missing."""
    adapter_dir = tmp_path / "adapters_missing_config"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # Only weights present.
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

    with pytest.raises(RuntimeError) as excinfo:
        publish_to_hub._validate_adapter_dir(adapter_dir)  # type: ignore[attr-defined]

    assert "adapter_config.json" in str(excinfo.value)


def test_validate_adapter_dir_missing_weights_raises(tmp_path: Path) -> None:
    """Adapter validation should fail fast if no adapter weights are present."""
    adapter_dir = tmp_path / "adapters_missing_weights"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config = {"base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.1"}
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as excinfo:
        publish_to_hub._validate_adapter_dir(adapter_dir)  # type: ignore[attr-defined]

    msg = str(excinfo.value)
    assert "adapter_model.safetensors" in msg or "adapter_model.bin" in msg