from pathlib import Path
import sys
from unittest import mock

import pytest


def _ensure_src_on_path() -> None:
    """Ensure that the 'src' directory is available on sys.path for imports."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()


def test_load_model_for_inference_4bit_uses_quantization_config() -> None:
    """
    Ensure that load_model_for_inference can be called in 4-bit mode without
    actually downloading a model, and that it wires BitsAndBytesConfig through
    to AutoModelForCausalLM.from_pretrained.

    If the local environment cannot import the necessary transformer stack
    (e.g. due to version constraints), this test is skipped instead of failing.
    """
    try:
        import text2sql.infer as infer  # isort: skip
    except ImportError as exc:
        pytest.skip(f"Skipping 4-bit quantization test due to import error: {exc}")

    with mock.patch.object(infer, "AutoTokenizer") as mock_tok_cls, \
        mock.patch.object(infer, "AutoModelForCausalLM") as mock_model_cls, \
        mock.patch.object(infer, "PeftModel") as mock_peft_cls, \
        mock.patch.object(infer.torch.cuda, "is_available", return_value=True):

        # Mock tokenizer to avoid any real downloads.
        tok = mock.Mock()
        tok.eos_token_id = 0
        tok.pad_token_id = None
        mock_tok_cls.from_pretrained.return_value = tok

        # Mock model loading.
        model_instance = mock.Mock()
        mock_model_cls.from_pretrained.return_value = model_instance

        # Make PEFT a no-op that simply returns the base model.
        mock_peft_cls.from_pretrained.side_effect = lambda base_model, adapter_dir: base_model

        model, tokenizer = infer.load_model_for_inference(
            base_model="dummy-model",
            adapter_dir=None,
            device="cuda",
            load_in_4bit=True,
            bnb_compute_dtype="float16",
            dtype="float16",
        )

        # Ensure we returned the mocked objects.
        assert model is model_instance
        assert tokenizer is tok

        # Check that quantization configuration and device_map were passed.
        mock_model_cls.from_pretrained.assert_called_once()
        _, kwargs = mock_model_cls.from_pretrained.call_args
        assert "quantization_config" in kwargs
        quant_config = kwargs["quantization_config"]
        assert getattr(quant_config, "load_in_4bit", False) is True
        assert kwargs.get("device_map") == "auto"
        assert kwargs.get("low_cpu_mem_usage") is True