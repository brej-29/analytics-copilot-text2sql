from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from text2sql.training.formatting import ensure_sql_only

logger = logging.getLogger(__name__)

_MODEL: Optional[PreTrainedModel] = None
_TOKENIZER: Optional[PreTrainedTokenizerBase] = None
_DEVICE: Optional[str] = None


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device '{device}'. Use 'auto', 'cpu', or 'cuda'.")
    return device


def load_model_for_inference(
    base_model: str,
    adapter_dir: Optional[str] = None,
    device: str = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """
    Load a model and tokenizer for text-to-SQL inference.

    Parameters
    ----------
    base_model : str
        Hugging Face model identifier or local path to a (possibly merged)
        Causal LM.
    adapter_dir : Optional[str]
        Optional path to a directory containing LoRA adapters produced by
        `scripts/train_qlora.py`. When provided, adapters are loaded on top of
        the base model. When omitted, the base model is used as-is.
    device : str
        "auto", "cpu", or "cuda". "auto" selects "cuda" when a GPU is
        available, otherwise "cpu".

    Returns
    -------
    (model, tokenizer, device_str)
        The loaded model and tokenizer, and the resolved device string.
    """
    global _MODEL, _TOKENIZER, _DEVICE

    resolved_device = _resolve_device(device)
    _DEVICE = resolved_device

    model_path = base_model
    adapter_path: Optional[Path] = None
    if adapter_dir:
        adapter_path = Path(adapter_dir).expanduser().resolve()
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    logger.info("Loading tokenizer from %s", adapter_path or model_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {}
    if resolved_device == "cuda":
        # Prefer 4-bit loading when bitsandbytes is available; fall back to
        # standard half precision otherwise.
        try:
            import bitsandbytes  # noqa: F401  # type: ignore[import]

            model_kwargs["load_in_4bit"] = True
            model_kwargs["torch_dtype"] = torch.float16
            logger.info("Loading base model '%s' in 4-bit on CUDA.", model_path)
        except Exception:  # noqa: BLE001
            logger.info(
                "bitsandbytes not available; loading base model '%s' in float16 on CUDA.",
                model_path,
            )
            model_kwargs["torch_dtype"] = torch.float16
    else:
        logger.info("Loading base model '%s' on CPU (float32).", model_path)
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if resolved_device == "cuda" else None,
        **model_kwargs,
    )

    if adapter_path is not None:
        try:
            from peft import PeftModel  # type: ignore[import]

        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import `peft` while an adapter_dir was provided. "
                "Ensure `peft` is installed."
            ) from exc

        logger.info("Loading LoRA adapters from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))

    if resolved_device == "cpu":
        model.to("cpu")

    model.eval()

    _MODEL = model
    _TOKENIZER = tokenizer

    logger.info("Model and tokenizer loaded for inference on device='%s'.", resolved_device)
    return model, tokenizer, resolved_device


def generate_sql(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> str:
    """
    Generate a SQL string from a text-to-SQL prompt.

    This function assumes that `load_model_for_inference` has been called
    beforehand to initialize the global model/tokenizer.

    Parameters
    ----------
    prompt : str
        Full prompt text (instruction + input/schema/question).
    max_new_tokens : int
        Maximum number of new tokens to generate.
    temperature : float
        Sampling temperature. When 0.0, greedy decoding is used.
    top_p : float
        Nucleus sampling parameter used when temperature > 0.

    Returns
    -------
    str
        A cleaned SQL string, with surrounding markdown fences removed and
        whitespace normalized.
    """
    if _MODEL is None or _TOKENIZER is None or _DEVICE is None:
        raise RuntimeError(
            "Model not initialized. Call `load_model_for_inference` first."
        )

    tokenizer = _TOKENIZER
    model = _MODEL
    device = _DEVICE

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    do_sample = temperature > 0.0

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens.
    generated_ids = output_ids[0, input_ids.shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return ensure_sql_only(raw_text)