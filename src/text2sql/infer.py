from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)

_MODEL: Optional[torch.nn.Module] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_DEVICE: Optional[str] = None


def _resolve_device(device: str = "auto") -> str:
    """
    Resolve the effective device string given a user-specified preference.

    - \"auto\": use CUDA if available, otherwise CPU.
    - \"cuda\": require CUDA, but fall back to CPU with a warning if unavailable.
    - \"cpu\": force CPU inference.
    """
    device = (device or "auto").lower()
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"device must be 'auto', 'cuda', or 'cpu', got: {device!r}")

    if device == "cpu":
        return "cpu"

    has_cuda = torch.cuda.is_available()
    if device == "cuda":
        if not has_cuda:
            logger.warning(
                "CUDA was requested but is not available. Falling back to CPU inference."
            )
            return "cpu"
        return "cuda"

    # auto
    return "cuda" if has_cuda else "cpu"


def load_model_for_inference(
    base_model: str,
    adapter_dir: Optional[str] = None,
    device: str = "auto",
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a base model and optional LoRA adapters for text-to-SQL inference.

    Parameters
    ----------
    base_model:
        Base model name or local path (e.g., \"mistralai/Mistral-7B-Instruct-v0.1\").
    adapter_dir:
        Optional path to a directory containing LoRA adapters produced by
        the QLoRA training pipeline. If ``None``, the base model is used as-is.
    device:
        \"auto\" (default), \"cuda\", or \"cpu\".

    Returns
    -------
    (model, tokenizer)
        The loaded model and tokenizer. The model is placed on the resolved
        device and ready for generation.
    """
    global _MODEL, _TOKENIZER, _DEVICE

    resolved_device = _resolve_device(device)
    logger.info(
        "Loading model for inference: base_model=%s, adapter_dir=%s, device=%s",
        base_model,
        adapter_dir,
        resolved_device,
    )

    if resolved_device == "cuda":
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model.to("cpu")

    tokenizer_source = adapter_dir if adapter_dir is not None else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if adapter_dir:
        logger.info("Loading LoRA adapters from %s", adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

    if resolved_device == "cpu":
        model.to("cpu")

    _MODEL = model
    _TOKENIZER = tokenizer
    _DEVICE = resolved_device

    logger.info("Model and tokenizer loaded successfully for inference on %s.", resolved_device)
    return model, tokenizer


def generate_sql(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """
    Generate a SQL query from an instruction-style prompt.

    This function assumes :func:`load_model_for_inference` has been called
    beforehand to initialize the global model and tokenizer.
    """
    from text2sql.training.formatting import ensure_sql_only  # Local import to avoid cycles.

    if _MODEL is None or _TOKENIZER is None:
        raise RuntimeError("Model not loaded. Call load_model_for_inference(...) first.")

    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}.")

    device = _DEVICE or "cpu"
    model = _MODEL
    tokenizer = _TOKENIZER

    model.eval()

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    do_sample = temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    with torch.no_grad():
        outputs = model.generate(**encoded, **gen_kwargs)

    # Slice off the prompt tokens so we only keep the generated continuation.
    generated_tokens = outputs[0, encoded["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return ensure_sql_only(raw_text)