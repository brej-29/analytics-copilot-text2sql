from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_MODEL: Optional[PreTrainedModel] = None
_TOKENIZER: Optional[PreTrainedTokenizerBase] = None
_DEVICE: Optional[str] = None


def _select_device(device: str) -> str:
    """
    Resolve the requested device string to an actual device ("cpu" or "cuda").

    If CUDA is requested but not available, this falls back to CPU and logs a
    warning since generation will be significantly slower.
    """
    device = (device or "auto").lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{device}'. Expected one of: auto, cpu, cuda.")

    if device == "cpu":
        return "cpu"

    if device in {"auto", "cuda"}:
        if torch.cuda.is_available():
            return "cuda"
        logger.warning(
            "CUDA was requested or auto-selected, but no GPU is available. "
            "Falling back to CPU; generation will be slow."
        )
        return "cpu"

    # Fallback; should not be reached.
    return "cpu"


def load_model_for_inference(
    base_model: str,
    adapter_dir: Optional[str] = None,
    device: str = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a base model and optional LoRA adapters for text-to-SQL inference.

    This function supports two main modes:
    1) Base HF model + LoRA adapters (adapter_dir points to the trained QLoRA
       adapters from the training pipeline).
    2) A locally merged model directory passed as `base_model` with
       `adapter_dir` left as None.

    Parameters
    ----------
    base_model : str
        Hugging Face model id or local path to the base (or merged) model.
    adapter_dir : Optional[str], optional
        Path to a directory containing LoRA adapters (PEFT), by default None.
    device : str, optional
        "auto", "cuda", or "cpu". If "auto", prefer CUDA when available,
        otherwise fall back to CPU.

    Returns
    -------
    (model, tokenizer)
        Loaded model and tokenizer ready for inference.
    """
    global _MODEL, _TOKENIZER, _DEVICE

    resolved_device = _select_device(device)
    logger.info(
        "Loading model for inference: base_model=%s, adapter_dir=%s, device=%s",
        base_model,
        adapter_dir,
        resolved_device,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model)
    if tokenizer.pad_token_id is None:
        # Many causal LM tokenizers do not have an explicit pad token; use EOS.
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use float16 on GPU and float32 on CPU for simplicity.
    torch_dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)

    if adapter_dir:
        adapter_path = Path(adapter_dir)
        if not adapter_path.is_dir():
            raise FileNotFoundError(
                f"Adapter directory not found: {adapter_dir}. "
                "Ensure you pass the correct path to the trained LoRA adapters."
            )
        logger.info("Loading LoRA adapters from %s", adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.to(resolved_device)
    model.eval()

    _MODEL = model
    _TOKENIZER = tokenizer
    _DEVICE = resolved_device

    return model, tokenizer


def generate_sql(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """
    Generate a SQL string from a prompt using the loaded model.

    `load_model_for_inference` must be called once before this function is
    used; otherwise a RuntimeError will be raised.
    """
    if _MODEL is None or _TOKENIZER is None or _DEVICE is None:
        raise RuntimeError(
            "Model is not loaded. Call load_model_for_inference(...) before generate_sql(...)."
        )

    model = _MODEL
    tokenizer = _TOKENIZER

    do_sample = temperature > 0.0

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(_DEVICE)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(_DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()