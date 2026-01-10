from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class TrainingConfig:
    """
    Configuration for QLoRA fine-tuning of Mistral-7B-Instruct.

    This dataclass is deliberately minimal and focused on the hyperparameters
    that are most likely to be tuned. It can be extended later as needed.
    """

    base_model: str
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)