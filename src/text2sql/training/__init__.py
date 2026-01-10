"""
Training utilities for the Analytics Copilot (Text-to-SQL) project.

This subpackage contains:
- Prompt formatting helpers for instruction-tuning.
- Configuration structures for QLoRA / LoRA fine-tuning.
- (Later) training orchestration utilities.

It is intentionally lightweight so it can be imported from both scripts and
notebooks without pulling in heavy training dependencies.
"""

__all__ = [
    "VERSION",
]

VERSION = "0.0.1"