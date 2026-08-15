"""Shared CLI helpers."""

from __future__ import annotations

import torch


def get_proper_device(name: str) -> torch.device:
    """Map --device cpu|gpu|cuda. GPU is opt-in: CUDA, else MPS, else error."""
    key = name.strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    elif key == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            raise ValueError(f"GPU requested (--device {name!r}) but neither CUDA nor MPS is available")
