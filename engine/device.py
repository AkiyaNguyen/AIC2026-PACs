from __future__ import annotations

import torch


def get_proper_device(name: str) -> torch.device:
    """Map cpu|gpu|cuda. GPU is opt-in: CUDA, else MPS, else error."""
    key = name.strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError(
            f"GPU requested ({name!r}) but neither CUDA nor MPS is available"
        )
    raise ValueError(f"Unknown device {name!r}; use cpu, gpu")
