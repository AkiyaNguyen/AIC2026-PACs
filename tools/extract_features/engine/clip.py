"""CLIP ViT-B/32 image encoder and --device resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


def parse_device_name(name: str) -> str:
    """Map CLI --device to a torch device type name (cpu | cuda)."""
    key = name.strip().lower()
    if key == "cpu":
        return "cpu"
    if key in ("gpu", "cuda"):
        return "cuda"
    raise SystemExit(f"Unknown --device {name!r}. Use cpu, gpu, or cuda.")


def resolve_device(name: str) -> torch.device:
    """Return a torch device. GPU is opt-in; missing CUDA is a hard error."""
    torch_name = parse_device_name(name)
    if torch_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested (--device gpu) but torch.cuda.is_available() is False"
        )
    return torch.device(torch_name)


@dataclass
class ClipEmbedder:
    name: str
    model: object
    preprocess: object
    device: torch.device

    @torch.no_grad()
    def embed_pils(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        batch = batch.to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.detach().cpu().numpy().astype(np.float32)


def load_clip(device: torch.device) -> ClipEmbedder:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    return ClipEmbedder("clip", model, preprocess, device)
