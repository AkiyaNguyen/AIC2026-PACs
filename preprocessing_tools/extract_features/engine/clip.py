"""CLIP ViT-B/32 image encoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


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
