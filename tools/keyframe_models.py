#!/usr/bin/env python3
"""Image embedders for NII-UIT-style keyframe selection.

Supported backends (CLI names):
  - clip   — OpenCLIP ViT-B-32 (same family as BTC / kis_search)
  - siglip — OpenCLIP ViT-B-16-SigLIP-256
  - beit3  — timm beit3_base_patch16_224.in22k_ft_in1k (vision tower)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from PIL import Image


MODEL_CHOICES = ("clip", "siglip", "beit3")


class ImageEmbedder(Protocol):
    name: str

    def embed_pils(self, images: list[Image.Image]) -> np.ndarray:
        """Return L2-normalized float32 vectors, shape (N, D)."""


@dataclass
class OpenClipEmbedder:
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


@dataclass
class TimmEmbedder:
    name: str
    model: object
    transform: object
    device: torch.device

    @torch.no_grad()
    def embed_pils(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images])
        batch = batch.to(self.device)
        feats = self.model(batch)
        if feats.ndim > 2:
            feats = feats.mean(dim=1)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.detach().cpu().numpy().astype(np.float32)


def load_embedder(model_name: str, device: torch.device) -> ImageEmbedder:
    key = model_name.strip().lower()
    if key not in MODEL_CHOICES:
        raise SystemExit(f"Unknown model {model_name!r}; choose from {MODEL_CHOICES}")

    if key == "clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = model.to(device).eval()
        return OpenClipEmbedder("clip", model, preprocess, device)

    if key == "siglip":
        import open_clip

        # Prefer SigLIP; fall back to a larger CLIP if this weight is unavailable.
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16-SigLIP-256", pretrained="webli"
            )
        except Exception as exc:  # noqa: BLE001 — surface clear fallback reason
            raise SystemExit(
                "Failed to load SigLIP ViT-B-16-SigLIP-256 (webli). "
                f"Underlying error: {exc}"
            ) from exc
        model = model.to(device).eval()
        return OpenClipEmbedder("siglip", model, preprocess, device)

    # beit3 via timm vision-only checkpoint
    import timm
    from timm.data import create_transform, resolve_model_data_config

    model = timm.create_model(
        "beit3_base_patch16_224.in22k_ft_in1k",
        pretrained=True,
        num_classes=0,
    )
    model = model.to(device).eval()
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    return TimmEmbedder("beit3", model, transform, device)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity for L2-normalized vectors."""
    return float(1.0 - np.dot(a, b))
