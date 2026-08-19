"""Query-time text encoders (CLIP / SigLIP2 / MiniLM towers)."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_DIM = 512
SIGLIP_MODEL_ID = "google/siglip2-so400m-patch16-256"
SIGLIP_DIM = 1152
SIGLIP_TEXT_MAX_LENGTH = 64
ASR_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ASR_DIM = 384


def _as_embed_tensor(out) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "text_embeds", "image_embeds"):
        pooled = getattr(out, attr, None)
        if pooled is not None:
            return pooled
    raise TypeError(f"Unexpected text features type: {type(out)}")


def hf_query_features(
    model, device: torch.device, inputs, normalize: bool
) -> np.ndarray:
    tensor_inputs = {
        k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)
    }
    with torch.no_grad():
        feat = _as_embed_tensor(model.get_text_features(**tensor_inputs))
        if normalize:
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feat.cpu().numpy().astype(np.float32)


class ClipTextQueryEncoder:
    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.normalize_embeddings = normalize_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        inputs = self.processor(
            text=[query], return_tensors="pt", padding=True, truncation=True
        )
        return hf_query_features(
            self.model, self.device, inputs, self.normalize_embeddings
        )


class Siglip2TextQueryEncoder:
    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.normalize_embeddings = normalize_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=SIGLIP_TEXT_MAX_LENGTH,
        )
        return hf_query_features(
            self.model, self.device, inputs, self.normalize_embeddings
        )


class ASRQueryEncoder:
    def __init__(
        self,
        model: SentenceTransformer,
        device: torch.device,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.normalize_embeddings = normalize_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        feat = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(feat, dtype=np.float32)
