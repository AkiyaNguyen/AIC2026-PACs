"""Visual candidate retrieval: CLIP + SigLIP2 ANN fused by RRF."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engine.search.config import SearchConfig
from engine.search.scoring import rrf_union

if TYPE_CHECKING:
    from engine.search_service import SearchService


class VisualKeyframeRetriever:
    def __init__(self, service: SearchService) -> None:
        self._service = service

    def retrieve(
        self,
        query_vi: str,
        query_en: str,
        cfg: SearchConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """CLIP uses English; SigLIP2 uses Vietnamese."""
        k = min(cfg.num_candidates_vis, self._service.clip_index.ntotal)
        clip_ids = self._service.ann(
            self._service.clip_encoder, self._service.clip_index, query_en, k
        ) # embed english query for clip retrieval
        sig_ids = self._service.ann(
            self._service.siglip_encoder, self._service.siglip_index, query_vi, k
        ) # vietnamese query for siglip2 retrieval
        return rrf_union(clip_ids, sig_ids, k=cfg.rrf_k)
