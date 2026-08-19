"""Search hyperparameters for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass

RRF_K = 60


@dataclass(frozen=True)
class SearchConfig:
    num_candidates_vis: int = 500
    bm25_top_segments: int = 50
    sem_top_segments: int = 50
    segment_min_gap: float = 1.0
    segment_pad: float = 0.5
    delta: float = 1.0
    num_results: int = 100
    rrf_k: int = RRF_K
