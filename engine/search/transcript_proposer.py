"""Transcript candidate proposal: BM25 + semantic segment hits → keyframes."""

from __future__ import annotations

import numpy as np

from engine.search.config import SearchConfig
from engine.search.transcript.bm25_index import AsrBm25Index
from engine.search.transcript.semantic_index import AsrSemanticIndex
from engine.search.types import TranscriptProposals


class TranscriptKeyframeProposer:
    def __init__(
        self,
        bm25: AsrBm25Index,
        semantic: AsrSemanticIndex,
        timeline: dict[str, list[tuple[int, float]]],
    ) -> None:
        self._bm25 = bm25
        self._semantic = semantic
        self._timeline = timeline

    def propose(
        self,
        query: str,
        q_asr: np.ndarray,
        cfg: SearchConfig,
    ) -> TranscriptProposals:
        bm25 = self._bm25.keyframe_candidates(
            query,
            cfg.bm25_top_segments,
            self._timeline,
            min_gap=cfg.segment_min_gap,
            pad=cfg.segment_pad,
        )
        semantic = self._semantic.keyframe_candidates(
            q_asr,
            cfg.sem_top_segments,
            self._timeline,
            min_gap=cfg.segment_min_gap,
            pad=cfg.segment_pad,
        )
        return TranscriptProposals(bm25=bm25, semantic=semantic)
