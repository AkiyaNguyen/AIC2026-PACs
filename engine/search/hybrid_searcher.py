"""Orchestrator: prepare (heavy) + rerank (light) hybrid keyframe search."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from engine.config import get_features_root
from engine.search.config import SearchConfig
from engine.search.fusion_ranker import KisFusionRanker
from engine.search.pool_builder import CandidatePoolBuilder
from engine.search.temporal import build_video_timeline
from engine.search.transcript.bm25_index import AsrBm25Index
from engine.search.transcript.semantic_index import AsrSemanticIndex
from engine.search.transcript_proposer import TranscriptKeyframeProposer
from engine.search.transcript_scorer import KeyframeTranscriptScorer
from engine.search.types import QueryFeatures
from engine.search.visual_retriever import VisualKeyframeRetriever

if TYPE_CHECKING:
    from engine.search_service import SearchService


class HybridSearcher:
    """Wire visual + transcript channels; expose prepare/rerank split."""

    def __init__(
        self,
        service: SearchService,
        bm25: AsrBm25Index,
        semantic: AsrSemanticIndex,
        cfg: SearchConfig | None = None,
    ) -> None:
        self.service = service
        self.cfg = cfg or SearchConfig()
        info = service.row_index_map_info
        self._info = info
        timeline = build_video_timeline(info["video_ids"], info["pts_list"])

        self._visual = VisualKeyframeRetriever(service)
        self._transcript_proposer = TranscriptKeyframeProposer(
            bm25, semantic, timeline
        )
        self._pool_builder = CandidatePoolBuilder()
        self._transcript_scorer = KeyframeTranscriptScorer(service, bm25)
        self._fusion = KisFusionRanker()

    @classmethod
    def from_service(
        cls,
        service: SearchService,
        *,
        asr_root: Path | None = None,
        cfg: SearchConfig | None = None,
    ) -> HybridSearcher:
        root = asr_root or (get_features_root() / "asr")
        bm25 = AsrBm25Index(root, service.asr_by_video)
        semantic = AsrSemanticIndex(service.asr_embed_mat, service.asr_by_video)
        return cls(service, bm25, semantic, cfg)

    def prepare(
        self,
        query_vi: str,
        query_en: str,
        cfg: SearchConfig | None = None,
    ) -> QueryFeatures:
        cfg = cfg or self.cfg
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        vis_rows, s_vis = self._visual.retrieve(query_vi, query_en, cfg)
        timings["visual_ann_rrf"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        q_asr = (
            self.service.asr_encoder.encode_query(query_vi)
            .reshape(-1)
            .astype(np.float32)
        )
        proposals = self._transcript_proposer.propose(query_vi, q_asr, cfg)
        timings["text_candidates"] = (time.perf_counter() - t0) * 1000

        pool = self._pool_builder.build(vis_rows, s_vis, proposals)

        t0 = time.perf_counter()
        r_sem, r_bm25 = self._transcript_scorer.score(
            pool.all_rows, query_vi, q_asr, cfg
        )
        timings["scoring"] = (time.perf_counter() - t0) * 1000

        timings["pool_vis"] = float(pool.n_vis)
        timings["pool_text"] = float(pool.n_text)
        timings["pool_total"] = float(len(pool.all_rows))

        return QueryFeatures(
            all_rows=pool.all_rows,
            s_vis=pool.s_vis,
            r_sem=r_sem,
            r_bm25=r_bm25,
            source=pool.source,
            timings_ms=timings,
        )

    def rerank(
        self,
        feats: QueryFeatures,
        *,
        weight_visual: float = 0.8,
        weight_transcript: float = 0.2,
        weight_sem_text: float = 0.6,
        weight_bm25: float = 0.4,
        num_results: int | None = None,
    ) -> list[dict]:
        return self._fusion.rank(
            feats,
            self._info,
            weight_visual=weight_visual,
            weight_transcript=weight_transcript,
            weight_sem_text=weight_sem_text,
            weight_bm25=weight_bm25,
            num_results=num_results or self.cfg.num_results,
        )

    def search(
        self,
        query_vi: str,
        query_en: str,
        *,
        weight_visual: float = 0.8,
        weight_transcript: float = 0.2,
        weight_sem_text: float = 0.6,
        weight_bm25: float = 0.4,
        cfg: SearchConfig | None = None,
    ) -> list[dict]:
        cfg = cfg or self.cfg
        feats = self.prepare(query_vi, query_en, cfg=cfg)
        return self.rerank(
            feats,
            weight_visual=weight_visual,
            weight_transcript=weight_transcript,
            weight_sem_text=weight_sem_text,
            weight_bm25=weight_bm25,
            num_results=cfg.num_results,
        )
