"""Score every keyframe in the pool against nearby ASR segments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engine.search.config import SearchConfig
from engine.search.scoring import max_asr_scores, max_bm25_scores
from engine.search.temporal import asr_rows_for_clip_rows
from engine.search.transcript.bm25_index import AsrBm25Index

if TYPE_CHECKING:
    from engine.search_service import SearchService


class KeyframeTranscriptScorer:
    def __init__(self, service: SearchService, bm25: AsrBm25Index) -> None:
        self._service = service
        self._bm25 = bm25
        self._info = service.row_index_map_info

    def score(
        self,
        all_rows: np.ndarray,
        query: str,
        q_asr: np.ndarray,
        cfg: SearchConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        row_lists = asr_rows_for_clip_rows(
            all_rows,
            self._info["video_ids"],
            self._info["pts_list"],
            self._service.asr_by_video,
            delta=cfg.delta,
        )
        r_sem = max_asr_scores(q_asr, self._service.asr_embed_mat, row_lists)
        bm25_doc_scores = self._bm25.score_query(query)
        r_bm25 = max_bm25_scores(
            bm25_doc_scores, self._bm25.global_rows, row_lists
        )
        return r_sem, r_bm25
