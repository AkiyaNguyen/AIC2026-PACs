"""Union visual and text keyframe candidates into one ranked pool."""

from __future__ import annotations

import numpy as np

from engine.search.types import CandidatePool, TranscriptProposals


class CandidatePoolBuilder:
    """Merge visual RRF rows with text-only rows; tag provenance."""

    def build(
        self,
        vis_rows: np.ndarray, # list of keyframe propose by visual retriever
        s_vis: np.ndarray, # score of keyframes p proposed by visual
        proposals: TranscriptProposals,
    ) -> CandidatePool:
        vis_list = np.asarray(vis_rows, dtype=np.int64).reshape(-1).tolist()
        vis_set = set(vis_list)
        text_rows = sorted(
            (set(proposals.bm25) | set(proposals.semantic)) - vis_set
        )
        all_rows = np.asarray(vis_list + text_rows, dtype=np.int64)

        source: dict[int, set[str]] = {}
        for row in vis_list:
            source.setdefault(row, set()).add("vis")
        for row in proposals.bm25:
            if row not in vis_set:
                source.setdefault(row, set()).add("bm25")
        for row in proposals.semantic:
            if row not in vis_set:
                source.setdefault(row, set()).add("sem")

        s_vis_full = np.zeros(len(all_rows), dtype=np.float32)
        s_vis_full[: len(vis_list)] = np.asarray(s_vis, dtype=np.float32).reshape(-1) # the rest is out of top k of visual, then set score to 0

        return CandidatePool(
            all_rows=all_rows,
            s_vis=s_vis_full,
            source=source,
            n_vis=len(vis_list),
            n_text=len(text_rows),
        )
