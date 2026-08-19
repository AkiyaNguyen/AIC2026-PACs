"""Fuse visual and transcript scores; produce ranked submission hits."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from engine.search.scoring import combine_text_scores, minmax_norm
from engine.search.types import QueryFeatures


class KisFusionRanker:
    def rank(
        self,
        feats: QueryFeatures,
        row_index_map_info: dict[str, Any],
        *,
        weight_visual: float,
        weight_transcript: float,
        num_results: int,
    ) -> list[dict]:
        t0 = time.perf_counter()
        r_text = combine_text_scores(feats.r_sem, feats.r_bm25)
        vis_norm = minmax_norm(feats.s_vis)
        wsum = weight_visual + weight_transcript
        if wsum < 1e-12:
            total = np.zeros(len(feats.all_rows), dtype=np.float32)
        else:
            total = (weight_visual / wsum) * vis_norm + (weight_transcript / wsum) * r_text

        order = np.argsort(-total)[:num_results]
        hits: list[dict] = []
        for rank, i in enumerate(order, 1):
            row = int(feats.all_rows[i])
            src = "+".join(sorted(feats.source.get(row, {"vis"})))
            hits.append(
                {
                    "rank": rank,
                    "score": float(total[i]),
                    "clip_row": row,
                    "video_id": row_index_map_info["video_ids"][row],
                    "pts_time": float(row_index_map_info["pts_list"][row]),
                    "row_idx_in_video": int(
                        row_index_map_info["row_to_idx_in_each_video"][row]
                    ),
                    "frame_idx": int(row_index_map_info["frame_idx_list"][row]),
                    "source": src,
                }
            )
        feats.timings_ms["rerank"] = (time.perf_counter() - t0) * 1000
        return hits
