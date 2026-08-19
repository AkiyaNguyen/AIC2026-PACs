"""MiniLM cosine retrieval over ASR segment embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.search.temporal import global_row_to_segment, keyframes_for_segment


@dataclass(frozen=True)
class AsrSemanticSegment:
    video_id: str
    start: float
    end: float
    global_row: int
    semantic_score: float


class AsrSemanticIndex:
    """Brute-force top-K over asr_embed_mat, rows aligned with asr_by_video."""

    def __init__(
        self,
        asr_embed_mat: np.ndarray,
        asr_by_video: dict[str, tuple[int, np.ndarray, np.ndarray]],
    ) -> None:
        mat = np.asarray(asr_embed_mat, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"asr_embed_mat must be 2-D, got shape {mat.shape}")

        expected = sum(len(starts) for _, starts, _ in asr_by_video.values())
        if mat.shape[0] != expected:
            raise ValueError(
                f"asr_embed_mat has {mat.shape[0]} rows, "
                f"asr_by_video expects {expected} segments"
            )

        self._asr_mat = mat
        self._asr_by_video = asr_by_video

    @property
    def n_segments(self) -> int:
        return self._asr_mat.shape[0]

    def segment_similarities(self, q_asr: np.ndarray) -> np.ndarray:
        q = np.asarray(q_asr, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._asr_mat.shape[1]:
            raise ValueError(
                f"query dim {q.shape[0]} != embedding dim {self._asr_mat.shape[1]}"
            )
        return self._asr_mat @ q

    def top_segments(self, q_asr: np.ndarray, top_k: int) -> list[AsrSemanticSegment]:
        sims = self.segment_similarities(q_asr)
        order = np.argsort(-sims)
        out: list[AsrSemanticSegment] = []
        for idx in order:
            sc = float(sims[idx])
            if sc < 1e-8 or len(out) >= top_k:
                break
            grow = int(idx)
            seg = global_row_to_segment(self._asr_by_video, grow)
            if seg is None:
                continue
            vid, start, end = seg
            out.append(
                AsrSemanticSegment(
                    video_id=vid,
                    start=start,
                    end=end,
                    global_row=grow,
                    semantic_score=sc,
                )
            )
        return out

    def keyframe_candidates(
        self,
        q_asr: np.ndarray,
        top_k: int,
        timeline: dict[str, list[tuple[int, float]]],
        *,
        min_gap: float,
        pad: float,
    ) -> dict[int, float]:
        kf_scores: dict[int, float] = {}
        for seg in self.top_segments(q_asr, top_k):
            for row_i in keyframes_for_segment(
                seg.video_id,
                seg.start,
                seg.end,
                timeline,
                min_gap=min_gap,
                pad=pad,
            ):
                kf_scores[row_i] = max(kf_scores.get(row_i, 0.0), seg.semantic_score)
        return kf_scores
