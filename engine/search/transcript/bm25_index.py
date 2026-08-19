"""In-memory BM25 index over Whisper ASR segments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from engine.search.temporal import keyframes_for_segment

_TOKEN_RE = re.compile(r"\w+")


def tokenize_asr(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class AsrSegment:
    video_id: str
    start: float
    end: float
    global_row: int
    text: str
    bm25_score: float = 0.0


class AsrBm25Index:
    """BM25 over ASR JSONL segments, rows aligned with asr_embed_mat gallery."""

    def __init__(
        self,
        asr_root: Path,
        asr_by_video: dict[str, tuple[int, np.ndarray, np.ndarray]],
    ) -> None:
        self._segments: list[AsrSegment] = []
        tokenized: list[list[str]] = []

        for video_id, (start_row, starts, ends) in asr_by_video.items():
            jsonl = asr_root / video_id.split("_")[0] / f"{video_id}.jsonl"
            if not jsonl.is_file():
                raise FileNotFoundError(f"Missing ASR jsonl for BM25 index: {jsonl}")

            lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(lines) != len(starts):
                raise ValueError(
                    f"{video_id}: asr jsonl has {len(lines)} lines, "
                    f"gallery expects {len(starts)} segments"
                )

            for i, line in enumerate(lines):
                obj = json.loads(line)
                text = obj.get("text", "")
                self._segments.append(
                    AsrSegment(
                        video_id=video_id,
                        start=float(starts[i]),
                        end=float(ends[i]),
                        global_row=start_row + i,
                        text=text,
                    )
                )
                tokenized.append(tokenize_asr(text))

        if not tokenized:
            raise ValueError(f"No ASR segments found under {asr_root}")

        self._bm25 = BM25Okapi(tokenized)
        self._global_rows = np.asarray(
            [seg.global_row for seg in self._segments], dtype=np.int64
        )

    @property
    def n_docs(self) -> int:
        return len(self._segments)

    @property
    def global_rows(self) -> np.ndarray:
        return self._global_rows

    def score_query(self, query: str) -> np.ndarray:
        tokens = tokenize_asr(query)
        return self._bm25.get_scores(tokens).astype(np.float32)

    def top_segments(self, query: str, top_m: int) -> list[AsrSegment]:
        scores = self.score_query(query)
        order = np.argsort(-scores)
        out: list[AsrSegment] = []
        for idx in order:
            sc = float(scores[idx])
            if sc < 1e-8 or len(out) >= top_m:
                break
            seg = self._segments[int(idx)]
            out.append(
                AsrSegment(
                    video_id=seg.video_id,
                    start=seg.start,
                    end=seg.end,
                    global_row=seg.global_row,
                    text=seg.text,
                    bm25_score=sc,
                )
            )
        return out

    def keyframe_candidates(
        self,
        query: str,
        top_m: int,
        timeline: dict[str, list[tuple[int, float]]],
        *,
        min_gap: float,
        pad: float,
    ) -> dict[int, float]:
        kf_scores: dict[int, float] = {}
        for seg in self.top_segments(query, top_m):
            for row_i in keyframes_for_segment(
                seg.video_id,
                seg.start,
                seg.end,
                timeline,
                min_gap=min_gap,
                pad=pad,
            ):
                kf_scores[row_i] = max(kf_scores.get(row_i, 0.0), seg.bm25_score)
        return kf_scores
