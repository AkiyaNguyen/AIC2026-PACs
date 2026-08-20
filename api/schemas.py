from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from engine.search.config import RRF_K, SearchConfig


class SearchRequest(BaseModel):
    """Hybrid KIS search parameters; unset fields use pipeline defaults."""

    query_vi: str = Field(min_length=1, description="Vietnamese query (SigLIP + ASR).")
    query_en: str | None = Field(
        default=None,
        description="English query for CLIP; if omitted, VI→EN is filled server-side.",
    )

    num_results: int = Field(default=100, ge=1, le=500)
    weight_visual: float = Field(default=0.8, ge=0.0)
    weight_transcript: float = Field(default=0.2, ge=0.0)
    weight_sem_text: float = Field(
        default=0.6,
        ge=0.0,
        description="Weight of MiniLM cosine inside the transcript mix.",
    )
    weight_bm25: float = Field(
        default=0.4,
        ge=0.0,
        description="Weight of BM25 inside the transcript mix.",
    )

    num_candidates_visual: int = Field(default=500, ge=1, le=1000)
    rrf_k: int = Field(default=RRF_K, ge=1, le=200)

    bm25_top_segments: int = Field(default=50, ge=1, le=500)
    sem_top_segments: int = Field(default=50, ge=1, le=500)
    delta: float = Field(
        default=1.0,
        ge=0.0,
        description="Keyframe <-> ASR segment eligibility window (seconds).",
    )
    segment_min_gap: float = Field(
        default=1.0,
        ge=0.0,
        description="Min seconds between sampled keyframes per ASR segment.",
    )
    segment_pad: float = Field(
        default=0.5,
        ge=0.0,
        description="Pad around segment [start, end] when mapping to keyframes.",
    )

    @model_validator(mode="after")
    def validate_fusion_weights(self) -> "SearchRequest":
        if self.weight_visual + self.weight_transcript <= 0:
            raise ValueError("At least one retrieval channel must have positive weight")
        if (
            self.weight_transcript > 0
            and self.weight_sem_text + self.weight_bm25 <= 0
        ):
            raise ValueError(
                "Transcript fusion requires a positive MiniLM or BM25 weight"
            )
        return self

    def to_search_config(self) -> SearchConfig:
        return SearchConfig(
            num_candidates_vis=self.num_candidates_visual,
            bm25_top_segments=self.bm25_top_segments,
            sem_top_segments=self.sem_top_segments,
            segment_min_gap=self.segment_min_gap,
            segment_pad=self.segment_pad,
            delta=self.delta,
            num_results=self.num_results,
            rrf_k=self.rrf_k,
        )


class Hit(BaseModel):
    rank: int
    score: float
    clip_row: int
    video_id: str
    pts_time: float
    row_idx_in_video: int
    frame_idx: int
    fps: float = Field(gt=0.0)
    source: str = Field(min_length=1)
    thumbnail_url: str | None = None
    video_url: str | None = None


class SearchResponse(BaseModel):
    query_vi: str
    query_en: str
    query_en_source: Literal["user", "translated"]
    hits: list[Hit] = Field(default_factory=list, max_length=500)
