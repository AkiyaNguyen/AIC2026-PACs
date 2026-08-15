from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    num_candidates: int = Field(default=500, ge=1, le=1000)
    num_results: int = Field(default=100, ge=1, le=500)
    weight_clip: float = Field(default=0.8, ge=0.0)
    weight_asr: float = Field(default=0.2, ge=0.0)
    delta: float = Field(default=3.0, ge=0.0)


class Hit(BaseModel):
    rank: int
    score: float
    clip_row: int
    video_id: str
    pts_time: float
    row_idx_in_video: int
    frame_idx: int


class SearchResponse(BaseModel):
    query: str
    hits: list[Hit] = Field(default_factory=list, max_length=500)
