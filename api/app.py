from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.media import router as media_router
from api.media import thumbnail_url_if_available, video_url_if_available
from api.schemas import Hit, SearchRequest, SearchResponse
from engine.search_service import SearchService

load_dotenv()

search_service: SearchService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_service
    print("Starting up...")
    search_service = SearchService(device=os.getenv("DEVICE", "gpu"))
    yield
    print("Shutting down...")
    search_service = None


app = FastAPI(lifespan=lifespan)

default_origins = "http://localhost:3000,http://127.0.0.1:3000"
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", default_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.include_router(media_router)


@app.get("/check_health")
def check_health():
    return {"Ok": True}


@app.post("/search")
def search(request: SearchRequest) -> SearchResponse:
    assert search_service is not None
    raw_hits = search_service.search(
        request.query,
        cfg=request.to_search_config(),
        weight_visual=request.weight_visual,
        weight_transcript=request.weight_transcript,
        weight_sem_text=request.weight_sem_text,
        weight_bm25=request.weight_bm25,
    )
    hits = [
        Hit(
            rank=i,
            **hit,
            thumbnail_url=thumbnail_url_if_available(
                hit["video_id"], hit["row_idx_in_video"]
            ),
            video_url=video_url_if_available(hit["video_id"]),
        )
        for i, hit in enumerate(raw_hits, start=1)
    ]
    return SearchResponse(query=request.query, hits=hits)
