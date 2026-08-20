from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from api.schemas import Hit, SearchRequest, SearchResponse
from engine.query_language import QueryTranslationError
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


@app.get("/check_health")
def check_health():
    return {"Ok": True}


@app.post("/search")
def search(request: SearchRequest) -> SearchResponse:
    assert search_service is not None
    try:
        resolved, raw_hits = search_service.search(
            request.query_vi,
            request.query_en,
            cfg=request.to_search_config(),
            weight_visual=request.weight_visual,
            weight_transcript=request.weight_transcript,
            weight_sem_text=request.weight_sem_text,
            weight_bm25=request.weight_bm25,
        )
    except QueryTranslationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    hits = [Hit(rank=i, **hit) for i, hit in enumerate(raw_hits, start=1)]
    return SearchResponse(
        query_vi=resolved.query_vi,
        query_en=resolved.query_en,
        query_en_source=resolved.query_en_source,
        hits=hits,
    )
