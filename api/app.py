from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from api.schemas import Hit, SearchRequest, SearchResponse
from engine.Embedder import Embedder

load_dotenv()

embedder: Embedder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    print("Starting up...")
    embedder = Embedder(device=os.getenv("DEVICE", "gpu"))
    yield
    print("Shutting down...")
    embedder = None


app = FastAPI(lifespan=lifespan)


@app.get("/check_health")
def check_health():
    return {"Ok": True}


@app.post("/search")
def search(request: SearchRequest) -> SearchResponse:
    assert embedder is not None
    raw_hits = embedder.search(
        request.query,
        request.num_candidates,
        request.num_results,
        request.weight_clip,
        request.weight_asr,
        request.delta,
    )
    hits = [Hit(rank=i, **hit) for i, hit in enumerate(raw_hits, start=1)]
    return SearchResponse(query=request.query, hits=hits)
