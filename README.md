# PACs — AIC 2026

Internal repository for team **PACs** in the Ho Chi Minh City AI Challenge (AIC) 2026: shared code and contest submission artifacts.

## Problem summary (preliminary round)

Three query types:

1. **Textual KIS** — Given a natural-language event description, return a matching `video_id` and any `frame_id` inside the correct segment.
2. **Q&A** — Find the relevant moment and answer a question about it (`video_id`, `frame_id`, `answer` in Vietnamese or English).
3. **TRAKE** — Retrieve the correct video, then return one semantic keyframe per ordered event in a sequence.

Each query allows up to **100** ranked answers. Scoring uses R-Score per answer and a **Final Score** that averages best R-Score at cutoffs `@1, @5, @20, @50, @100` — so ranking order matters.

## Repository layout

| Path | Purpose |
|------|---------|
| `engine/` | Retrieval: CLIP/ASR encoders, FAISS, keyframe→ASR map, `search()` |
| `api/` | FastAPI wrapper around `SearchService.search`, including local media endpoints |
| `ui/` | Web UI for Textual KIS search, result inspection, copy and CSV export |
| `preprocessing_tools/` | Offline extract (keyframes/CLIP, Whisper ASR) |
| `kaggle_script/` | Kaggle / local notebooks (CLIP merge, ASR embed, ASR merge, SigLIP2 embed) |
| `features/` | Local retrieval index (gitignored) |

Set `FEATURES_ROOT` in `.env` to the `features/` directory you search over. Optional: `DEVICE=cpu` or `gpu` (API default `gpu`). Both `engine/` and `api/` read this file.

## `features/` tree (retrieval index)

Set `FEATURES_ROOT` to this directory. Full corpus on the media drive is L21–L30; a smaller working set may omit later batches.

```text
features/
  clip/                        # required — CLIP FAISS (dim 512)
    model.json
    embeddings.npy             # (N_kf, 512) row i = FAISS id i
    gallery_map.csv            # video_id, start_row, n_rows
    index.faiss                # IndexFlatIP
    L21/
      L21_V001.npy             # (N_frames, 512); rows align with maps CSV
      ...
    L22/ ... L30/
  SigLIP2/                     # required — 2nd visual FAISS (dim 1152); same rows as clip/
    model.json
    embeddings.npy
    gallery_map.csv
    index.faiss
    L21/
      L21_V001.npy             # (N_frames, 1152)
      ...
    L22/ ... L30/
  maps/                        # required — one CSV per video (flat)
    L21_V001.csv               # n, pts_time, fps, frame_idx
    ...                        # row k = clip/Lxx/VIDEO_ID.npy row k = SigLIP2 row k
  asr_emb/                     # required when weight_transcript > 0
    model.json                 # MiniLM id / dim 384 / normalize
    embeddings.npy             # (N_seg, 384)
    gallery_map.csv            # video_id, start_row, n_rows
    L21/
      L21_V001.npy             # (N_seg, 384)
      L21_V001.jsonl           # {start, end, text} — line i ↔ npy row i
      ...
    L22/ ... L30/
  asr/                         # optional at query if asr_emb jsonl is present
    L21/
      L21_V001.jsonl           # Whisper {start, end, text}
      ...
```

Search uses `clip/` + `SigLIP2/` + `maps/` always; `asr_emb/` when mixing ASR. `clip_row` is shared across CLIP and SigLIP2. `n_rows` in CLIP/SigLIP2 maps = keyframes; in ASR map = speech segments.

## Status

- Offline: CLIP + SigLIP2 galleries + FAISS, Whisper segments, MiniLM ASR embeddings + maps.
- Online: `SearchService` + `HybridSearcher` — visual RRF + BM25/semantic text candidates, union pool, weighted fusion. See `engine/search/`.
- HTTP: [`api/README.md`](api/README.md)

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
# .env: FEATURES_ROOT=.../features  and optional DEVICE=cpu|gpu
python -m engine.search_service
uvicorn api.app:app --host 127.0.0.1 --port 8000

# In another terminal
cd ui
npm install
npm run dev
```
