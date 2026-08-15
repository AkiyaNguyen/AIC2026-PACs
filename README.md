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
| `api/` | FastAPI wrapper around `Embedder.search` (`GET /check_health`, `POST /search`) |
| `preprocessing_tools/` | Offline extract (keyframes/CLIP, Whisper ASR) |
| `kaggle_script/` | Kaggle / local notebooks (CLIP merge, ASR embed, ASR merge) |
| `features/` | Local retrieval index (gitignored) |

Set `FEATURES_ROOT` in `.env` to the `features/` directory you search over. Optional: `DEVICE=cpu` or `gpu` (API default `gpu`). Both `engine/` and `api/` read this file.

## `features/` tree (retrieval index)

Working set is usually **L21–L24**. Same layout for the full corpus on the media drive.

```text
features/
  clip/
    model.json                 # CLIP id / dim (ViT-B-32, dim 512)
    embeddings.npy             # concatenated keyframe vectors (N_clip, 512)
    gallery_map.csv            # video_id, start_row, n_rows  (into embeddings / FAISS)
    index.faiss                # IndexFlatIP over embeddings.npy (row i = FAISS id i)
    L21/
      L21_V001.npy             # (N_frames, 512) one video; rows align with maps CSV
      ...
    L22/ L23/ L24/ ...
  maps/
    L21_V001.csv               # per keyframe: frame_idx, pts_time, fps, ...
    ...                        # row k of this CSV = row k of clip/Lxx/VIDEO_ID.npy
  asr_emb/
    model.json                 # MiniLM id / dim 384 / normalize
    embeddings.npy             # concatenated segment vectors (N_seg, 384)
    gallery_map.csv            # video_id, start_row, n_rows  (into embeddings.npy)
    L21/
      L21_V001.npy             # (N_seg, 384)
      L21_V001.jsonl           # {start, end, text} — line i ↔ npy row i
      ...
    L22/ L23/ L24/ ...
```

<!-- How rows connect at search time:

1. CLIP FAISS hit `i` → `clip/gallery_map.csv` → `video_id` + local keyframe row.
2. `maps/VIDEO_ID.csv` local row → `pts_time`, `frame_idx` (submission id).
3. `asr_emb/gallery_map.csv` → slice of `asr_emb/embeddings.npy` for that video.
4. JSONL `start`/`end` vs `pts_time` (distance &lt; 3s) → ASR rows to score.

`n_rows` in CLIP map = keyframes. `n_rows` in ASR map = speech segments (not the same count). -->

## Status

- Offline: CLIP gallery + FAISS, Whisper segments, MiniLM ASR embeddings + maps.
- Online: `engine.Embedder.search` — visual top-k pool, ASR max-cosine on that pool, weighted sum, rerank. OCR later.
- HTTP: [`api/README.md`](api/README.md)

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
# .env: FEATURES_ROOT=.../features  and optional DEVICE=cpu|gpu
python -m engine.Embedder
uvicorn api.app:app --host 127.0.0.1 --port 8000
```
