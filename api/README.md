# Search API

HTTP wrapper for `engine.Embedder.search`. Run from the repo root with `.venv` active.

## `.env`

Required for `engine/` and `api/` (do not commit secrets):

```
FEATURES_ROOT=<absolute path to features/>
KEYFRAMES_ROOT=<absolute path to media/keyframes/>
VIDEOS_ROOT=<absolute path to media/videos/>
DEVICE=cpu
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

`DEVICE` is optional (`cpu` | `gpu`; default `gpu`).
`CORS_ORIGINS` is optional and controls which browser origins may call the API.
`KEYFRAMES_ROOT` contains flat `VIDEO_ID/000000.webp` folders. Search hits include
`thumbnail_url` only when the matching still is available locally.
`VIDEOS_ROOT` contains `video_BATCH/VIDEO_ID.mp4` files. Search hits include
`video_url` only when the matching MP4 exists. The video endpoint supports HTTP byte
ranges through Starlette `FileResponse`, allowing browser seeking without a full download.
Each hit also includes the source `fps`, which the UI uses to convert the current
video playhead into an adjusted `frame_idx` for KIS submission.

`POST /search` field `weight_clip` is the **visual** weight \(w_v\) on min-max’d RRF (CLIP + SigLIP2 ranks), not raw CLIP cosine. `weight_asr` is \(w_a\) after min-max on the same pool. Both `FEATURES_ROOT/clip` and `FEATURES_ROOT/SigLIP2` are required.

## Run

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Wait until models finish loading. Docs: http://127.0.0.1:8000/docs

| Method | Path |
|--------|------|
| `GET` | `/check_health` |
| `GET` | `/media/keyframes/{video_id}/{row_idx_in_video}` |
| `GET` | `/media/videos/{video_id}` |
| `POST` | `/search` |

Git Bash: POST JSON with `--data-binary @search.json`, not `curl -d "..."`.
