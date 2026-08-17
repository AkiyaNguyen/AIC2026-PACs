# Search API

HTTP wrapper for `engine.Embedder.search`. Run from the repo root with `.venv` active.

## `.env`

Required for `engine/` and `api/` (do not commit secrets):

```
FEATURES_ROOT=<absolute path to features/>
DEVICE=cpu
```

`DEVICE` is optional (`cpu` | `gpu`; default `gpu`).

`POST /search` field `weight_clip` is the **visual** weight \(w_v\) on min-max’d RRF (CLIP + SigLIP2 ranks), not raw CLIP cosine. `weight_asr` is \(w_a\) after min-max on the same pool. Both `FEATURES_ROOT/clip` and `FEATURES_ROOT/SigLIP2` are required.

## Run

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Wait until models finish loading. Docs: http://127.0.0.1:8000/docs

| Method | Path |
|--------|------|
| `GET` | `/check_health` |
| `POST` | `/search` |

Git Bash: POST JSON with `--data-binary @search.json`, not `curl -d "..."`.
