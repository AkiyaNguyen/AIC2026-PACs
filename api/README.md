# Search API

HTTP wrapper for `engine.Embedder.search`. Run from the repo root with `.venv` active.

## `.env`

Required for `engine/` and `api/` (do not commit secrets):

```
FEATURES_ROOT=<absolute path to features/>
DEVICE=cpu
```

`DEVICE` is optional (`cpu` | `gpu`; default `gpu`).

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
