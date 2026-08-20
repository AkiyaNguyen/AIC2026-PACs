# Search API

HTTP wrapper for `SearchService.search`. Run from the repo root with `.venv` active.

## `.env`

Required for `engine/` and `api/` (do not commit secrets):

```
FEATURES_ROOT=<absolute path to features/>
DEVICE=cpu
```

`DEVICE` is optional (`cpu` | `gpu`; default `gpu`).

## `POST /search` body

All fields optional except `query`. Defaults match `engine/search/config.py`.

| Field | Default | Role |
|-------|---------|------|
| `query` | — | Natural-language KIS query (required) |
| `num_results` | `100` | Max ranked hits returned |
| `weight_visual` | `0.8` | Fusion weight on visual RRF channel |
| `weight_transcript` | `0.2` | Fusion weight on transcript mix (semantic + BM25) |
| `weight_sem_text` | `0.6` | MiniLM cosine share **inside** transcript (normalized with `weight_bm25`) |
| `weight_bm25` | `0.4` | BM25 share **inside** transcript |
| `num_candidates_visual` | `500` | ANN top-K per CLIP and SigLIP2 tower |
| `rrf_k` | `60` | RRF constant for CLIP + SigLIP2 merge |
| `bm25_top_segments` | `50` | BM25 segment proposals |
| `sem_top_segments` | `50` | MiniLM semantic segment proposals |
| `delta` | `1.0` | Keyframe ↔ ASR segment eligibility (seconds) |
| `segment_min_gap` | `1.0` | Min gap between keyframes sampled per segment |
| `segment_pad` | `0.5` | Pad around segment interval when mapping to keyframes |

Minimal request:

```json
{"query": "Xuất khẩu gạo Việt Nam"}
```

Transcript-heavy tuning example:

```json
{
  "query": "Xuất khẩu gạo Việt Nam",
  "weight_visual": 0.3,
  "weight_transcript": 0.7,
  "weight_sem_text": 0.5,
  "weight_bm25": 0.5,
  "bm25_top_segments": 80,
  "delta": 1.5
}
```

Both `FEATURES_ROOT/clip` and `FEATURES_ROOT/SigLIP2` are required for visual search. Transcript fields need `asr/`, `asr_emb/`, and Whisper JSONL under `asr/`.

## Run

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
# after API is up: source tools/bashes/aliases.sh
# search --input test_query/doctor_search.json --out result_search.json
```

Wait until models finish loading. Docs: http://127.0.0.1:8000/docs

| Method | Path |
|--------|------|
| `GET` | `/check_health` |
| `POST` | `/search` |

Git Bash: POST JSON with `--data-binary @search.json`, not `curl -d "..."`.
