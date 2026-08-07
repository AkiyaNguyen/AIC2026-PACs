# PACs — AIC 2026

Internal repository for team **PACs** in the Ho Chi Minh City AI Challenge (AIC) 2026: shared code, docs, and contest submission artifacts.

## Problem summary (preliminary round)

Three query types:

1. **Textual KIS** — Given a natural-language event description, return a matching `video_id` and any `frame_id` inside the correct segment.
2. **Q&A** — Find the relevant moment and answer a question about it (`video_id`, `frame_id`, `answer` in Vietnamese or English).
3. **TRAKE** — Retrieve the correct video, then return one semantic keyframe per ordered event in a sequence.

Each query allows up to **100** ranked answers. Scoring uses R-Score per answer and a **Final Score** that averages best R-Score at cutoffs `@1, @5, @20, @50, @100` — so ranking order matters.

Full official statement and scoring details: [docs/organization-board/thong-tin-vong-so-tuyen-aic2026.md](docs/organization-board/thong-tin-vong-so-tuyen-aic2026.md).

## Repository layout

| Path | Purpose |
|------|---------|
| `docs/organization-board/` | Official Organization Board (BTC) materials (PDF + parsed text) |
| `map-keyframes-aic25-b1/` | Keyframe → `frame_idx` maps (CSV) for batch-1 sample data |
| `media-info-aic25-b1/` | Per-video metadata JSON (YouTube-style fields) |
| `clip-features-32-aic25-b1/` | Per-video CLIP ViT-B/32 `.npy` features |
| `objects-aic25-b1/` | Per-keyframe object detections (JSON) |
| `tools/` | CLI tools (KIS search, NII-UIT-style keyframe extract/compare). Avoid a top-level `scripts/` folder on Windows — it collides with venv `Scripts/`. |
| `queries/` | Sample text query files |
| `PreviousTeamSubmission/` | Prior AIC team papers (PDF + `.md` extract); summaries in `SUMMARIES.md` |
| `docs/memory/` | `DiscussionNotes.md` — our method direction (human↔agent) |
| `.venv/` | Local Python virtualenv (not committed) |

## Status

Early setup. Baseline Textual KIS CLI exists; fuller pipeline still TBD.

## Quick start (KIS search)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python tools/kis_search.py queries/sample_kis.txt --top_k 10
```

## Keyframe extraction (NII-UIT-style)

Stride + semantic dedup (`clip` | `siglip` | `beit3`). Output: `OUT/VIDEO_ID/{map.csv,*.webp,embeddings.npy}`.

```bash
python tools/extract_keyframes_niiuit.py path/to/videos_or_video.mp4 \
  --out-dir keyframes-out/clip --model clip --stride 10 --min-cosine-distance 0.15

python tools/compare_keyframes.py \
  --roots \
    clip:keyframes-out/clip \
    siglip:keyframes-out/siglip \
    beit3:keyframes-out/beit3 \
    btc:keyframes-out/btc \
  --out-dir compare-out/L21_4way
```

Each root must contain `VIDEO_ID/map.csv` (+ images). Open `compare-out/.../index.html`. Density stats flag gaps larger than ~10 frames (typical TRAKE answer window length). Legacy `--dir-a/--dir-b` still works for pairwise compares.

Coding-agent constraints live in [AGENTS.md](AGENTS.md).
