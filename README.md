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
| `tools/` | CLI tools (KIS search, `extract_features` package, optional keyframe compare). Avoid a top-level `scripts/` folder on Windows — it collides with venv `Scripts/`. |
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

## Keyframe extract + embed (NII-UIT + CLIP)

Package: `tools/extract_features`. Default **CPU**; add `--device gpu` (or `cuda`) to run CLIP on CUDA. Extract samples every `--stride` frame and keeps a frame only if CLIP cosine distance from the last kept frame is large enough.

```bash
# VIDEO_DIR can be the dataset parent — finds all nested videos in one run
python -m tools.extract_features extract /kaggle/input/.../dataset-aic \
  --out-dir /kaggle/working/keyframes-out --device gpu

# After extract: copy VIDEO_ID/embeddings.npy → gallery/VIDEO_ID.npy (no second CLIP pass)
python -m tools.extract_features embed /kaggle/working/keyframes-out \
  --out-dir /kaggle/working/clip-gallery --copy-embeddings
```

Extract walks the input folder recursively. Output is **flat** `keyframes-out/VIDEO_ID/...` (not a mirror of `Videos_L21_a/`). Embed `--copy-embeddings` writes `clip-gallery/VIDEO_ID.npy` for `kis_search.py --clip-dir`. Kaggle notebook: [`tools/aic2026-extract-features.ipynb`](tools/aic2026-extract-features.ipynb).

Coding-agent constraints live in [AGENTS.md](AGENTS.md).
