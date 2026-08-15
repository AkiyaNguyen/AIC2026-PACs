# Tools

CLI packages for offline preprocess and (later) search. Prefer `python -m tools.<pkg>` from the repo root. Avoid a top-level `scripts/` folder on Windows — it collides with venv `Scripts/`.

## Packages

| Package | Role |
|---------|------|
| `extract_features` | Keyframe extract + CLIP gallery organize |
| `extract_transcript` | Video → WAV → Whisper ASR JSONL |
| `util` | Shared helpers (e.g. `get_proper_device`) |

Kaggle notebooks that wrap these flows: [`kaggle_script/`](../kaggle_script/).

---

## Keyframe extract + embed

Package: `tools.extract_features`. Default **CPU**; add `--device gpu` (or `cuda`) to run CLIP on CUDA. Extract samples every `--stride` frame and keeps a frame only if CLIP cosine distance from the last kept frame is large enough.

```bash
# VIDEO_DIR can be the dataset parent — finds all nested videos in one run
python -m tools.extract_features extract /kaggle/input/.../dataset-aic \
  --out-dir /kaggle/working/keyframes-out --device gpu

# After extract: copy VIDEO_ID/embeddings.npy → gallery/VIDEO_ID.npy (no second CLIP pass)
python -m tools.extract_features embed /kaggle/working/keyframes-out \
  --out-dir /kaggle/working/clip-gallery --copy-embeddings
```

Extract walks the input folder recursively. Output is **flat** `keyframes-out/VIDEO_ID/...` (not a mirror of `Videos_L21_a/`). Embed `--copy-embeddings` writes `clip-gallery/VIDEO_ID.npy`.

Organize into the retrieval layout (`features/clip/Lxx/`, `features/maps/`) via the Kaggle notebook [`kaggle_script/kf_embed_extraction.ipynb`](../kaggle_script/kf_embed_extraction.ipynb), or equivalent local scripts. Pipeline overview: [`docs/memory/pipeline.md`](../docs/memory/pipeline.md).

---

## ASR transcript extract

```bash
pip install -r preprocessing_tools/requirements-asr.txt   # faster-whisper; ffmpeg on PATH

python -m preprocessing_tools.extract_transcript /path/to/Videos_L21 \
  --audio-dir features/audio/L21 \
  --out-dir features/asr/L21 \
  --device gpu
```

Writes `VIDEO_ID.jsonl` segments `{start, end, text}`. Model is fixed **`large-v3`**. Kaggle: [`kaggle_script/asr_extraction.ipynb`](../kaggle_script/asr_extraction.ipynb).

ASR **segment embeddings** use root [`requirements.txt`](../requirements.txt) (`sentence-transformers`): [`kaggle_script/asr_embed_extraction.ipynb`](../kaggle_script/asr_embed_extraction.ipynb) → `features/asr_emb/`.
