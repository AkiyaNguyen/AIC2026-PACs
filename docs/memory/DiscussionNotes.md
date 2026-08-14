# Discussion notes — PACs method direction

Living notes from **human ↔ agent** discussion about what **we** may build for AIC 2026.  
Not a copy of prior-team papers — for paper facts use `PreviousTeamSubmission/SUMMARIES.md`.

Update this file when a design idea is refined, accepted, deferred, or rejected.

---

## Current working baseline

| Item | Notes |
|------|--------|
| Status | Active first slice |
| Idea | Organizer CLIP ViT-B/32 `.npy` gallery + CLIP text query → top-k → `map-keyframes` → `(video_id, frame_id)`; preview via YouTube `watch?v=&t=` (`moment_url`) |
| Code | `tools/kis_search.py` |
| Open | FAISS, second embedding, ASR + semantic text embed/search, contest UI |
| Keyframes | Package `tools/extract_features`: **extract** (videos → stills + `embeddings.npy`) and **embed** (still tree → `VIDEO_ID.npy`). After extract, `embed --copy-embeddings` copies those npy files (no second CLIP pass). Default CPU; `--device gpu`. |
| OCR (v1) | Package `tools/extract_ocr`: default **EasyOCR** (en+vi) over still tree → `OUT/VIDEO_ID/ocr.jsonl` (1:1 with map rows). PaddleOCR optional via `--backend paddleocr` when wheels exist. Semantic embed of `ocr_text` is a later slice. Deps: `requirements-ocr.txt`. |

---

## Ideas under discussion (refine here)

### Storage / preview

- Large media hosting undecided (local SSD vs Kaggle indexing vs stream-only preview).
- Do not lock a path in README until chosen; ask before inventing layout in code.
- YouTube watch/`t=` links are **unreliable** (team has hit dead/blocked videos) → **not** the only preview path.
- **Index vs media (working rule):** Organize **embeddings** (and maps) for search — flat `VIDEO_ID.npy` + `map.csv` / `frame_idx` is enough for KIS ranking. Media uses lookups (`video_id` → keyframe dir, `video_id` → mp4 path). Do not merge every Kaggle zip into one mirrored mega-tree unless forced.
- **Index layout (accepted 2026-08-12):** `features/clip/L**/VIDEO_ID.npy` + flat `features/maps/VIDEO_ID.csv` (same rows as keyframe `map.csv`). Kaggle stills datasets: zip root = `VIDEO_ID/{map.csv,*.webp,embeddings.npy}` (see `tools/kf_embed_extraction.ipynb`). Local bulky stills may live outside the repo (e.g. `C:/AIC2026-media/`).
- **Minimum offline media (draft, pending host choice):**
  1. **Hot (search):** CLIP gallery + maps only — laptop/GPU box running query.
  2. **Warm (UI preview — required when YT fails):** **keyframe stills** on a disk the UI can open fast (local SSD / NAS / object storage with HTTP). Path pattern TBD; lookup by `video_id` + keyframe file from `map.csv`. This is the **minimum** visual DB for result grids (same idea as VBS systems: VISIONE / diveXplore / VideoEase serve keyframe images, not YouTube).
  3. **Cold (playback / verify motion):** **raw mp4** on bulk storage (external HDD, Kaggle dataset, or same NAS). Not loaded into the vector store; open/seek only when user clicks a hit. Optional later: reduced proxies for scrubbing.
  4. YouTube `moment_url` remains a **bonus** when it works, never the sole preview.
  5. Later ANN: FAISS/Milvus for vectors only; media stays outside (Vortex/U-CESE/NII-UIT + Milvus blog pattern).

### Temporal captions / near-past context (ReCap-style)

- U-CESE **ReCap** builds captions with recurrent memory over past shots + current shot, so indexed text can match queries that refer to **near-past** context, not only what is visible “now.”
- That helps retrieval/UI for sequential wording; it does **not** by itself produce TRAKE-aligned semantic frames.
- Temporal / memory-based video understanding is a common research line (recurrent captioning, memory-augmented VLMs, temporal transformers, etc.). **Research later** for stronger variants; do not treat ReCap as the only option.
- Related prior art locally: `PreviousTeamSubmission/SUMMARIES.md` (U-CESE ReCap; Vortex temporal Alg. 1 is a different mechanism — ranking boost, not caption memory).

### Semantic text channel vs ElasticSearch (OCR / ASR)

- Classic ES/BM25 matches **surface tokens** (plus optional synonyms/analyzers). It will **not** reliably match English query “fruit” to OCR “chuối” without translation, synonym lists, or semantic embeddings.
- Direction: keep keyword/ES-style filters as a **cheap** tool; plan a **semantic text channel** (embed OCR/ASR/captions with a multilingual text encoder or CLIP-class text tower; cosine search; optional query expansion / translation). Same spirit as U-CESE TextualDB embeddings, not only raw ES.
- Keep this channel **separate** from visual RRF unless we later decide to fuse ranks deliberately.

### ASR / transcript extractor models (discussion 2026-08-14)

- **Local papers:** Vortex and U-CESE both name **OpenAI Whisper** (Radford et al. 2022) as the ASR; they do **not** publish size (`large-v2` vs `large-v3`) or runtime (`openai-whisper` vs `faster-whisper`). Vortex aligns subtitle intervals onto keyframes (`a ≤ t_k ≤ b`) and **carries last speech through silence**. U-CESE uses Whisper subtitles as ReCap input, then indexes caption/ASR in ES + MobileCLIP text. NII-UIT VBS write-up has **no ASR**. Repo stub: `tools/extract_transcript/` (empty); OCR JSONL contract is the intended sibling.
- **U-CESE related work:** Whisper is “the primary ASR model used in several systems”; **Vintern-1B is captioning/OCR, not ASR**.
- **Web / VBS:** diveXplore uses Whisper; Fusionista2.0 (VBS, UIT/AISIA) switched vanilla Whisper → **faster-whisper** (CTranslate2, ~4×) because V3C audio is often ambient, so a huge checkpoint is not worth the cost. VBS also ships shared ASR dumps (Rossetto et al.) — AIC does not, so we must run our own.
- **Not the same thing:** `faster-whisper` = same Whisper weights, faster engine. **PhoWhisper** (VinAI, arXiv:2406.02555) = Whisper fine-tuned on 844h Vietnamese (large = Whisper **large-v2** architecture). **WhisperX** = Whisper + forced alignment for tighter timestamps.
- **PACs (2026-08-14):** `faster-whisper` + **`large-v3`**. CLI: `python -m tools.extract_transcript INPUT --audio-dir features/audio/Lxx --out-dir features/asr/Lxx --device gpu`. ffmpeg WAV 16 kHz mono → JSONL segments (`start`/`end`/`text`). Deps: `requirements-asr.txt`. Kaggle: [`kaggle_script/asr_extraction.ipynb`](../../kaggle_script/asr_extraction.ipynb) (zip JSONL only). Shared device: `tools/util.get_proper_device`.

### After baseline KIS (candidates, not committed)

1. **Second visual encoder + RRF** (Vortex-style CLIP↔SigLIP-class fusion).
2. **Text channel** — keyword filter **and/or** semantic OCR/ASR/caption retrieval (see above).
3. **TRAKE-oriented ranking unit:** multi-event queries + time window maximizing unique events covered (U-CESE-style), vs Before/Now/After boost on a single “current” list (Vortex-style) as a lighter UI heuristic; NII-UIT neighbor-shot multi-stage is a third soft-temporal pattern. All three mainly find a **co-located event cluster** (same video / nearby shots), not a hard proof of `t1 < t2 < … < tn`.
4. Interactive feedback (Rocchio-like) — later, after search quality is usable.
5. Temporal-memory captioning research follow-up (beyond copying ReCap).
6. **Stable Diffusion as query channel** — optional later; evidence mixed (helpful as complementary / UI imagination; noisy alone; prefer verify or web-image alternatives). Not needed for first KIS loop.

### Soft temporal vs ordered TRAKE (open debate)

- **Local prior art:** Vortex Alg. 1 boosts Current if same `video_id` has strong Prev/Next hits (no timestamp order check). U-CESE ranks clips by **#distinct queries covered** inside window \(T\). NII-UIT explores **neighbor shots** around an anchor and aggregates stages (order intentionally soft when KIS-T before/after is unclear). See `PreviousTeamSubmission/SUMMARIES.md`.
- **Web/VBS line:** Many interactive systems *can* enforce user-specified order (e.g. vitrivr temporal sequences with order + distance decay; SOMHunter/CVHunter fuse first hit with best match inside a time delta). That is stronger than pure “cluster cover,” but still usually **pairwise / window heuristics**, not full sequence alignment for TRAKE-length chains.
- **Reading for PACs:** The three papers we studied optimize for **finding a video neighborhood that contains the events**, then rely on UI/operator to pick ordered frames. True **ordered alignment** (penalize wrong permutation, require monotonic timestamps) is under-used in those write-ups — a plausible gap if AIC TRAKE scoring cares about frame order/identity, not only co-occurrence in a clip.
- **Not claiming novelty:** ordered temporal fusion and A*-style temporal scoring already appear in VBS literature; the gap is whether **we** implement hard order for TRAKE vs stay with soft cover + UI.

### Stable Diffusion query generation (effectiveness)

- NII-UIT uses SD at **query time** (text → image → visual ANN), fused with text/paraphrase channels; paper does **not** ablate “with vs without SD” for the VBS win.
- Prior art: Ma et al. (MMM 2024) motivate LLM + generative imagination for interactive KIS; Wu et al. arXiv:2407.12341 find T2I paraphrases **complement** text but can inject noise — **QA/self-verification** helps; ACCV 2024 (Nguyen et al.) report **SD-only** retrieval can be weak vs text embeddings, and argue **web image search** can be faster/better than generating.
- **PACs stance (tentative):** treat SD as optional diversity channel / UI aid, not a core win condition; prefer solid CLIP(+RRF) + paraphrase/text channel first; if SD is tried, fuse with missing=0 union (not intersection) and consider verification or human pick among generated images.

### Pluggable system code pattern

- Prefer a **pluggable retrieval stack**: shared gallery/metadata interfaces; swap backends (CLIP-only → RRF → +semantic text → temporal/clip-cover) without rewriting the UI.
- Exact package layout deferred; principle: easy A/B of baselines during method development.

### UI as a first-class track

- Interactive UI is part of the **contest strategy**, not an afterthought (inspect hits, multi-query / temporal fields, prefer/reject, export submission rows) — aligned with Vortex/U-CESE practice.
- Prioritize UI once the CLIP KIS loop is trustworthy enough to drive interaction.

### Repo / agent conventions (settled unless revisited)

- English for repo-authored docs; BTC Vietnamese sources unchanged.
- Local machine / checkout setup (venv paths, CLI folder layout, etc.) lives in `AGENTS.md` only — not in these notes (docs are shared via GitHub).
- Paper PDFs → sibling `.md` extract; summaries in `PreviousTeamSubmission/SUMMARIES.md`; memory for **our** direction → this file.
- When discussing methods/system design: **cite existing work first** (local summaries + this file + repo code, then short web/arXiv search). See AGENTS.md / project-memory rule.
- **Human learning + best practice:** follow the human’s idea and repo patterns when they are valid; if a common practice is better, **say so and discuss before implementing**. Explain load-bearing concepts (tests, extract vs embed, device flags, packages). Hard rule also in `AGENTS.md`.

---

## Decision log (compact)

| Date | Topic | Outcome |
|------|--------|---------|
| 2026-08-05 | README vs AGENTS | Lean README; hard rules in AGENTS |
| 2026-08-05 | Storage story in README | Deferred — keep vague |
| 2026-08-05 | Baseline stack | CLIP KIS CLI first |
| 2026-08-05 | Moment links | `moment_url` = watch + `t=` |
| 2026-08-06 | Prior papers | `PreviousTeamSubmission/` + extracts + summaries |
| 2026-08-06 | Memory shape | Single `DiscussionNotes.md` for our method discussion; paper facts in `PreviousTeamSubmission/SUMMARIES.md` |
| 2026-08-06 | Anti-reinvention | Cite local prior art + web/arXiv before proposing “new” methods |
| 2026-08-06 | UI priority | Interactive UI is a first-class track for contest strategy |
| 2026-08-06 | OCR/ASR | Keyword filter alone insufficient for EN↔VI semantics; plan semantic text channel |
| 2026-08-06 | DiscussionNotes scope | No local-machine checkout details here; keep those in `AGENTS.md` |
| 2026-08-06 | NII-UIT paper | Summarized in `SUMMARIES.md`; extract `NII-UIT_VBS2025.md` |
| 2026-08-06 | Soft temporal | Vortex / U-CESE / NII-UIT mostly cluster/neighborhood scoring; hard order under-specified — open for TRAKE |
| 2026-08-07 | Keyframe CLI | NII-UIT-style extract (`clip`/`siglip`/`beit3`) + compare (density/TRAKE gaps + HTML samples) under `tools/` |
| 2026-08-12 | Extract encoder | Commit **CLIP** as the only default extract; skip 3-way compare as a gate. SigLIP = optional later embed on same frames; our BEiT-3 checkpoint is not NII-UIT’s. |
| 2026-08-12 | extract_features pkg | `tools/extract_features` engines (extract + embed); CLI `--device cpu\|gpu\|cuda`; removed `extract_keyframes_niiuit.py` / `keyframe_models.py`. |
| 2026-08-12 | Agent collab | Do not blindly follow suggestions; recommend best practice and discuss first. Explain vital concepts so the human learns the stack. |
| 2026-08-12 | Media vs index | Prioritize organized CLIP gallery + maps; videos/stills only need resolvable paths (lookup), not one merged tree. |
| 2026-08-12 | OCR v1 | EasyOCR default (`tools/extract_ocr` → `ocr.jsonl`); PaddleOCR optional (no wheel on Py3.14 here). ASR / text-embed deferred. |
| 2026-08-12 | features layout | `features/clip/L**/VIDEO_ID.npy` + `features/maps/VIDEO_ID.csv`; Kaggle stills zip root = `VIDEO_ID/…` (`kf_embed_extraction.ipynb`). |
| 2026-08-14 | ASR extract | CLI + engine: ffmpeg WAV → faster-whisper `large-v3` → `features/asr/Lxx/VIDEO_ID.jsonl`. `--audio-dir` for wavs. |

---

## Next discussion prompts

Use this section when continuing design chats:

- [ ] What do we want ranked first for TRAKE: **clips covering N events** or **boosted single-event lists**?
- [ ] For TRAKE, do we add **hard timestamp order** (monotonic `f1 < f2 < …`) on top of soft cluster cover?
- [ ] Do we keep BTC keyframes only, or also index the CLIP re-extract (stride/threshold tweak if TRAKE gaps look large)? SigLIP/BEiT-3 trees not required.
- [x] OCR extract v1: EasyOCR → `ocr.jsonl` (`tools/extract_ocr`; Paddle optional)
- [ ] Semantic text channel: which encoder (multilingual sentence transformer vs CLIP text) for OCR/ASR/captions?
- [ ] ASR engine (Whisper) → `features/asr/Lxx/VIDEO_ID.jsonl` (CLI dry-run exists)
- [ ] Research note: survey temporal-memory / recurrent video captioning beyond ReCap
- [ ] When to start UI relative to second embedding / RRF
- [ ] SD / generative queries: skip for v1, or A/B as optional fused channel with verification?
