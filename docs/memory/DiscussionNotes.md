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
| Open | FAISS, second embedding, OCR/ASR index, contest UI — not required for first loop |
| Keyframes | CLI: `tools/extract_keyframes_niiuit.py` (stride + semantic dedup; models `clip`/`siglip`/`beit3`); compare: `tools/compare_keyframes.py` (density vs ~10-frame TRAKE windows + HTML manual samples) |

---

## Ideas under discussion (refine here)

### Storage / preview

- Large media hosting undecided (local SSD vs Kaggle indexing vs stream-only preview).
- Do not lock a path in README until chosen; ask before inventing layout in code.
- YouTube embeds with `autoplay=0` often unusable → prefer watch URLs with `t=`.

### Temporal captions / near-past context (ReCap-style)

- U-CESE **ReCap** builds captions with recurrent memory over past shots + current shot, so indexed text can match queries that refer to **near-past** context, not only what is visible “now.”
- That helps retrieval/UI for sequential wording; it does **not** by itself produce TRAKE-aligned semantic frames.
- Temporal / memory-based video understanding is a common research line (recurrent captioning, memory-augmented VLMs, temporal transformers, etc.). **Research later** for stronger variants; do not treat ReCap as the only option.
- Related prior art locally: `PreviousTeamSubmission/SUMMARIES.md` (U-CESE ReCap; Vortex temporal Alg. 1 is a different mechanism — ranking boost, not caption memory).

### Semantic text channel vs ElasticSearch (OCR / ASR)

- Classic ES/BM25 matches **surface tokens** (plus optional synonyms/analyzers). It will **not** reliably match English query “fruit” to OCR “chuối” without translation, synonym lists, or semantic embeddings.
- Direction: keep keyword/ES-style filters as a **cheap** tool; plan a **semantic text channel** (embed OCR/ASR/captions with a multilingual text encoder or CLIP-class text tower; cosine search; optional query expansion / translation). Same spirit as U-CESE TextualDB embeddings, not only raw ES.
- Keep this channel **separate** from visual RRF unless we later decide to fuse ranks deliberately.

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

---

## Next discussion prompts

Use this section when continuing design chats:

- [ ] What do we want ranked first for TRAKE: **clips covering N events** or **boosted single-event lists**?
- [ ] For TRAKE, do we add **hard timestamp order** (monotonic `f1 < f2 < …`) on top of soft cluster cover?
- [ ] Do we keep BTC keyframes only, or also re-extract (NII-UIT stride+dedup / DAKE / AutoShot)? Compare density (`compare_keyframes.py`) before committing.
- [ ] Semantic text channel: which encoder (multilingual sentence transformer vs CLIP text) for OCR/ASR/captions?
- [ ] Research note: survey temporal-memory / recurrent video captioning beyond ReCap
- [ ] When to start UI relative to second embedding / RRF
- [ ] SD / generative queries: skip for v1, or A/B as optional fused channel with verification?
