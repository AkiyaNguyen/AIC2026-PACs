# Prior-team paper summaries

Concise factual summaries of papers in this folder. Clarification tables only cover common misreadings.  
PACs design discussion belongs in [`docs/memory/DiscussionNotes.md`](../docs/memory/DiscussionNotes.md).

| System | Team | PDF | Full extract |
|--------|------|-----|----------------|
| [Vortex](#vortex--focusonfun-aic-2025) | FocusOnFun | [`Vortex_MultiModalFusion.pdf`](Vortex_MultiModalFusion.pdf) | [`Vortex_MultiModalFusion.md`](Vortex_MultiModalFusion.md) |
| [U-CESE](#u-cese--nomial-aic-2025) | Nomial | [`U-CESE.pdf`](U-CESE.pdf) | [`U-CESE.md`](U-CESE.md) |
| [NII-UIT (VBS2025)](#nii-uit--vbs-2025) | NII–UIT | Springer chapter (no local PDF yet) | [`NII-UIT_VBS2025.md`](NII-UIT_VBS2025.md) (§§2–3 transcribed) |

---

## Vortex — FocusOnFun (AIC 2025)

> arXiv:2606.19682

### What the system is

**Vortex** is an interactive multimodal video retrieval system for AIC 2025 (Textual KIS, Video KIS, Q&A, TRAKE). Reported prelim score **79.6/88**; finals described as strong overall, especially Q&A.

### Offline pipeline

1. **Keyframes:** AutoShot shot detection, then within-shot sampling (every 8th frame) + CLIP L2 relative-difference filter (threshold 0.4) to drop redundant frames.
2. **Metadata:** Qwen2.5-VL for OCR and captions; Whisper for timestamped ASR aligned onto keyframes.
3. **Embeddings:** CLIP (global) and SigLIP2 (fine-grained) image vectors → **Milvus**.
4. **Text index:** OCR / captions / ASR strings → **Elasticsearch**; **Redis** for result cache.

### Online pipeline

1. Text or image query embedded with **both** CLIP and SigLIP2.
2. Two independent ANN searches in Milvus → two ranked lists.
3. Lists merged by **Reciprocal Rank Fusion (RRF)** (\(k \approx 60\)).
4. Optional Elasticsearch filters (e.g. OCR keyword).
5. Optional **Rocchio** relevance feedback (like/dislike keyframes → update query vector → re-search).
6. Optional **temporal search** UI: three fields Previous / Current / Next (Algorithm 1).

### Temporal Algorithm 1 (as published)

- `Rprev`, `Rcur`, `Rnext` ← independent `Search` on each query.
- Per video: `bestPrev[vid]`, `bestNext[vid]` = max score in those lists.
- For each hit in `Rcur`: `S_final = S(cur) + bestPrev[vid] + bestNext[vid]` (missing side → 0).
- Re-sort `Rcur` by `S_final`. Output is a re-ranked **Current** list.

### Clarifications (common reading mistakes)

| Misreading | What the paper actually does |
|------------|------------------------------|
| OCR, ASR, captions are all embedded and RRF’d together with CLIP/SigLIP | RRF fuses **CLIP ranks vs SigLIP2 ranks**. OCR/ASR/captions are mainly **Elasticsearch text/filter** channels. |
| Temporal Alg. 1 enforces `time(prev) < time(cur) < time(next)` | It only requires the **same `video_id`** to have strong prev/next hits somewhere in top-K. **No** timestamp order check in Alg. 1. |
| Temporal output = full TRAKE submission `(f1,…,fn)` | Alg. 1 ranks **Current** candidates. TRAKE still needs **one frame per event**; the algorithm is a retrieval/UI ranking heuristic, not the TRAKE answer format. |
| “Current” is privileged because TRAKE only submits one frame | “Current” is the **UI ranking pivot** (Before/Now/After). Submission format for TRAKE remains multi-frame. |

### Stack (as stated)

Milvus, Elasticsearch, Redis; CLIP + SigLIP2; Qwen2.5-VL; Whisper; AutoShot; interactive web UI.

---

## U-CESE — Nomial (AIC 2025)

> arXiv:2605.23274

### What the system is

**U-CESE** (Unified Clip-based Event Search Engine) extends **CESE** by merging CESE’s three separate modality modules (visual / textual semantic / keywords) into **one** retrieval and ranking pipeline. Emphasis: return **clips (suggestions)** that cover multiple event descriptions, not only single frames. Finals reported as strong on **TRAKE** and **QA**.

### Offline pipeline

1. **DAKE (Dynamic-Aware Keyframe Extraction):** Training-free keyframes from **JPEG file-size steepness** (motion/transition proxy); select top fraction \(\rho\) of frames (paper uses \(\rho \approx 0.02\)), with denser guarantees in windows. Compared against AutoShot in ablations.
2. **ASR:** Whisper → timestamped subtitles.
3. **Captioning:** Per-keyframe / shot captions with **Gemini**; **ReCap** adds **recurrent memory** across shots (RNN-style \(M_{t-1} \rightarrow (C_t, M_t)\)) for temporal consistency.
4. **Indexes:**
   - **VisualDB (Milvus):** MobileCLIP **image** embeddings of keyframes.
   - **TextualDB:** caption/ASR as MobileCLIP **text** embeddings (Milvus) + **raw text** (Elasticsearch).

### Online pipeline — Unified Clipping (Algorithm 2)

1. User may issue **multiple queries** \(q_1,\ldots,q_n\) (each can use frame emb / text emb / raw text flags).
2. **RetrieveAll:** pull hits from VisualDB + TextualDB embedding + TextualDB raw; flatten; sort by `(video, timestamp)`.
3. **Per video:** two-pointer windows with `end − start ≤ T` → each window is a **suggestion** (clip) holding the retrieved frames inside it.
4. **Rank suggestions:** prefer more **unique query IDs covered**; tie-break by max per-frame score.
5. Return top-K suggestions to a **single** web UI (viewer, video player, TRAKE helpers such as Tab to append frame index).

### Clarifications (common reading mistakes)

| Misreading | What the paper actually does |
|------------|------------------------------|
| Same as Vortex: rank keyframes then RRF CLIP+SigLIP | Primary unit is a **time-bounded clip/suggestion**. Fusion is **multi-channel retrieve + cover-count ranking**, not Vortex-style RRF of two VLMs. |
| One text query → one best frame is the whole story | Designed for **several scene/event queries at once**; a good clip is one that matches **more distinct queries** inside window \(T\). |
| ReCap replaces ASR | Whisper still produces subtitles; ReCap uses keyframes + subtitles (+ memory) to write **richer captions** for text search. |
| DAKE replaces all organizer keyframes by definition | DAKE is **their** extraction method for building their DB; contest still scores on official video frame indices. |
| Unified Clipping alone outputs finished TRAKE answers | It surfaces candidate **clips and frames**; UI/workflow still used to pick exact ordered semantic frames (paper highlights TRAKE UX shortcuts). |

### Stack (as stated)

DAKE, AutoShot (shots for ReCap), Whisper, Gemini (caption/ReCap), MobileCLIP, Milvus, Elasticsearch, unified web UI (Python/HTML/JS).

---

## NII-UIT — VBS 2025

> DOI: [10.1007/978-981-96-2074-6_38](https://doi.org/10.1007/978-981-96-2074-6_38) · MMM 2025 LNCS 15524, pp. 318–325  
> Extract: [`NII-UIT_VBS2025.md`](NII-UIT_VBS2025.md) (methods §§2–3 from team-supplied text)

### What the system is

Interactive multimodal video retrieval for **VBS 2025**. Team **NII–UIT** (UIT + NII); overall/expert winner. Offline indexes keyframes; online fuses several query channels, optionally expands text with an LLM / SD, filters by objects, and supports multi-stage **dynamic temporal** search.

### Offline pipeline

1. Keyframe selection (Vibro-inspired + **BEiT-3** every 10th frame; keep semantically different frames; store as **WebP**).
2. VLM **feature vectors** → **Milvus**.
3. **Co-DETR** object detections (COCO) → tabular DB.

### Online pipeline

1. Inputs: text and/or image (+ optional object constraints; Q/A → text description via LLM).
2. Optional **GPT-4o** paraphrases (≈5); optional **Stable Diffusion** text→image queries; run channels in parallel.
3. Per-model / per-channel ranks → **normalize** → **mean-pool fusion**.
4. **Object filter** drops shots missing required objects; rerank to UI.
5. **Temporal / multi-stage:** chain stages; for ambiguous KIS-T context, score **neighboring shots** around a hit (vitrivr-inspired) and aggregate scores across stages — not only “must be before/after.”

### Clarifications (common reading mistakes)

| Misreading | What the paper actually does |
|------------|------------------------------|
| LLM does the ANN search | LLM **paraphrases / rewrites** queries (and Q/A→description); ANN is VLM embeddings in Milvus. |
| SD builds the gallery | SD is **query-time** visual generation only. |
| Fusion = Vortex RRF of CLIP+SigLIP | They **normalize + mean-pool** scores across textual / visual / paraphrase / SD channels (exact VLM set described as “advanced VLMs”; cites OpenCLIP, CLIP2Video, ALADIN, BEiT-3, InternVL-G as field context). |
| Temporal = fixed Before/Now/After boost on current list | Multi-stage search; neighbor-shot exploration around initial hits + score aggregation (inspired by vitrivr), aimed at unclear before/after KIS-T wording. |
| Object DB is an embedding channel in the mean pool | Objects are a **post-fusion filter** (Co-DETR tabular constraints), not another fused embedding list. |

### Stack (as stated)

BEiT-3 (keyframe selection features); Milvus; Co-DETR; GPT-4o; Stable Diffusion; multiple VLMs + normalize/mean-pool fusion; web UI with Advanced Mode (weights / paraphrase / SD).

---

## Contrast snapshot

| | U-CESE | Vortex | NII-UIT (VBS2025) |
|--|--------|--------|------------------|
| Rank unit | Clip covering many queries | Keyframe (RRF dual VLM) | Shot/keyframe after mean-pool fusion + object filter |
| Temporal idea | Window \(T\) + #queries covered | Before/Now/After score boost on Now | Multi-stage + neighbor-shot re-score (vitrivr-inspired) |
| Extra VLMs | MobileCLIP (+ text DBs) | CLIP + SigLIP2 RRF | Multi-VLM fusion; LLM/SD at query time |
| Caption / text | ReCap + Gemini memory | Qwen2.5-VL OCR/caption | GPT-4o paraphrase; Q/A→text; SD visual queries |
| Objects | (not central in our notes) | ES filters | Co-DETR post-filter |
