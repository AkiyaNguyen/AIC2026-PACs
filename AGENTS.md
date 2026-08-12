# AGENTS.md — PACs / AIC 2026

Hard constraints for coding agents working in this repository.

## Project

- Team **PACs**, AI Challenge (AIC) 2026 — video retrieval / VQA / temporal alignment.
- Official problem and scoring text: `docs/organization-board/`. Do **not** rewrite BTC original PDFs or Vietnamese source wording. Parsed `.md` copies may receive formula cleanups for readability only.

## Language

- Repo-authored docs, comments, and commit messages: **English**.
- Organization Board source text may remain Vietnamese.

## Secrets and bulky artifacts

- Do not commit secrets (`.env`, credentials, API keys).
- Do not commit bulky binary artifacts (videos, raw keyframe trees, large `.npy` / index dumps) unless the team explicitly asks.

## Data layout (undecided)

- Storage and indexing host are **not decided** yet.
- Do **not** invent a data-layout design in docs or code comments (local paths, Kaggle-only, streaming-only, etc.).
- Ask before assuming where media, features, or indexes live.

## Contest submissions

- **Textual KIS:** `(video_id, frame_id)`
- **Q&A:** `(video_id, frame_id, answer)`
- **TRAKE:** `(video_id, frame_id1, …, frame_idn)`
- Up to **100** ranked answers per query; ranking order matters for Final Score (`R@1/5/20/50/100`).

## Scope discipline

- Prefer a working **Textual KIS** loop before Q&A or TRAKE complexity, unless the user asks otherwise.
- Do not invent ground-truth frame intervals or claim scores without an evaluation harness.
- Put CLI utilities under `tools/` (not a top-level `scripts/` directory). On Windows, `scripts` and venv `Scripts` are the same path.
- Create the virtualenv only as `.venv` (`python -m venv .venv`), never as `python -m venv .` in the repo root.

## Memory and PDFs

- **Our** method discussion: `docs/memory/DiscussionNotes.md`.
- Prior-team papers: `PreviousTeamSubmission/` — PDF + full `.md` extract; all paper summaries in `PreviousTeamSubmission/SUMMARIES.md`.
- When reading a PDF: extract to **Markdown beside the PDF** (skill `pdf-extract`); prefer `.md` over `.txt`. For studied prior-team papers, append/update a section in `SUMMARIES.md`.

## Collaboration (human is learning the stack)

- Treat the human as a **teammate who wants to understand**, not only a requester. When a concept is load-bearing (tests, embeddings vs extract, GPU opt-in, package vs script), **explain it in the reply** instead of only implementing.
- **Follow their idea and existing code patterns** when that is a valid way to ship.
- **Do not blindly implement a suggestion** if a common practice is clearly better (safer CLI, less duplicate GPU work, clearer layout, etc.). **Notice them, recommend the practice, and wait for agreement** before producing the final solution.
- After a collaboration or design rule is settled, record it in `docs/memory/DiscussionNotes.md`.

## Anti-reinvention (methods / system design)

When discussing methods or system design, **before** proposing something as new:

1. Cite local prior art: `PreviousTeamSubmission/SUMMARIES.md`, `docs/memory/DiscussionNotes.md`, and existing repo code (e.g. `tools/kis_search.py`).
2. Run a short **web/arXiv search** for similar techniques; cite 1–3 known approaches if found.
3. State what exists → what to reuse → what would be new for PACs.

Search is not exhaustive; do not claim novelty without this check; citation + abstract is enough when PDFs are paywalled.