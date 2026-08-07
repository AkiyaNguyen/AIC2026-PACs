---
name: project-memory
description: >-
  Maintain DiscussionNotes and prior-team SUMMARIES; when discussing methods or
  system design, cite local notes/code then search web/arXiv for similar
  techniques before proposing new ideas. Use for retrieval architecture,
  prior papers, design debates, or remember/recall of contest approach.
---

# Project memory

## Where things live

| Path | Purpose |
|------|---------|
| [`docs/memory/DiscussionNotes.md`](../../../docs/memory/DiscussionNotes.md) | **Our** ideas: baseline, debates, decisions (human↔agent) |
| [`PreviousTeamSubmission/SUMMARIES.md`](../../../PreviousTeamSubmission/SUMMARIES.md) | Concise prior-team paper summaries + misreading clarifications |
| [`PreviousTeamSubmission/*.md`](../../../PreviousTeamSubmission/) | Full PDF extracts |
| Repo code (e.g. `tools/kis_search.py`) | What we already implemented |

## When to read

- Architecture / method proposals → `DiscussionNotes.md` + `SUMMARIES.md` first.
- “What did Vortex/U-CESE do?” → `SUMMARIES.md`.

## Anti-reinvention (methods / system design)

Before proposing a new approach:

1. **Local:** Cite relevant sections of `SUMMARIES.md`, `DiscussionNotes.md`, and existing code. Say what is reused vs missing.
2. **Web/arXiv:** WebSearch for similar techniques (e.g. “temporal memory video captioning”, “multilingual OCR semantic retrieval”, “reciprocal rank fusion video retrieval”). Cite 1–3 hits if they exist.
3. **Reply structure:** existing (local + web) → reuse → only then PACs-specific novelty.

Do not claim an idea is original without this check. Search is not exhaustive; prefer named prior art over novelty claims. Citation + abstract is enough when full PDF is paywalled.

## When to write

- Refined **our** design → update `DiscussionNotes.md`.
- New prior-team PDF studied → append/update a section in `SUMMARIES.md` (factual; clarifications table only for common misreadings). Do not put PACs roadmap into paper summaries.

## Style

- English for repo-authored notes.
- Short and actionable; link instead of pasting long paper text.
