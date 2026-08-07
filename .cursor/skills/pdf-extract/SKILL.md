---
name: pdf-extract
description: >-
  Extract text from PDF files into Markdown (preferred) or plain text beside the
  source PDF. Use whenever the user asks to read, summarize, or analyze a PDF,
  or when working with papers under PreviousTeamSubmission, docs, or any *.pdf
  in this repo.
---

# PDF extract (same folder as source)

## When this applies

Any time you need to **read or discuss a PDF** in this project:

1. Prefer an existing `.md` extract next to the PDF if it is already present and up to date.
2. Otherwise **extract before deep reading**, then use the extract for analysis.

## Output location and naming

- Put the extract **in the same directory as the PDF**.
- Prefer **Markdown** (`.md`). Use `.txt` only if Markdown conversion fails badly.
- Naming: `<pdf-stem>.md`  
  Examples:
  - `PreviousTeamSubmission/U-CESE.pdf` → `PreviousTeamSubmission/U-CESE.md`
  - `docs/organization-board/foo.pdf` → `docs/organization-board/foo.md`

Do **not** leave one-off extracts in unrelated folders (e.g. do not dump into `docs/` root when the PDF lives under `PreviousTeamSubmission/`).

## Extraction steps

1. Locate the PDF path (absolute or repo-relative).
2. Run extraction with UTF-8 layout when possible:

```bash
pdftotext -enc UTF-8 -layout "PATH/TO/file.pdf" "PATH/TO/file.md.tmp"
```

3. Wrap into Markdown with a short English header:

```markdown
# <Document title or PDF stem>

> Source: `<original-filename.pdf>`  
> Extracted for easier reading. Formulas may need cleanup if PDF used special fonts.

---

<extracted body>
```

4. Save as `PATH/TO/file.md` (replace `.md.tmp`).
5. If math glyphs are mangled, clean critical formulas into readable ASCII/LaTeX blocks (same practice as organization-board parses).
6. Prefer reading/citing the `.md` afterward; keep the PDF as the canonical binary.
7. For **prior-team papers** under `PreviousTeamSubmission/`, append or update a section in `SUMMARIES.md` (concise factual summary + clarifications table for common misreadings). Do not put PACs roadmap into paper summaries — that belongs in `docs/memory/DiscussionNotes.md`.

Optional helper:

```bash
.venv/Scripts/python tools/pdf_extract.py "PreviousTeamSubmission/SomePaper.pdf"
```

## Language

- Repo-authored headers/notes: **English**.
- Body text: keep the PDF’s original language (Vietnamese or English).

## Do not

- Overwrite a carefully cleaned `.md` without checking with the user if they edited it.
- Commit huge binary intermediates.
- Use a top-level folder named `scripts/` for helper tools (Windows clash with venv `Scripts/`). Optional helpers belong under `tools/`.
