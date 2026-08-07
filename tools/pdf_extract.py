#!/usr/bin/env python3
"""Extract a PDF to a sibling Markdown file (UTF-8)."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def extract_pdf(pdf: Path) -> Path:
    if not pdf.is_file():
        raise SystemExit(f"Not a file: {pdf}")
    tmp = pdf.with_suffix(".md.tmp")
    out = pdf.with_suffix(".md")
    subprocess.check_call(
        ["pdftotext", "-enc", "UTF-8", "-layout", str(pdf), str(tmp)],
    )
    raw = tmp.read_bytes().decode("utf-8", errors="replace")
    raw = raw.replace("\x0c", "\n\n").replace("\x00", "")
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip() + "\n"
    title = pdf.stem.replace("_", " ")
    md = (
        f"# {title}\n\n"
        f"> Source: `{pdf.name}`  \n"
        f"> Extracted for easier reading. Formulas may need cleanup if PDF used special fonts.\n\n"
        f"---\n\n"
        f"{raw}"
    )
    out.write_text(md, encoding="utf-8")
    tmp.unlink(missing_ok=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf", type=Path, help="Path to PDF")
    args = p.parse_args()
    out = extract_pdf(args.pdf)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
