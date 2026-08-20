"""Resolve bilingual KIS queries: CLIP uses EN; SigLIP/ASR use VI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from deep_translator import GoogleTranslator


class QueryTranslationError(Exception):
    """Raised when VI→EN machine translation fails
       app.py would catch this kind of exception and return a 503 error code.
        """


@dataclass(frozen=True)
class ResolvedQueries:
    query_vi: str
    query_en: str
    query_en_source: Literal["user", "translated"]


def translate_vi_to_en(text: str) -> str:
    """Translate Vietnamese text to English via deep-translator (Google)."""
    try:
        out = GoogleTranslator(source="vi", target="en").translate(text)
    except Exception as exc:  # noqa: BLE001 — wrap any MT/network failure
        raise QueryTranslationError(f"VI→EN translation failed: {exc}") from exc
    if not out or not str(out).strip():
        raise QueryTranslationError("VI→EN translation returned an empty string")
    return str(out).strip()


def resolve_queries(query_vi: str, query_en: str | None = None) -> ResolvedQueries:
    """Require Vietnamese; fill English from MT when omitted or blank."""
    vi = query_vi.strip()
    if not vi:
        raise ValueError("query_vi must be non-empty")

    en = (query_en or "").strip()
    if en:
        return ResolvedQueries(query_vi=vi, query_en=en, query_en_source="user")

    return ResolvedQueries(
        query_vi=vi,
        query_en=translate_vi_to_en(vi),
        query_en_source="translated",
    )
