"""Score normalization, fusion helpers, and channel aggregators."""

from __future__ import annotations

import numpy as np


def max_asr_scores(
    q_asr: np.ndarray,
    asr_mat: np.ndarray,
    row_lists: list[np.ndarray],
) -> np.ndarray:
    q = np.asarray(q_asr, dtype=np.float32).reshape(-1)
    scores = np.zeros(len(row_lists), dtype=np.float32)
    for i, rows in enumerate(row_lists):
        if rows.size == 0:
            continue
        idx = np.asarray(rows, dtype=np.int64)
        scores[i] = float(np.max(asr_mat[idx] @ q))
    return scores


def max_bm25_scores(
    bm25_doc_scores: np.ndarray,
    bm25_global_rows: np.ndarray,
    row_lists: list[np.ndarray],
) -> np.ndarray:
    by_global: dict[int, float] = {}
    for i, grow in enumerate(bm25_global_rows):
        sc = float(bm25_doc_scores[i])
        if sc <= 0:
            continue
        g = int(grow)
        by_global[g] = max(by_global.get(g, 0.0), sc)

    out = np.zeros(len(row_lists), dtype=np.float32)
    for i, rows in enumerate(row_lists):
        if rows.size == 0:
            continue
        out[i] = max((by_global.get(int(r), 0.0) for r in rows), default=0.0)
    return out


def minmax_norm(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def combine_text_scores(
    r_sem: np.ndarray,
    r_bm25: np.ndarray,
    weight_sem_text: float = 0.6,
    weight_bm25: float = 0.4,
) -> np.ndarray:
    """Convex combination of min-max semantic cosine and BM25 on the pool."""
    n = np.asarray(r_sem, dtype=np.float32).reshape(-1).shape[0]
    acc = np.zeros(n, dtype=np.float32)
    parts: list[np.ndarray] = []
    weights: list[float] = []
    if weight_sem_text > 0:
        parts.append(minmax_norm(r_sem))
        weights.append(float(weight_sem_text))
    if weight_bm25 > 0:
        parts.append(minmax_norm(r_bm25))
        weights.append(float(weight_bm25))
    if not parts:
        return acc
    wsum = float(sum(weights))
    for part, w in zip(parts, weights):
        acc += (w / wsum) * part
    return acc


def mix_scores(
    s_visual: np.ndarray,
    r_asr: np.ndarray,
    weight_visual: float,
    weight_transcript: float,
    r_ocr: np.ndarray | None = None,
    w_ocr: float = 0.0,
) -> np.ndarray:
    n = np.asarray(s_visual, dtype=np.float32).reshape(-1).shape[0]
    acc = np.zeros(n, dtype=np.float32)
    parts: list[np.ndarray] = []
    weights: list[float] = []
    if weight_visual > 0:
        parts.append(minmax_norm(s_visual))
        weights.append(float(weight_visual))
    if weight_transcript > 0:
        parts.append(minmax_norm(r_asr))
        weights.append(float(weight_transcript))
    if w_ocr > 0 and r_ocr is not None:
        parts.append(minmax_norm(r_ocr))
        weights.append(float(w_ocr))
    if not parts:
        return acc
    wsum = float(sum(weights))
    for part, w in zip(parts, weights):
        acc += (w / wsum) * part
    return acc


def rrf_union(
    clip_ids: np.ndarray,
    sig_ids: np.ndarray,
    k: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    scores: dict[int, float] = {}
    union: list[int] = []
    """
    Return the union of clip_ids and sig_ids, and the RRF scores of the union keyframes
    """
    def add(ids: np.ndarray) -> None:
        seen: set[int] = set()
        for rank, row in enumerate(np.asarray(ids, dtype=np.int64).reshape(-1), start=1):
            row = int(row)
            if row < 0 or row in seen:
                continue
            seen.add(row)
            if row not in scores:
                union.append(row)
                scores[row] = 0.0
            scores[row] += 1.0 / (k + rank)

    add(clip_ids)
    add(sig_ids)
    out = np.asarray(union, dtype=np.int64)
    s = np.empty(len(union), dtype=np.float32)
    for i, row in enumerate(union):
        s[i] = scores[row]
    return out, s
