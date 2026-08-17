from __future__ import annotations

import numpy as np
import torch


def get_proper_device(name: str) -> torch.device:
    """Map cpu|gpu|cuda. GPU is opt-in: CUDA, else MPS, else error."""
    key = name.strip().lower()
    if key == "cpu":
        return torch.device("cpu")
    if key in ("gpu", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError(
            f"GPU requested (--device {name!r}) but neither CUDA nor MPS is available"
        )
    raise ValueError(f"Unknown device {name!r}; use cpu, gpu, or cuda")


def distance_to_interval(pts: float, start: float, end: float) -> float:
    """0 if inside [start, end]; else distance to nearest endpoint."""
    if start <= pts <= end:
        return 0.0
    if pts < start:
        return start - pts
    return pts - end


def asr_rows_for_clip_rows(
    clip_rows: np.ndarray | list[int], # input rows of index matrix
    clip_video_id: list[str], # i_th row of index matrix is from video_id[i]
    clip_pts: np.ndarray, # i_th row of index matrix is at pts_list[i] seconds
    asr_by_video: dict[str, tuple[int, np.ndarray, np.ndarray]], # map vid_name -> (start_row, list n of starts, list n of ends) in asr matrix
    delta: float = 3.0, # distance threshold in seconds
) -> list[np.ndarray]:
    """For each CLIP/FAISS row, global asr_embed_mat rows with interval distance < delta.

    Empty array if that keyframe has no ASR or no eligible segment.
    """
    n_clip = len(clip_video_id)
    out: list[np.ndarray] = []
    for raw in np.asarray(clip_rows).reshape(-1):
        clip_row = int(raw)
        assert clip_row >= 0 and clip_row < n_clip, \
            f"clip_row {clip_row} out of range {n_clip}"

        video_id = clip_video_id[clip_row]
        pts = float(clip_pts[clip_row])
        info = asr_by_video.get(video_id)
        if info is None:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        offset, starts, ends = info
        if starts.size == 0:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        hits = [
            offset + i
            for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist()))
            if distance_to_interval(pts, float(s), float(e)) < delta
        ]
        out.append(np.asarray(hits, dtype=np.int64))
    return out


def max_asr_scores(
    q_asr: np.ndarray,
    asr_mat: np.ndarray,
    row_lists: list[np.ndarray],
) -> np.ndarray:
    """Max cosine of q_asr vs asr_mat[rows] per candidate. Empty rows → 0."""
    q = np.asarray(q_asr, dtype=np.float32).reshape(-1)
    scores = np.zeros(len(row_lists), dtype=np.float32)
    for i, rows in enumerate(row_lists):
        if rows.size == 0:
            continue
        idx = np.asarray(rows, dtype=np.int64)
        scores[i] = float(np.max(asr_mat[idx] @ q))
    return scores


def minmax_norm(values: np.ndarray) -> np.ndarray:
    """Map to [0, 1] over the pool. All-equal -> zeros"""
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def mix_scores(
    s_visual: np.ndarray,
    r_asr: np.ndarray,
    w_visual: float,
    w_asr: float,
    r_ocr: np.ndarray | None = None,
    w_ocr: float = 0.0,
) -> np.ndarray:
    """Drop zero-weight channels; renorm remaining to 1; min-max then weighted sum."""
    n = np.asarray(s_visual, dtype=np.float32).reshape(-1).shape[0]
    acc = np.zeros(n, dtype=np.float32)
    parts: list[np.ndarray] = []
    weights: list[float] = []
    if w_visual > 0:
        parts.append(minmax_norm(s_visual))
        weights.append(float(w_visual))
    if w_asr > 0:
        parts.append(minmax_norm(r_asr))
        weights.append(float(w_asr))
    if w_ocr > 0 and r_ocr is not None:
        parts.append(minmax_norm(r_ocr))
        weights.append(float(w_ocr))
    if not parts:
        return acc
    wsum = float(sum(weights))
    for part, w in zip(parts, weights):
        acc += (w / wsum) * part # Normalize the weights to sum = 1
    return acc


def rrf_union(
    clip_ids: np.ndarray,
    sig_ids: np.ndarray,
    k: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    [2,3,5], [5,7,3] -> [2,3,5,7]
    then return ids [2,3,5,7] 
    and score of [candidates 2, candidates 3, candidates 5, candidates 7] 
    for example
    """
    scores: dict[int, float] = {}
    union: list[int] = []

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

