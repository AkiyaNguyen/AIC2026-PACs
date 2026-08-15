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


# def minmax_norm(values: np.ndarray) -> np.ndarray:
#     """Map to [0, 1] over the pool. All-equal → zeros (no fake spread)."""
#     raise NotImplementedError


# def mix_scores(
#     s_visual: np.ndarray,
#     r_asr: np.ndarray,
#     r_ocr: np.ndarray | None,
#     w_visual: float,
#     w_asr: float,
#     w_ocr: float = 0.0,
# ) -> np.ndarray:
#     """Renorm weights to sum 1; weighted sum of min-max'd channels."""
#     raise NotImplementedError

