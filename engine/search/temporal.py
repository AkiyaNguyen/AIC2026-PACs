"""Temporal alignment between keyframes and ASR segments."""

from __future__ import annotations

import numpy as np


def distance_to_interval(pts: float, start: float, end: float) -> float:
    """0 if inside [start, end]; else distance to nearest endpoint."""
    if start <= pts <= end:
        return 0.0
    if pts < start:
        return start - pts
    return pts - end


def build_video_timeline(
    video_ids: list[str],
    pts_list: np.ndarray,
) -> dict[str, list[tuple[int, float]]]:
    """Map video_id -> [(clip_row, pts_time), ...] in gallery row order."""
    out: dict[str, list[tuple[int, float]]] = {}
    for row_i, vid in enumerate(video_ids):
        pts = float(np.asarray(pts_list).reshape(-1)[row_i])
        out.setdefault(vid, []).append((row_i, pts))
    return out


def keyframes_for_segment(
    video_id: str,
    start: float,
    end: float,
    timeline: dict[str, list[tuple[int, float]]],
    *,
    min_gap: float,
    pad: float,
) -> list[int]:
    """Sample keyframes in [start - pad, end + pad], at least min_gap apart."""
    lo, hi = start - pad, end + pad
    picked: list[int] = []
    last_pts: float | None = None
    for row_i, pts in timeline.get(video_id, []):
        if pts < lo or pts > hi:
            continue
        if last_pts is None or (pts - last_pts) >= min_gap:
            picked.append(row_i)
            last_pts = pts
    return picked


def global_row_to_segment(
    asr_by_video: dict[str, tuple[int, np.ndarray, np.ndarray]],
    global_row: int,
) -> tuple[str, float, float] | None:
    for vid, (offset, starts, ends) in asr_by_video.items():
        n = len(starts)
        if offset <= global_row < offset + n:
            i = global_row - offset
            return vid, float(starts[i]), float(ends[i])
    return None


def asr_rows_for_clip_rows(
    clip_rows: np.ndarray | list[int],
    clip_video_id: list[str],
    clip_pts: np.ndarray,
    asr_by_video: dict[str, tuple[int, np.ndarray, np.ndarray]],
    delta: float = 1.0,
) -> list[np.ndarray]:
    """For each keyframe row, global asr_embed_mat rows within delta of segment."""
    n_clip = len(clip_video_id)
    out: list[np.ndarray] = []
    for raw in np.asarray(clip_rows).reshape(-1):
        clip_row = int(raw)
        if clip_row < 0 or clip_row >= n_clip:
            raise IndexError(f"clip_row {clip_row} out of range {n_clip}")

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
