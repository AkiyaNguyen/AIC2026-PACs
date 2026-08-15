"""NII-UIT-style extract: stride sample + CLIP cosine keep/drop."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from tools.extract_features.engine.clip import ClipEmbedder

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity for L2-normalized vectors."""
    return float(1.0 - np.dot(a, b))


@dataclass
class KeptFrame:
    n: int
    frame_idx: int
    pts_time: float
    fps: float
    path: str


def safe_print(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def select_keyframes(embeddings: np.ndarray, min_cosine_distance: float) -> list[int]:
    """Return indices of candidates to keep (compare to last kept only)."""
    if len(embeddings) == 0:
        return []
    keep = [0]
    last = embeddings[0]
    for i in range(1, len(embeddings)):
        if cosine_distance(last, embeddings[i]) >= min_cosine_distance:
            keep.append(i)
            last = embeddings[i]
    return keep


def discover_videos(inputs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for raw in inputs:
        p = raw.expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"Input path not found: {p}")
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTS:
                found.append(p)
            else:
                raise SystemExit(f"Not a video file: {p}")
            continue
        direct = sorted(
            q for q in p.iterdir() if q.is_file() and q.suffix.lower() in VIDEO_EXTS
        )
        if direct:
            found.extend(direct)
            continue
        nested = sorted(
            q
            for q in p.rglob("*")
            if q.is_file() and q.suffix.lower() in VIDEO_EXTS
        )
        if not nested:
            raise SystemExit(f"No videos under {p}")
        found.extend(nested)
    seen: set[Path] = set()
    out: list[Path] = []
    for v in found:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def video_id_from_path(path: Path) -> str:
    return path.stem


def load_list_file(path: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[Path] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(Path(s))
    return out


def extract_one_video(
    video_path: Path,
    out_root: Path,
    embedder: ClipEmbedder,
    *,
    stride: int,
    min_cosine_distance: float,
    batch_size: int,
    webp_quality: int,
    device_label: str,
) -> dict:
    import cv2

    vid = video_id_from_path(video_path)
    out_dir = out_root / vid
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    candidates: list[tuple[int, Image.Image]] = []
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            candidates.append((frame_idx, Image.fromarray(rgb)))
        frame_idx += 1
    cap.release()

    if not candidates:
        raise SystemExit(f"No frames decoded from {video_path}")

    emb_chunks: list[np.ndarray] = []
    for start in range(0, len(candidates), batch_size):
        chunk = [im for _, im in candidates[start : start + batch_size]]
        emb_chunks.append(embedder.embed_pils(chunk))
    embeddings = np.concatenate(emb_chunks, axis=0)

    keep_pos = select_keyframes(embeddings, min_cosine_distance)

    kept: list[KeptFrame] = []
    for n, pos in enumerate(keep_pos, start=1):
        fi, im = candidates[pos]
        fname = f"{n - 1:06d}.webp"
        fpath = out_dir / fname
        im.save(fpath, format="WEBP", quality=webp_quality, method=4)
        kept.append(
            KeptFrame(
                n=n,
                frame_idx=fi,
                pts_time=fi / fps,
                fps=fps,
                path=str(fpath.relative_to(out_root)).replace("\\", "/"),
            )
        )

    map_path = out_dir / "map.csv"
    with map_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        w.writeheader()
        for row in kept:
            w.writerow(
                {
                    "n": row.n,
                    "pts_time": f"{row.pts_time:.6f}",
                    "fps": row.fps,
                    "frame_idx": row.frame_idx,
                }
            )

    np.save(out_dir / "embeddings.npy", embeddings[keep_pos])

    gaps = np.diff([k.frame_idx for k in kept]) if len(kept) > 1 else np.array([])
    summary = {
        "video_id": vid,
        "source": str(video_path),
        "fps": fps,
        "decoded_frame_count": frame_idx,
        "reported_frame_count": total,
        "candidates": len(candidates),
        "kept": len(kept),
        "stride": stride,
        "min_cosine_distance": min_cosine_distance,
        "model": embedder.name,
        "device": device_label,
        "mean_gap_frames": float(gaps.mean()) if gaps.size else None,
        "max_gap_frames": int(gaps.max()) if gaps.size else None,
        "map_csv": str(map_path.relative_to(out_root)).replace("\\", "/"),
    }
    safe_print(
        f"[{vid}] candidates={len(candidates)} kept={len(kept)} "
        f"mean_gap={summary['mean_gap_frames']} max_gap={summary['max_gap_frames']}"
    )
    return summary


def run_extract(
    inputs: list[Path],
    out_dir: Path,
    embedder: ClipEmbedder,
    *,
    stride: int,
    min_cosine_distance: float,
    batch_size: int,
    webp_quality: int,
    device_label: str,
) -> dict:
    videos = discover_videos(inputs)
    out_root = out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        extract_one_video(
            vp,
            out_root,
            embedder,
            stride=stride,
            min_cosine_distance=min_cosine_distance,
            batch_size=batch_size,
            webp_quality=webp_quality,
            device_label=device_label,
        )
        for vp in videos
    ]
    meta = {
        "algorithm": "niiuit_stride_semantic_dedup",
        "model": embedder.name,
        "stride": stride,
        "min_cosine_distance": min_cosine_distance,
        "batch_size": batch_size,
        "webp_quality": webp_quality,
        "device": device_label,
        "n_videos": len(summaries),
        "total_kept": int(sum(s["kept"] for s in summaries)),
        "videos": summaries,
    }
    meta_path = out_root / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    safe_print(f"Wrote {meta_path}")
    safe_print(f"Done: {len(summaries)} video(s), {meta['total_kept']} keyframes")
    return meta
