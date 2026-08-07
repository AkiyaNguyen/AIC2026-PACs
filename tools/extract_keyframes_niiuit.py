#!/usr/bin/env python3
"""Extract keyframes with an NII-UIT / Vibro-inspired pipeline.

Algorithm (detailed)
--------------------
NII-UIT (VBS 2025) describes keyframe selection as:

  1. Do **not** decode/embed every frame (too expensive at VBS/AIC scale).
  2. Sample a fixed temporal grid: every ``stride``-th frame (paper: every 10th).
  3. Embed each candidate with a vision encoder (paper: BEiT-3; here: clip /
     siglip / beit3).
  4. Keep a candidate only if it is **semantically different enough** from the
     last kept keyframe (cosine distance on L2-normalized embeddings).
  5. Store kept frames as WebP (paper) and write a map CSV (frame_idx, time).

This script implements that selection rule. It does **not** run TransNet /
AutoShot shot-boundary detection for the keep/drop decision (same spirit as
NII-UIT's published preprocess description). Shots are optional future UI
structure, not required here.

Parameters that matter
----------------------
- ``--stride`` (default 10): temporal grid. Larger → cheaper, sparser.
- ``--min-cosine-distance`` (default 0.15): keep if ``1 - cos(last, cur)`` is
  at least this. Higher → fewer, more distinct keyframes. Lower → denser.
- ``--model``: ``clip`` | ``siglip`` | ``beit3`` (run all three on a small
  video list to compare).

Output tree (one video folder per input)
----------------------------------------
::

    OUT_DIR/
      VIDEO_ID/
        000000.webp
        000001.webp
        ...
        map.csv          # n,pts_time,fps,frame_idx  (BTC-like columns)
      run_meta.json      # models, thresholds, source paths

Contest density note
--------------------
BTC TRAKE answer windows are typically **under ~10 frames**. If consecutive
kept keyframes are often >> 10 frames apart, a TRAKE moment can fall between
indexed frames (you must still seek the raw video / denser extract). Use
``tools/compare_keyframes.py`` to inspect gap distributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from keyframe_models import MODEL_CHOICES, cosine_distance, load_embedder  # noqa: E402


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


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
        # directory: videos directly inside, or nested one level
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
    # de-dupe preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for v in found:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def video_id_from_path(path: Path) -> str:
    return path.stem


def select_keyframes(
    embeddings: np.ndarray,
    frame_indices: list[int],
    min_cosine_distance: float,
) -> list[int]:
    """Return indices into the candidate arrays of frames to keep."""
    if len(embeddings) == 0:
        return []
    keep = [0]
    last = embeddings[0]
    for i in range(1, len(embeddings)):
        if cosine_distance(last, embeddings[i]) >= min_cosine_distance:
            keep.append(i)
            last = embeddings[i]
    return keep


def extract_one_video(
    video_path: Path,
    out_root: Path,
    embedder,
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

    # --- Stage A: sample every `stride`-th frame ---
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

    # --- Stage B: embed in batches ---
    emb_chunks: list[np.ndarray] = []
    for start in range(0, len(candidates), batch_size):
        chunk = [im for _, im in candidates[start : start + batch_size]]
        emb_chunks.append(embedder.embed_pils(chunk))
    embeddings = np.concatenate(emb_chunks, axis=0)
    frame_indices = [fi for fi, _ in candidates]

    # --- Stage C: keep if semantically different from last kept ---
    keep_pos = select_keyframes(embeddings, frame_indices, min_cosine_distance)

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

    # also dump embeddings of kept frames for compare script (optional small)
    kept_emb = embeddings[keep_pos]
    np.save(out_dir / "embeddings.npy", kept_emb)

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "NII-UIT-style keyframe extraction: every-N frames + semantic "
            "dedup (clip | siglip | beit3)."
        )
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Video file(s) and/or folder(s) containing videos",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output root (creates VIDEO_ID/ subfolders)",
    )
    p.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="clip",
        help="Difference encoder (default: clip)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sample every Nth frame before embedding (NII-UIT: 10)",
    )
    p.add_argument(
        "--min-cosine-distance",
        type=float,
        default=0.15,
        help="Keep frame if 1-cos(last,cur) >= this (default: 0.15)",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--webp-quality", type=int, default=80)
    p.add_argument("--device", default="cuda" if _cuda_ok() else "cpu")
    p.add_argument(
        "--list-file",
        type=Path,
        default=None,
        help="Optional text file: one video path or folder per line (# comments ok)",
    )
    return p.parse_args(argv)


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_list_file(path: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[Path] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(Path(s))
    return out


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args(argv)
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if not (0.0 <= args.min_cosine_distance <= 2.0):
        raise SystemExit("--min-cosine-distance should be in [0, 2]")

    inputs = list(args.inputs)
    if args.list_file:
        inputs.extend(load_list_file(args.list_file))

    videos = discover_videos(inputs)
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    import torch

    device = torch.device(args.device)
    safe_print(f"Loading embedder model={args.model} device={device} ...")
    embedder = load_embedder(args.model, device)

    summaries = []
    for vp in videos:
        summaries.append(
            extract_one_video(
                vp,
                out_root,
                embedder,
                stride=args.stride,
                min_cosine_distance=args.min_cosine_distance,
                batch_size=args.batch_size,
                webp_quality=args.webp_quality,
                device_label=str(device),
            )
        )

    meta = {
        "algorithm": "niiuit_stride_semantic_dedup",
        "model": args.model,
        "stride": args.stride,
        "min_cosine_distance": args.min_cosine_distance,
        "batch_size": args.batch_size,
        "webp_quality": args.webp_quality,
        "device": str(device),
        "n_videos": len(summaries),
        "total_kept": int(sum(s["kept"] for s in summaries)),
        "videos": summaries,
    }
    meta_path = out_root / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    safe_print(f"Wrote {meta_path}")
    safe_print(f"Done: {len(summaries)} video(s), {meta['total_kept']} keyframes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
