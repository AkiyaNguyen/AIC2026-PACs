#!/usr/bin/env python3
"""Textual KIS baseline: CLIP text query over merged keyframe features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import open_clip
import torch


@dataclass(frozen=True)
class KeyframeRef:
    video_id: str
    keyframe_index: int  # 0-based row in that video's .npy / map CSV
    frame_idx: int
    pts_time: float
    fps: float
    title: str
    watch_url: str
    moment_url: str  # watch URL with t= seek (usable in a browser)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_queries(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    queries = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        queries.append(s)
    if not queries:
        raise SystemExit(f"No queries found in {path}")
    return queries


def safe_print(text: str = "") -> None:
    """Print with UTF-8 when possible; avoid Windows cp1252 crashes on titles."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))
        sys.stdout.buffer.flush()


def youtube_moment_url(watch_url: str, pts_time: float) -> str:
    """Open the normal YouTube watch page already seeked to pts_time.

    Embed URLs often fail (embedding disabled / blocked outside an iframe).
    A watch URL with ``t=`` is reliable; the browser usually waits for a click
    before playing (autoplay policy), which is close to “paused at that moment”.
    """
    seconds = max(0, int(round(float(pts_time))))
    parsed = urlparse(watch_url.strip())
    qs = parse_qs(parsed.query)
    video_id = None
    if "v" in qs and qs["v"]:
        video_id = qs["v"][0]
    elif "youtu.be" in (parsed.netloc or ""):
        video_id = parsed.path.lstrip("/").split("/")[0] or None

    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"

    sep = "&" if parsed.query else "?"
    return f"{watch_url}{sep}t={seconds}s"


def load_media_info(media_dir: Path) -> dict[str, dict]:
    info: dict[str, dict] = {}
    if not media_dir.is_dir():
        return info
    for path in media_dir.glob("*.json"):
        try:
            info[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return info


def load_map_csv(map_path: Path) -> list[dict[str, str]]:
    with map_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_gallery(
    clip_dir: Path,
    map_dir: Path,
    media_dir: Path,
) -> tuple[np.ndarray, list[KeyframeRef]]:
    media = load_media_info(media_dir)
    npy_paths = sorted(clip_dir.glob("*.npy"))
    if not npy_paths:
        raise SystemExit(f"No .npy files under {clip_dir}")

    blocks: list[np.ndarray] = []
    refs: list[KeyframeRef] = []

    for npy_path in npy_paths:
        video_id = npy_path.stem
        feats = np.load(npy_path)
        if feats.ndim != 2:
            raise SystemExit(f"Expected 2D array in {npy_path}, got {feats.shape}")

        map_path = map_dir / f"{video_id}.csv"
        if not map_path.is_file():
            print(f"warning: missing map CSV for {video_id}, skipping", file=sys.stderr)
            continue
        rows = load_map_csv(map_path)
        n = min(len(rows), feats.shape[0])
        if len(rows) != feats.shape[0]:
            print(
                f"warning: {video_id} map rows={len(rows)} npy rows={feats.shape[0]}; "
                f"using first {n}",
                file=sys.stderr,
            )

        meta = media.get(video_id, {})
        title = str(meta.get("title") or "")
        watch_url = str(meta.get("watch_url") or "")

        feats = feats[:n].astype(np.float32, copy=False)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        feats = feats / norms
        blocks.append(feats)

        for i in range(n):
            row = rows[i]
            pts = float(row["pts_time"])
            frame_idx = int(float(row["frame_idx"]))
            fps = float(row["fps"])
            moment = youtube_moment_url(watch_url, pts) if watch_url else ""
            refs.append(
                KeyframeRef(
                    video_id=video_id,
                    keyframe_index=i,
                    frame_idx=frame_idx,
                    pts_time=pts,
                    fps=fps,
                    title=title,
                    watch_url=watch_url,
                    moment_url=moment,
                )
            )

    gallery = np.concatenate(blocks, axis=0)
    return gallery, refs


def load_clip_text_encoder(device: torch.device):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def encode_queries(
    model,
    tokenizer,
    queries: list[str],
    device: torch.device,
) -> np.ndarray:
    tokens = tokenizer(queries).to(device)
    text_feats = model.encode_text(tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats.cpu().numpy().astype(np.float32)


def search_topk(
    gallery: np.ndarray,
    refs: list[KeyframeRef],
    query_vecs: np.ndarray,
    top_k: int,
) -> list[list[tuple[float, KeyframeRef]]]:
    k = min(top_k, gallery.shape[0])
    # (Q, D) @ (D, N) -> (Q, N)
    scores = query_vecs @ gallery.T
    results: list[list[tuple[float, KeyframeRef]]] = []
    for q in range(scores.shape[0]):
        idx = np.argpartition(-scores[q], kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[q, idx])]
        results.append([(float(scores[q, i]), refs[i]) for i in idx])
    return results


def format_result(rank: int, score: float, ref: KeyframeRef) -> str:
    lines = [
        f"  #{rank}  score={score:.4f}",
        f"      video_id={ref.video_id}  frame_id={ref.frame_idx}  "
        f"keyframe_index={ref.keyframe_index}  pts_time={ref.pts_time:.3f}s  fps={ref.fps}",
    ]
    if ref.title:
        lines.append(f"      title={ref.title}")
    if ref.watch_url:
        lines.append(f"      watch_url={ref.watch_url}")
    if ref.moment_url:
        lines.append(f"      moment_url={ref.moment_url}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = repo_root()
    p = argparse.ArgumentParser(
        description="CLIP Textual KIS search over merged per-video .npy features."
    )
    p.add_argument(
        "queries_file",
        type=Path,
        help="Text file with one query per non-empty line",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results per query (default: 10)",
    )
    p.add_argument(
        "--clip-dir",
        type=Path,
        default=root / "clip-features-32-aic25-b1" / "clip-features-32",
        help="Directory of per-video CLIP .npy files",
    )
    p.add_argument(
        "--map-dir",
        type=Path,
        default=root / "map-keyframes-aic25-b1" / "map-keyframes",
        help="Directory of keyframe map CSVs",
    )
    p.add_argument(
        "--media-dir",
        type=Path,
        default=root / "media-info-aic25-b1" / "media-info",
        help="Directory of media-info JSON files",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device for text encoding (default: cpu)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Prefer UTF-8 on Windows consoles so Vietnamese titles do not crash.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top_k must be >= 1")

    queries = read_queries(args.queries_file)
    safe_print(f"Loaded {len(queries)} quer(y/ies) from {args.queries_file}")

    safe_print(f"Building gallery from {args.clip_dir} ...")
    gallery, refs = build_gallery(args.clip_dir, args.map_dir, args.media_dir)
    safe_print(f"Gallery: {gallery.shape[0]} keyframes, dim={gallery.shape[1]}")

    device = torch.device(args.device)
    safe_print(f"Loading CLIP ViT-B-32 text encoder on {device} ...")
    model, tokenizer = load_clip_text_encoder(device)
    query_vecs = encode_queries(model, tokenizer, queries, device)

    hits = search_topk(gallery, refs, query_vecs, args.top_k)

    for qi, query in enumerate(queries):
        safe_print()
        safe_print("=" * 72)
        safe_print(f"Query [{qi + 1}]: {query}")
        safe_print("=" * 72)
        for rank, (score, ref) in enumerate(hits[qi], start=1):
            safe_print(format_result(rank, score, ref))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
