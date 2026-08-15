"""Embed a keyframe still tree into BTC-shaped per-video .npy files."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

IMAGE_EXTS = {".webp", ".jpg", ".jpeg", ".png"}


class ImageEmbedder(Protocol):
    name: str

    def embed_pils(self, images: list[Image.Image]) -> np.ndarray: ...


def safe_print(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def sorted_images(folder: Path) -> list[Path]:
    imgs = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(imgs, key=lambda p: p.name)


def list_video_dirs(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    out: dict[str, Path] = {}
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "map.csv").is_file():
            out[p.name] = p
    if not out:
        raise SystemExit(f"No VIDEO_ID/map.csv trees under {root}")
    return out


def load_map_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _l2_normalize(feats: np.ndarray) -> np.ndarray:
    feats = feats.astype(np.float32, copy=False)
    if feats.ndim != 2:
        raise SystemExit(f"Expected 2D embeddings, got shape {feats.shape}")
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.maximum(norms, 1e-12)


def copy_one_video(folder: Path, video_id: str) -> np.ndarray:
    src = folder / "embeddings.npy"
    if not src.is_file():
        raise SystemExit(
            f"{video_id}: missing {src.name} (extract writes this next to map.csv)"
        )
    rows = load_map_rows(folder / "map.csv")
    feats = np.load(src)
    if feats.ndim != 2:
        raise SystemExit(f"{video_id}: {src.name} must be 2D, got {feats.shape}")
    if not rows:
        raise SystemExit(f"{video_id}: empty map.csv")
    if feats.shape[0] != len(rows):
        raise SystemExit(
            f"{video_id}: embeddings.npy rows={feats.shape[0]} map rows={len(rows)}"
        )
    return _l2_normalize(feats)


def embed_one_video(
    folder: Path,
    video_id: str,
    embedder: ImageEmbedder,
    *,
    batch_size: int,
) -> np.ndarray:
    rows = load_map_rows(folder / "map.csv")
    imgs = sorted_images(folder)
    n = min(len(rows), len(imgs))
    if not rows:
        raise SystemExit(f"{video_id}: empty map.csv")
    if len(rows) != len(imgs):
        safe_print(
            f"WARN {video_id}: images={len(imgs)} map={len(rows)}; using first {n}"
        )
    if n == 0:
        raise SystemExit(f"{video_id}: no images to embed")

    pils = [Image.open(p).convert("RGB") for p in imgs[:n]]
    chunks: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        chunks.append(embedder.embed_pils(pils[start : start + batch_size]))
    feats = np.concatenate(chunks, axis=0)
    if feats.ndim != 2 or feats.shape[0] != n:
        raise SystemExit(
            f"{video_id}: embedder returned shape {feats.shape}, expected ({n}, D)"
        )
    return _l2_normalize(feats)


def embed_tree(
    src_root: Path,
    out_dir: Path,
    embedder: ImageEmbedder | None = None,
    *,
    batch_size: int = 16,
    device_label: str = "cpu",
    copy_embeddings: bool = False,
) -> dict:
    if copy_embeddings and embedder is not None:
        safe_print("WARN --copy-embeddings: ignoring CLIP embedder, copying .npy")
    if not copy_embeddings and embedder is None:
        raise SystemExit("embed_tree needs an embedder unless copy_embeddings=True")

    video_dirs = list_video_dirs(src_root)
    out_root = out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    videos: list[dict] = []
    for video_id, folder in video_dirs.items():
        if copy_embeddings:
            feats = copy_one_video(folder, video_id)
            source = "embeddings.npy"
        else:
            assert embedder is not None
            feats = embed_one_video(
                folder, video_id, embedder, batch_size=batch_size
            )
            source = "clip"
        npy_path = out_root / f"{video_id}.npy"
        np.save(npy_path, feats)
        videos.append(
            {
                "video_id": video_id,
                "n_rows": int(feats.shape[0]),
                "dim": int(feats.shape[1]),
                "npy": npy_path.name,
                "source": source,
            }
        )
        safe_print(f"[{video_id}] wrote {npy_path.name} shape={feats.shape} via {source}")

    meta = {
        "model": "copy:embeddings.npy" if copy_embeddings else embedder.name,
        "device": device_label,
        "batch_size": batch_size,
        "copied": copy_embeddings,
        "n_videos": len(videos),
        "videos": videos,
    }
    meta_path = out_root / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    safe_print(f"Wrote {meta_path}")
    return meta
