"""Load FAISS indexes, gallery maps, and ASR interval metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from engine.gallery.encoders import (
    ASR_DIM,
    ASR_MODEL_ID,
    ASRQueryEncoder,
    CLIP_DIM,
    CLIP_MODEL_ID,
    ClipTextQueryEncoder,
    SIGLIP_DIM,
    SIGLIP_MODEL_ID,
    Siglip2TextQueryEncoder,
)


def load_gallery_map(path: Path) -> list[dict[str, str]]:
    """Load gallery_map.csv."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}; run the merge-gallery notebook first: {path}"
        )
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def gallery_map_signature(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    """Summarize gallery_map for alignment checks: [(video_id, n_rows), …] in file order."""
    return [(r["video_id"], int(r["n_rows"])) for r in rows]


def read_faiss_index(path: Path, expected_dim: int) -> faiss.Index:
    """Load index.faiss; row i in the index = global keyframe clip_row i."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing FAISS index: {path}")
    index = faiss.read_index(str(path))
    if index.d != expected_dim:
        raise ValueError(f"Expected index dim {expected_dim}, got {index.d} ({path})")
    return index


def read_asr_intervals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read file json then return numpy array of starts and ends"""
    starts, ends = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        starts.append(float(obj["start"]))
        ends.append(float(obj["end"]))
    return np.asarray(starts, dtype=np.float32), np.asarray(ends, dtype=np.float32)


def load_visual_tower(
    gallery_dir: Path,
    *,
    model_id: str,
    dim: int,
    model_cls,
    processor_cls,
    encoder_cls,
    device: torch.device,
):
    """Load one visual channel: query encoder + FAISS index from gallery_dir.

    In:  gallery_dir/index.faiss + HF model weights.
    Out: (encoder, index) — encoder.encode_query(text) → vector for ANN search.
    """
    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"Features root missing gallery: {gallery_dir}")
    model = model_cls.from_pretrained(model_id).to(device).eval()
    processor = processor_cls.from_pretrained(model_id)
    encoder = encoder_cls(model, processor, device, normalize_embeddings=True)
    index = read_faiss_index(gallery_dir / "index.faiss", dim)
    return encoder, index


def assert_clip_siglip_aligned(features_root: Path, clip_index: faiss.Index, siglip_index: faiss.Index) -> None:
    """Raise if CLIP and SigLIP2 indexes differ in row count or gallery_map order."""
    if siglip_index.ntotal != clip_index.ntotal:
        raise ValueError(
            f"SigLIP2 ntotal={siglip_index.ntotal} != CLIP ntotal={clip_index.ntotal}"
        )
    clip_map = load_gallery_map(features_root / "clip" / "gallery_map.csv")
    sig_map = load_gallery_map(features_root / "SigLIP2" / "gallery_map.csv")
    if gallery_map_signature(clip_map) != gallery_map_signature(sig_map):
        raise ValueError(
            "SigLIP2 gallery_map does not match CLIP (video_id, n_rows order)"
        )


def load_asr_gallery(asr_dir: Path, device: torch.device) -> tuple[ASRQueryEncoder, np.ndarray]:
    """Load MiniLM query encoder + segment embedding matrix (asr_emb/).

    In:  asr_dir/embeddings.npy or per-video *.npy blocks.
    Out: (encoder, mat) — mat shape (n_segments, 384); row j = global ASR segment j.
    """
    if not asr_dir.is_dir():
        raise FileNotFoundError(f"Features root missing asr_emb/: {asr_dir}")

    model = SentenceTransformer(ASR_MODEL_ID, device=str(device))
    encoder = ASRQueryEncoder(model, device, normalize_embeddings=True)

    cache = asr_dir / "embeddings.npy"
    if cache.is_file():
        asr_embed_mat = np.load(cache)
    else:
        npy_paths = sorted(asr_dir.glob("*/*.npy"))
        if not npy_paths:
            npy_paths = sorted(
                p for p in asr_dir.glob("*.npy") if p.name != "embeddings.npy"
            )
        npy_paths = [p for p in npy_paths if p.name != "embeddings.npy"]
        if not npy_paths:
            raise FileNotFoundError(f"No per-video .npy under {asr_dir}")
        blocks = [np.load(p).astype(np.float32, copy=False) for p in npy_paths]
        asr_embed_mat = np.concatenate(blocks, axis=0)
        np.save(cache, asr_embed_mat)

    if asr_embed_mat.ndim != 2 or asr_embed_mat.shape[-1] != ASR_DIM:
        raise ValueError(
            f"Expected ASR gallery (*, {ASR_DIM}), got {asr_embed_mat.shape}"
        )
    return encoder, asr_embed_mat


def build_row_index_map_info(
    clip_map: list[dict[str, str]], # clip/gallery_map.csv
    maps_dir: Path, # features/maps/
    *,
    ntotal: int, # total number of keyframes
) -> dict[str, Any]:
    """Build per-keyframe metadata aligned with FAISS clip_row index."""
    info: dict[str, Any] = {
        "video_ids": [],
        "pts_list": [],
        "frame_idx_list": [],
        "row_to_idx_in_each_video": [],
    }
    n_clip = 0
    for row in clip_map:
        video_id = row["video_id"]
        n = int(row["n_rows"])
        map_csv = maps_dir / f"{video_id}.csv"
        if not map_csv.is_file():
            raise FileNotFoundError(f"Missing keyframe map: {map_csv}")
        with map_csv.open(encoding="utf-8", newline="") as f:
            map_rows = list(csv.DictReader(f))
        pts = [float(r["pts_time"]) for r in map_rows]
        frame_idx = [int(r["frame_idx"]) for r in map_rows]
        if len(pts) < n or len(frame_idx) < n:
            raise ValueError(
                f"{video_id}: map.csv has {len(pts)} pts / {len(frame_idx)} "
                f"frame_idx, clip gallery n_rows={n}"
            )
        info["video_ids"].extend([video_id] * n)
        info["pts_list"].extend(pts[:n])
        info["frame_idx_list"].extend(frame_idx[:n])
        info["row_to_idx_in_each_video"].extend(list(range(n)))
        n_clip += n

    if n_clip != ntotal:
        raise ValueError(
            f"clip gallery_map covers {n_clip} rows, FAISS ntotal={ntotal}"
        )
    info["pts_list"] = np.asarray(info["pts_list"], dtype=np.float32)
    return info


def build_asr_by_video(
    asr_map: list[dict[str, str]], # asr_emb/gallery_map.csv
    features_root: Path, # features/
) -> dict[str, tuple[int, np.ndarray, np.ndarray]]:
    """Map each video -> its slice in the global ASR embedding matrix.
        Output format: {video_id: (start_row, starts, ends)}
    """
    asr_by_video: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    for row in asr_map:
        video_id = row["video_id"]
        start_row = int(row["start_row"])
        n = int(row["n_rows"])
        jsonl = features_root / "asr" / video_id.split("_")[0] / f"{video_id}.jsonl"
        if not jsonl.is_file():
            starts = np.zeros(0, dtype=np.float32)
            ends = np.zeros(0, dtype=np.float32)
        else:
            starts, ends = read_asr_intervals(jsonl)
        if len(starts) != n or len(ends) != n:
            raise ValueError(
                f"{video_id}: asr jsonl segs={len(starts)}={len(ends)} "
                f"gallery n_rows={n}"
            )
        asr_by_video[video_id] = (start_row, starts, ends)
    return asr_by_video
