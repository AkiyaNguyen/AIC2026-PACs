"""argparse CLI: extract | embed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.extract_features.engine.clip import load_clip
from tools.util import get_proper_device
from tools.extract_features.engine.embed import embed_tree
from tools.extract_features.engine.extract import load_list_file, run_extract, safe_print


def _add_device(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--device",
        choices=("cpu", "gpu", "cuda"),
        default="cpu",
        help="cpu (default) or gpu (CUDA, else MPS)",
    )


def _add_batch_size(p: argparse.ArgumentParser) -> None:
    p.add_argument("--batch-size", type=int, default=16)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.extract_features",
        description=(
            "NII-UIT-style keyframe extract (CLIP keep/drop) and CLIP embed "
            "of a still tree into per-video .npy files."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser(
        "extract",
        help="Sample every Nth frame, keep if CLIP-cosine distance is large enough",
    )
    pe.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Video file(s) and/or folder(s) containing videos",
    )
    pe.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output root (creates VIDEO_ID/ subfolders)",
    )
    pe.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Sample every Nth frame before embedding (NII-UIT: 10)",
    )
    pe.add_argument(
        "--min-cosine-distance",
        type=float,
        default=0.15,
        help="Keep frame if 1-cos(last,cur) >= this (default: 0.15)",
    )
    pe.add_argument("--webp-quality", type=int, default=80)
    pe.add_argument(
        "--list-file",
        type=Path,
        default=None,
        help="Optional text file: one video path or folder per line (# comments ok)",
    )
    _add_batch_size(pe)
    _add_device(pe)

    pb = sub.add_parser(
        "embed",
        help="Embed ROOT/VIDEO_ID/{map.csv,images} into OUT/VIDEO_ID.npy",
    )
    pb.add_argument(
        "root",
        type=Path,
        help="Keyframe tree root (VIDEO_ID folders with map.csv)",
    )
    pb.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output folder for VIDEO_ID.npy files",
    )
    pb.add_argument(
        "--copy-embeddings",
        action="store_true",
        help=(
            "Copy each VIDEO_ID/embeddings.npy to OUT/VIDEO_ID.npy "
            "(no CLIP reload; use after extract)"
        ),
    )
    _add_batch_size(pb)
    _add_device(pb)

    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = build_parser().parse_args(argv)

    if args.command == "extract":
        if args.stride < 1:
            raise SystemExit("--stride must be >= 1")
        if not (0.0 <= args.min_cosine_distance <= 2.0):
            raise SystemExit("--min-cosine-distance should be in [0, 2]")
        device = get_proper_device(args.device)
        safe_print(f"Loading CLIP ViT-B/32 device={device} ...")
        embedder = load_clip(device)
        inputs = list(args.inputs)
        if args.list_file:
            inputs.extend(load_list_file(args.list_file))
        run_extract(
            inputs,
            args.out_dir,
            embedder,
            stride=args.stride,
            min_cosine_distance=args.min_cosine_distance,
            batch_size=args.batch_size,
            webp_quality=args.webp_quality,
            device_label=str(device),
        )
        return 0

    if args.command == "embed":
        if args.copy_embeddings:
            embed_tree(
                args.root,
                args.out_dir,
                embedder=None,
                copy_embeddings=True,
                device_label="cpu",
            )
            return 0
        device = get_proper_device(args.device)
        safe_print(f"Loading CLIP ViT-B/32 device={device} ...")
        embedder = load_clip(device)
        embed_tree(
            args.root,
            args.out_dir,
            embedder,
            batch_size=args.batch_size,
            device_label=str(device),
        )
        return 0

    raise SystemExit(f"Unknown command: {args.command}")
