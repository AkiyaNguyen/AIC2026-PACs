#!/usr/bin/env python3
"""Compare 2+ keyframe extraction trees (same VIDEO_ID / map.csv layout).

Expects each root to contain::

    ROOT/
      VIDEO_ID/
        map.csv          # n,pts_time,fps,frame_idx
        *.webp|jpg|png
        embeddings.npy   # optional

Usage
-----
Multi-method (recommended for 3 extractors + BTC)::

    python tools/compare_keyframes.py \\
      --roots clip:/path/clip/Keyframes_L21 \\
              siglip:/path/siglip/Keyframes_L21 \\
              beit3:/path/beit3/Keyframes_L21 \\
              btc:/path/btc/Keyframes_L21 \\
      --out-dir compare-out/L21_4way

Legacy pairwise still works::

    python tools/compare_keyframes.py --dir-a A --dir-b B --label-a clip --label-b btc ...

What we measure
---------------
1. Density: kf/s, gap stats, fraction of gaps > ``--trake-gap`` (default 10).
2. Uniqueness: consecutive embedding cosine distance if embeddings.npy exists.
3. Pairwise coverage matrix: for every ordered pair (i→j), fraction of i's
   frames matched in j within ``--match-window``.
4. HTML manual gallery + plots (gap hist, density bars, coverage heatmap).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".webp", ".jpg", ".jpeg", ".png"}


@dataclass
class VideoMaps:
    video_id: str
    frame_idx: np.ndarray
    pts_time: np.ndarray
    fps: float
    image_paths: list[Path]
    embeddings: np.ndarray | None


@dataclass
class MethodRoot:
    label: str
    path: Path
    video_dirs: dict[str, Path]


def safe_print(text: str = "") -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


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


def load_map_csv(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), 25.0
    frame_idx = np.array([int(float(r["frame_idx"])) for r in rows], dtype=np.int64)
    pts = np.array([float(r["pts_time"]) for r in rows], dtype=np.float64)
    fps = float(rows[0].get("fps") or 25.0)
    return frame_idx, pts, fps


def load_btc_map(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return np.array([int(float(r["frame_idx"])) for r in rows], dtype=np.int64)


def sorted_images(folder: Path) -> list[Path]:
    imgs = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(imgs, key=lambda p: p.name)


def load_video_maps(folder: Path, video_id: str) -> VideoMaps:
    frame_idx, pts, fps = load_map_csv(folder / "map.csv")
    imgs = sorted_images(folder)
    emb = None
    emb_path = folder / "embeddings.npy"
    if emb_path.is_file():
        emb = np.load(emb_path)
        if emb.shape[0] != len(frame_idx):
            safe_print(
                f"WARN {video_id}: embeddings.npy rows={emb.shape[0]} "
                f"!= map rows={len(frame_idx)}; ignoring embeddings"
            )
            emb = None
    if imgs and len(imgs) != len(frame_idx):
        safe_print(
            f"WARN {video_id}: images={len(imgs)} map={len(frame_idx)} "
            "(HTML will zip by min length)"
        )
    return VideoMaps(video_id, frame_idx, pts, fps, imgs, emb)


def gap_stats(frame_idx: np.ndarray, fps: float, trake_gap: int) -> dict:
    if frame_idx.size <= 1:
        return {
            "n": int(frame_idx.size),
            "duration_s": float(frame_idx[-1] / fps) if frame_idx.size else 0.0,
            "kf_per_s": None,
            "mean_gap": None,
            "median_gap": None,
            "p95_gap": None,
            "max_gap": None,
            "frac_gaps_gt_trake": None,
            "n_gaps_gt_trake": 0,
        }
    gaps = np.diff(frame_idx.astype(np.float64))
    duration_s = float(frame_idx[-1] / fps) if fps > 0 else float("nan")
    return {
        "n": int(frame_idx.size),
        "duration_s": duration_s,
        "kf_per_s": float(frame_idx.size / duration_s) if duration_s > 0 else None,
        "mean_gap": float(gaps.mean()),
        "median_gap": float(np.median(gaps)),
        "p95_gap": float(np.percentile(gaps, 95)),
        "max_gap": float(gaps.max()),
        "frac_gaps_gt_trake": float(np.mean(gaps > trake_gap)),
        "n_gaps_gt_trake": int(np.sum(gaps > trake_gap)),
    }


def consecutive_cosine_distances(emb: np.ndarray) -> np.ndarray:
    if emb is None or len(emb) < 2:
        return np.array([], dtype=np.float64)
    a = emb[:-1]
    b = emb[1:]
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return 1.0 - np.sum(a * b, axis=1)


def uniqueness_stats(emb: np.ndarray | None) -> dict:
    d = consecutive_cosine_distances(emb) if emb is not None else np.array([])
    if d.size == 0:
        return {"mean_consec_dist": None, "min_consec_dist": None, "p05_consec_dist": None}
    return {
        "mean_consec_dist": float(d.mean()),
        "min_consec_dist": float(d.min()),
        "p05_consec_dist": float(np.percentile(d, 5)),
    }


def match_coverage(src: np.ndarray, dst: np.ndarray, window: int) -> dict:
    if src.size == 0:
        return {"coverage": None, "matched": 0, "total": 0, "mean_abs_err": None}
    if dst.size == 0:
        return {"coverage": 0.0, "matched": 0, "total": int(src.size), "mean_abs_err": None}
    dst_sorted = np.sort(dst)
    matched = 0
    errs: list[float] = []
    for x in src:
        j = int(np.searchsorted(dst_sorted, x))
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(dst_sorted):
                err = abs(int(dst_sorted[k]) - int(x))
                if best is None or err < best:
                    best = err
        if best is not None and best <= window:
            matched += 1
            errs.append(float(best))
    return {
        "coverage": matched / len(src),
        "matched": matched,
        "total": int(src.size),
        "mean_abs_err": float(np.mean(errs)) if errs else None,
    }


def unmatched_indices(src: np.ndarray, dst: np.ndarray, window: int) -> list[int]:
    out: list[int] = []
    if src.size == 0:
        return out
    dst_sorted = np.sort(dst) if dst.size else dst
    for i, x in enumerate(src):
        if dst_sorted.size == 0:
            out.append(i)
            continue
        j = int(np.searchsorted(dst_sorted, x))
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(dst_sorted):
                err = abs(int(dst_sorted[k]) - int(x))
                if best is None or err < best:
                    best = err
        if best is None or best > window:
            out.append(i)
    return out


def largest_gap_indices(frame_idx: np.ndarray, k: int) -> list[tuple[int, int, int]]:
    if frame_idx.size < 2:
        return []
    gaps = np.diff(frame_idx)
    order = np.argsort(-gaps)[:k]
    return [(int(i), int(i + 1), int(gaps[i])) for i in order]


def nearest_index(frame_idx: np.ndarray, x: int) -> int | None:
    if frame_idx.size == 0:
        return None
    return int(np.argmin(np.abs(frame_idx.astype(np.int64) - x)))


def parse_root_spec(spec: str) -> tuple[str, Path]:
    """Parse ``label:path`` or plain path (label = folder name)."""
    if ":" in spec:
        # Allow Windows drive letters: C:\... — only split on first ':' if
        # it looks like label:path (label has no path sep and path has sep or is absolute-ish)
        label, rest = spec.split(":", 1)
        # Heuristic: if label is a single letter and rest starts with \ or /, it's a Windows path
        if len(label) == 1 and rest[:1] in "\\/":
            return Path(spec).name, Path(spec)
        if label and rest:
            return label.strip(), Path(rest.strip())
    p = Path(spec)
    return p.name, p


def resolve_methods(args: argparse.Namespace) -> list[MethodRoot]:
    specs: list[tuple[str, Path]] = []
    if args.roots:
        for raw in args.roots:
            specs.append(parse_root_spec(raw))
    if args.dir_a is not None or args.dir_b is not None:
        if args.dir_a is None or args.dir_b is None:
            raise SystemExit("Provide both --dir-a and --dir-b, or use --roots")
        specs.append((args.label_a, args.dir_a))
        specs.append((args.label_b, args.dir_b))
    if len(specs) < 2:
        raise SystemExit("Need at least 2 method roots (--roots ... or --dir-a/--dir-b)")

    methods: list[MethodRoot] = []
    seen_labels: set[str] = set()
    for label, path in specs:
        if label in seen_labels:
            raise SystemExit(f"Duplicate method label: {label}")
        seen_labels.add(label)
        path = path.expanduser().resolve()
        methods.append(MethodRoot(label=label, path=path, video_dirs=list_video_dirs(path)))
    return methods


def plot_gap_histograms_multi(
    out_path: Path,
    gaps_by_label: dict[str, list[float]],
    trake_gap: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = 40
    for label, gaps in gaps_by_label.items():
        if gaps:
            ax.hist(gaps, bins=bins, alpha=0.45, label=label, density=True)
    ax.axvline(trake_gap, color="black", linestyle="--", linewidth=1.2, label=f"TRAKE~{trake_gap}f")
    ax.set_xlabel("Consecutive keyframe gap (frames)")
    ax.set_ylabel("Density")
    ax.set_title("Keyframe gap distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_density_bars_multi(out_path: Path, per_video: list[dict], labels: list[str]) -> None:
    import matplotlib.pyplot as plt

    vids = [r["video_id"] for r in per_video]
    n_m = len(labels)
    x = np.arange(len(vids))
    width = min(0.8 / max(n_m, 1), 0.25)
    fig, ax = plt.subplots(figsize=(max(6, len(vids) * 0.5), 4.8))
    for i, lab in enumerate(labels):
        vals = [r["methods"][lab].get("kf_per_s") or 0.0 for r in per_video]
        ax.bar(x + (i - (n_m - 1) / 2) * width, vals, width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(vids, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Keyframes / second")
    ax.set_title("Density by video")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_coverage_heatmap(out_path: Path, labels: list[str], matrix: list[list[float | None]]) -> None:
    import matplotlib.pyplot as plt

    n = len(labels)
    data = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            if v is not None:
                data[i, j] = v
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("dst (covers)")
    ax.set_ylabel("src (to be covered)")
    ax.set_title("Mean coverage: src frames matched in dst")
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            txt = "—" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if (not np.isnan(val) and val < 0.55) else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def rel_uri(path: Path, report_dir: Path) -> str:
    try:
        return path.resolve().relative_to(report_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def write_html_report(
    report_path: Path,
    *,
    labels: list[str],
    samples: list[dict],
    summary: dict,
) -> None:
    report_dir = report_path.parent
    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Keyframe compare</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;}",
        "table{border-collapse:collapse;width:100%;margin:12px 0;}",
        "th,td{border:1px solid #ccc;padding:6px 8px;vertical-align:top;}",
        "img{max-width:240px;height:auto;background:#111;}",
        ".meta{color:#444;font-size:13px;}",
        "pre{white-space:pre-wrap;background:#f6f6f6;padding:12px;}",
        "</style></head><body>",
        "<h1>Keyframe extraction comparison</h1>",
        f"<p class='meta'>Methods: <b>{html.escape(', '.join(labels))}</b></p>",
        "<p class='meta'>Plots: gaps_hist.png, density_bars.png, coverage_heatmap.png</p>",
        "<h2>Aggregate</h2><pre>",
        html.escape(json.dumps(summary, indent=2)),
        "</pre><h2>Manual samples</h2>",
    ]
    for s in samples:
        parts.append(f"<h3>{html.escape(s['title'])}</h3>")
        parts.append(f"<p class='meta'>{html.escape(s.get('note', ''))}</p>")
        cols = s.get("columns") or []
        if not cols:
            continue
        parts.append("<table><tr>")
        for c in cols:
            parts.append(f"<th>{html.escape(c.get('label', ''))}</th>")
        parts.append("</tr><tr>")
        for c in cols:
            img = c.get("image")
            caption = c.get("caption", "")
            if img:
                uri = rel_uri(Path(img), report_dir)
                parts.append(
                    "<td>"
                    f"<div class='meta'>{html.escape(caption)}</div>"
                    f"<img src='{html.escape(uri)}' alt=''/>"
                    "</td>"
                )
            else:
                parts.append(
                    f"<td><div class='meta'>{html.escape(caption or 'n/a')}</div></td>"
                )
        parts.append("</tr></table>")
    parts.append("</body></html>")
    report_path.write_text("\n".join(parts), encoding="utf-8")


def _img(vm: VideoMaps, i: int) -> str | None:
    if 0 <= i < len(vm.image_paths):
        return str(vm.image_paths[i])
    return None


def build_manual_samples_multi(
    per_video_maps: list[dict[str, VideoMaps]],
    labels: list[str],
    *,
    match_window: int,
    n_gap: int,
    n_unmatched: int,
    n_random: int,
    seed: int,
) -> list[dict]:
    """Build HTML samples with one column per method where possible."""
    rng = random.Random(seed)
    samples: list[dict] = []
    if not labels:
        return samples

    primary = labels[0]

    for maps in per_video_maps:
        vid = next(iter(maps.values())).video_id

        # Largest gaps on each method (show gap endpoints for that method only)
        for lab in labels:
            vm = maps[lab]
            for left_i, right_i, gap in largest_gap_indices(vm.frame_idx, n_gap):
                samples.append(
                    {
                        "title": f"{vid} — largest gap in {lab} ({gap} frames)",
                        "note": (
                            "Do endpoints look like a real scene change? "
                            f"TRAKE windows often <10 frames; gap={gap}."
                        ),
                        "columns": [
                            {
                                "label": f"{lab} left",
                                "image": _img(vm, left_i),
                                "caption": f"frame_idx={int(vm.frame_idx[left_i])}",
                            },
                            {
                                "label": f"{lab} right",
                                "image": _img(vm, right_i),
                                "caption": f"frame_idx={int(vm.frame_idx[right_i])}",
                            },
                        ],
                    }
                )

        # Frames in primary not covered by each other method
        vp = maps[primary]
        for lab in labels[1:]:
            um = unmatched_indices(vp.frame_idx, maps[lab].frame_idx, match_window)
            for i in um[:n_unmatched]:
                x = int(vp.frame_idx[i])
                cols = [
                    {
                        "label": primary,
                        "image": _img(vp, i),
                        "caption": f"{primary} frame_idx={x} (no {lab} within ±{match_window})",
                    }
                ]
                for other in labels:
                    if other == primary:
                        continue
                    vo = maps[other]
                    j = nearest_index(vo.frame_idx, x)
                    if j is None:
                        cols.append({"label": other, "image": None, "caption": "empty"})
                    else:
                        cols.append(
                            {
                                "label": other,
                                "image": _img(vo, j),
                                "caption": (
                                    f"{other} nearest={int(vo.frame_idx[j])} "
                                    f"|Δ|={abs(int(vo.frame_idx[j]) - x)}"
                                ),
                            }
                        )
                samples.append(
                    {
                        "title": f"{vid} — {primary}-only vs {lab}",
                        "note": "Is this a true keyframe the other method missed, or noise?",
                        "columns": cols,
                    }
                )

        # Random primary frame vs nearest in every method
        if vp.frame_idx.size and vp.image_paths:
            for _ in range(n_random):
                i = rng.randrange(len(vp.frame_idx))
                x = int(vp.frame_idx[i])
                cols = []
                for lab in labels:
                    vm = maps[lab]
                    j = nearest_index(vm.frame_idx, x)
                    if j is None:
                        cols.append({"label": lab, "image": None, "caption": "empty"})
                    else:
                        cols.append(
                            {
                                "label": lab,
                                "image": _img(vm, j),
                                "caption": (
                                    f"{lab} frame_idx={int(vm.frame_idx[j])} "
                                    f"|Δ|={abs(int(vm.frame_idx[j]) - x)}"
                                ),
                            }
                        )
                samples.append(
                    {
                        "title": f"{vid} — random align @ {primary}={x}",
                        "note": "Nearest frame in each method to the same target index.",
                        "columns": cols,
                    }
                )
    return samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare 2+ NII-UIT-style keyframe trees (multi-root)."
    )
    p.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help="Method roots as label:path (e.g. clip:/path/Keyframes_L21). Repeatable list.",
    )
    p.add_argument("--dir-a", type=Path, default=None, help="Legacy: first root")
    p.add_argument("--dir-b", type=Path, default=None, help="Legacy: second root")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--match-window", type=int, default=5)
    p.add_argument("--trake-gap", type=int, default=10)
    p.add_argument(
        "--ref-map-dir",
        type=Path,
        default=None,
        help="Optional flat BTC map-keyframes dir (VIDEO_ID.csv) for extra coverage stats",
    )
    p.add_argument("--n-gap-samples", type=int, default=1)
    p.add_argument("--n-unmatched-samples", type=int, default=2)
    p.add_argument("--n-random-samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args(argv)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = resolve_methods(args)
    labels = [m.label for m in methods]
    n_m = len(methods)

    # Intersection of VIDEO_IDs present in all methods
    common = set(methods[0].video_dirs)
    for m in methods[1:]:
        common &= set(m.video_dirs)
    common_vids = sorted(common)
    if not common_vids:
        raise SystemExit("No overlapping VIDEO_ID folders with map.csv across all roots")

    only: dict[str, list[str]] = {}
    all_ids = set()
    for m in methods:
        all_ids |= set(m.video_dirs)
    for m in methods:
        only[m.label] = sorted(set(m.video_dirs) - common)

    per_video: list[dict] = []
    gaps_by_label: dict[str, list[float]] = {lab: [] for lab in labels}
    per_video_maps: list[dict[str, VideoMaps]] = []
    # Accumulate coverage sums for heatmap
    cov_sum = [[0.0] * n_m for _ in range(n_m)]
    cov_cnt = [[0] * n_m for _ in range(n_m)]

    for vid in common_vids:
        maps: dict[str, VideoMaps] = {}
        for m in methods:
            maps[m.label] = load_video_maps(m.video_dirs[vid], vid)
        per_video_maps.append(maps)

        row: dict = {"video_id": vid, "methods": {}, "coverage": {}}
        for lab in labels:
            vm = maps[lab]
            stats = {**gap_stats(vm.frame_idx, vm.fps, args.trake_gap), **uniqueness_stats(vm.embeddings)}
            row["methods"][lab] = stats
            if vm.frame_idx.size > 1:
                gaps_by_label[lab].extend(np.diff(vm.frame_idx).astype(float).tolist())

        for i, src_lab in enumerate(labels):
            for j, dst_lab in enumerate(labels):
                if i == j:
                    cov = {"coverage": 1.0, "matched": int(maps[src_lab].frame_idx.size), "total": int(maps[src_lab].frame_idx.size), "mean_abs_err": 0.0}
                else:
                    cov = match_coverage(
                        maps[src_lab].frame_idx,
                        maps[dst_lab].frame_idx,
                        args.match_window,
                    )
                row["coverage"][f"{src_lab}->{dst_lab}"] = cov
                if cov["coverage"] is not None:
                    cov_sum[i][j] += cov["coverage"]
                    cov_cnt[i][j] += 1

        if args.ref_map_dir:
            ref_csv = args.ref_map_dir.expanduser().resolve() / f"{vid}.csv"
            if ref_csv.is_file():
                ref = load_btc_map(ref_csv)
                row["ref"] = {}
                for lab in labels:
                    row["ref"][f"ref_covered_by_{lab}"] = match_coverage(
                        ref, maps[lab].frame_idx, args.match_window
                    )
                    row["ref"][f"{lab}_covered_by_ref"] = match_coverage(
                        maps[lab].frame_idx, ref, args.match_window
                    )

        per_video.append(row)
        dens = " | ".join(
            f"{lab} n={row['methods'][lab]['n']} kf/s={row['methods'][lab]['kf_per_s']}"
            for lab in labels
        )
        safe_print(f"{vid}: {dens}")

    mean_cov_matrix: list[list[float | None]] = []
    for i in range(n_m):
        row_m: list[float | None] = []
        for j in range(n_m):
            if cov_cnt[i][j] == 0:
                row_m.append(None)
            else:
                row_m.append(cov_sum[i][j] / cov_cnt[i][j])
        mean_cov_matrix.append(row_m)

    aggregate = {
        "labels": labels,
        "paths": {m.label: str(m.path) for m in methods},
        "n_common_videos": len(common_vids),
        "only_in_method": only,
        "match_window": args.match_window,
        "trake_gap": args.trake_gap,
        "mean_kf_per_s": {
            lab: _mean([r["methods"][lab].get("kf_per_s") for r in per_video]) for lab in labels
        },
        "mean_gap": {
            lab: _mean([r["methods"][lab].get("mean_gap") for r in per_video]) for lab in labels
        },
        "mean_frac_gaps_gt_trake": {
            lab: _mean([r["methods"][lab].get("frac_gaps_gt_trake") for r in per_video])
            for lab in labels
        },
        "mean_consec_dist": {
            lab: _mean([r["methods"][lab].get("mean_consec_dist") for r in per_video])
            for lab in labels
        },
        "mean_coverage_matrix": {
            "labels": labels,
            "rows_are_src": True,
            "matrix": mean_cov_matrix,
        },
    }

    report = {"aggregate": aggregate, "per_video": per_video}
    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    safe_print(f"Wrote {report_json}")

    try:
        plot_gap_histograms_multi(out_dir / "gaps_hist.png", gaps_by_label, args.trake_gap)
        plot_density_bars_multi(out_dir / "density_bars.png", per_video, labels)
        plot_coverage_heatmap(out_dir / "coverage_heatmap.png", labels, mean_cov_matrix)
        safe_print(f"Wrote plots under {out_dir}")
    except Exception as exc:  # noqa: BLE001
        safe_print(f"Plotting skipped/failed: {exc}")

    samples = build_manual_samples_multi(
        per_video_maps,
        labels,
        match_window=args.match_window,
        n_gap=args.n_gap_samples,
        n_unmatched=args.n_unmatched_samples,
        n_random=args.n_random_samples,
        seed=args.seed,
    )
    html_path = out_dir / "index.html"
    write_html_report(html_path, labels=labels, samples=samples, summary=aggregate)
    safe_print(f"Open manual review: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
