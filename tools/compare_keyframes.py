#!/usr/bin/env python3
"""Compare two keyframe extraction trees (same VIDEO_ID folder layout).

Expects each root to contain::

    ROOT/
      VIDEO_ID/
        map.csv          # n,pts_time,fps,frame_idx
        000000.webp ...
        embeddings.npy   # optional; used for uniqueness if present

What we measure
---------------
1. **Density** (contest-relevant): keyframes / second, mean/median/p95/max
   gap in frames, fraction of consecutive gaps **> 10 frames** (BTC TRAKE
   windows are typically under ~10 frames — large gaps risk missing a moment
   between indexed keyframes).

2. **Keyframiness / uniqueness proxy**: if ``embeddings.npy`` exists (written
   by ``extract_keyframes_niiuit.py``), report mean/min consecutive cosine
   distance. Higher mean distance ⇒ less redundant neighbors. Without
   embeddings, we still report temporal stats.

3. **Agreement between methods**: for each video, greedy match of frame
   indices within ``--match-window`` frames; precision/recall-style coverage
   of A by B and B by A.

4. **Manual check**: writes an HTML gallery under ``--out-dir`` with
   side-by-side samples — largest gaps, unmatched frames, and random pairs —
   so you can open images in a browser and judge whether they look like real
   keyframes / scene changes.

Optional ``--ref-map-dir`` (BTC ``map-keyframes`` CSVs) adds coverage of
organizer keyframes by each method within the match window.
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
    frame_idx: np.ndarray  # int
    pts_time: np.ndarray
    fps: float
    image_paths: list[Path]
    embeddings: np.ndarray | None


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
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
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
    # assume roughly normalized; renormalize defensively
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    sims = np.sum(a * b, axis=1)
    return 1.0 - sims


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
    """Fraction of src frames that have a dst frame within ±window."""
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
    """Indices into src that have no dst neighbor within window."""
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
    """Return list of (left_i, right_i, gap) for the k largest gaps."""
    if frame_idx.size < 2:
        return []
    gaps = np.diff(frame_idx)
    order = np.argsort(-gaps)[:k]
    return [(int(i), int(i + 1), int(gaps[i])) for i in order]


def plot_gap_histograms(
    out_path: Path,
    gaps_a: list[float],
    gaps_b: list[float],
    label_a: str,
    label_b: str,
    trake_gap: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = 40
    if gaps_a:
        ax.hist(gaps_a, bins=bins, alpha=0.55, label=label_a, density=True)
    if gaps_b:
        ax.hist(gaps_b, bins=bins, alpha=0.55, label=label_b, density=True)
    ax.axvline(trake_gap, color="black", linestyle="--", linewidth=1.2, label=f"TRAKE~{trake_gap}f")
    ax.set_xlabel("Consecutive keyframe gap (frames)")
    ax.set_ylabel("Density")
    ax.set_title("Keyframe gap distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_density_bars(out_path: Path, rows: list[dict], label_a: str, label_b: str) -> None:
    import matplotlib.pyplot as plt

    vids = [r["video_id"] for r in rows]
    ka = [r["a"]["kf_per_s"] or 0.0 for r in rows]
    kb = [r["b"]["kf_per_s"] or 0.0 for r in rows]
    x = np.arange(len(vids))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(vids) * 0.45), 4.5))
    ax.bar(x - width / 2, ka, width, label=label_a)
    ax.bar(x + width / 2, kb, width, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(vids, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Keyframes / second")
    ax.set_title("Density by video")
    ax.legend()
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
    label_a: str,
    label_b: str,
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
        "img{max-width:320px;height:auto;background:#111;}",
        ".meta{color:#444;font-size:13px;}",
        "</style></head><body>",
        "<h1>Keyframe extraction comparison</h1>",
        f"<p class='meta'>A=<b>{html.escape(label_a)}</b> &nbsp; B=<b>{html.escape(label_b)}</b></p>",
        "<h2>Aggregate</h2><pre>",
        html.escape(json.dumps(summary, indent=2)),
        "</pre><h2>Manual samples</h2>",
    ]
    for s in samples:
        parts.append(f"<h3>{html.escape(s['title'])}</h3>")
        parts.append(f"<p class='meta'>{html.escape(s.get('note', ''))}</p>")
        parts.append("<table><tr><th>A</th><th>B</th></tr><tr>")
        for side in ("a", "b"):
            cell = s.get(side) or {}
            img = cell.get("image")
            caption = cell.get("caption", "")
            if img:
                uri = rel_uri(Path(img), report_dir)
                parts.append(
                    "<td>"
                    f"<div class='meta'>{html.escape(caption)}</div>"
                    f"<img src='{html.escape(uri)}' alt=''/>"
                    "</td>"
                )
            else:
                parts.append(f"<td><div class='meta'>{html.escape(caption or 'n/a')}</div></td>")
        parts.append("</tr></table>")
    parts.append("</body></html>")
    report_path.write_text("\n".join(parts), encoding="utf-8")


def build_manual_samples(
    paired: list[tuple[VideoMaps, VideoMaps]],
    *,
    match_window: int,
    n_gap: int,
    n_unmatched: int,
    n_random: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []

    def _img(vm: VideoMaps, i: int) -> str | None:
        if 0 <= i < len(vm.image_paths):
            return str(vm.image_paths[i])
        return None

    for va, vb in paired:
        for left_i, right_i, gap in largest_gap_indices(va.frame_idx, n_gap):
            samples.append(
                {
                    "title": f"{va.video_id} — largest gap in A ({gap} frames)",
                    "note": (
                        "Do these endpoints look like a real scene change? "
                        f"TRAKE windows are often <10 frames; gap={gap}."
                    ),
                    "a": {
                        "image": _img(va, left_i),
                        "caption": f"A left frame_idx={int(va.frame_idx[left_i])}",
                    },
                    "b": {
                        "image": _img(va, right_i),
                        "caption": f"A right frame_idx={int(va.frame_idx[right_i])}",
                    },
                }
            )
        for left_i, right_i, gap in largest_gap_indices(vb.frame_idx, n_gap):
            samples.append(
                {
                    "title": f"{vb.video_id} — largest gap in B ({gap} frames)",
                    "note": (
                        "Do these endpoints look like a real scene change? "
                        f"TRAKE windows are often <10 frames; gap={gap}."
                    ),
                    "a": {
                        "image": _img(vb, left_i),
                        "caption": f"B left frame_idx={int(vb.frame_idx[left_i])}",
                    },
                    "b": {
                        "image": _img(vb, right_i),
                        "caption": f"B right frame_idx={int(vb.frame_idx[right_i])}",
                    },
                }
            )

        # unmatched A frames (in A, missing near B)
        um_a = unmatched_indices(va.frame_idx, vb.frame_idx, match_window)
        um_b = unmatched_indices(vb.frame_idx, va.frame_idx, match_window)
        for i in um_a[:n_unmatched]:
            samples.append(
                {
                    "title": f"{va.video_id} — A-only (no B within ±{match_window})",
                    "note": "Is this a true scene change that B missed, or redundant?",
                    "a": {
                        "image": str(va.image_paths[i]) if i < len(va.image_paths) else None,
                        "caption": f"A frame_idx={int(va.frame_idx[i])}",
                    },
                    "b": {"image": None, "caption": "no nearby B keyframe"},
                }
            )
        for i in um_b[:n_unmatched]:
            samples.append(
                {
                    "title": f"{vb.video_id} — B-only (no A within ±{match_window})",
                    "note": "Is this a true scene change that A missed, or redundant?",
                    "a": {"image": None, "caption": "no nearby A keyframe"},
                    "b": {
                        "image": str(vb.image_paths[i]) if i < len(vb.image_paths) else None,
                        "caption": f"B frame_idx={int(vb.frame_idx[i])}",
                    },
                }
            )

        # random aligned-ish pairs: pick random A index, nearest B
        if va.frame_idx.size and vb.frame_idx.size and va.image_paths and vb.image_paths:
            for _ in range(n_random):
                i = rng.randrange(len(va.frame_idx))
                x = int(va.frame_idx[i])
                j = int(np.argmin(np.abs(vb.frame_idx - x)))
                samples.append(
                    {
                        "title": f"{va.video_id} — random A vs nearest B",
                        "note": f"|Δframe|={abs(int(vb.frame_idx[j]) - x)}",
                        "a": {
                            "image": str(va.image_paths[i]) if i < len(va.image_paths) else None,
                            "caption": f"A frame_idx={x}",
                        },
                        "b": {
                            "image": str(vb.image_paths[j]) if j < len(vb.image_paths) else None,
                            "caption": f"B frame_idx={int(vb.frame_idx[j])}",
                        },
                    }
                )
    return samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two NII-UIT-style keyframe trees.")
    p.add_argument("--dir-a", type=Path, required=True, help="First extraction root")
    p.add_argument("--dir-b", type=Path, required=True, help="Second extraction root")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Write report.json, plots, and index.html here",
    )
    p.add_argument(
        "--match-window",
        type=int,
        default=5,
        help="Frames ± for matching A↔B (and optional BTC ref)",
    )
    p.add_argument(
        "--trake-gap",
        type=int,
        default=10,
        help="Gap threshold used for 'too sparse for TRAKE-sized windows'",
    )
    p.add_argument(
        "--ref-map-dir",
        type=Path,
        default=None,
        help="Optional BTC map-keyframes dir (VIDEO_ID.csv)",
    )
    p.add_argument("--n-gap-samples", type=int, default=2)
    p.add_argument("--n-unmatched-samples", type=int, default=3)
    p.add_argument("--n-random-samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args(argv)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs_a = list_video_dirs(args.dir_a)
    dirs_b = list_video_dirs(args.dir_b)
    common = sorted(set(dirs_a) & set(dirs_b))
    only_a = sorted(set(dirs_a) - set(dirs_b))
    only_b = sorted(set(dirs_b) - set(dirs_a))
    if not common:
        raise SystemExit("No overlapping VIDEO_ID folders with map.csv")

    per_video: list[dict] = []
    gaps_a_all: list[float] = []
    gaps_b_all: list[float] = []
    paired_maps: list[tuple[VideoMaps, VideoMaps]] = []

    for vid in common:
        va = load_video_maps(dirs_a[vid], vid)
        vb = load_video_maps(dirs_b[vid], vid)
        paired_maps.append((va, vb))

        sa = gap_stats(va.frame_idx, va.fps, args.trake_gap)
        sb = gap_stats(vb.frame_idx, vb.fps, args.trake_gap)
        ua = uniqueness_stats(va.embeddings)
        ub = uniqueness_stats(vb.embeddings)
        cov_ab = match_coverage(va.frame_idx, vb.frame_idx, args.match_window)
        cov_ba = match_coverage(vb.frame_idx, va.frame_idx, args.match_window)

        row: dict = {
            "video_id": vid,
            "a": {**sa, **ua},
            "b": {**sb, **ub},
            "a_covered_by_b": cov_ab,
            "b_covered_by_a": cov_ba,
        }

        if va.frame_idx.size > 1:
            gaps_a_all.extend(np.diff(va.frame_idx).astype(float).tolist())
        if vb.frame_idx.size > 1:
            gaps_b_all.extend(np.diff(vb.frame_idx).astype(float).tolist())

        if args.ref_map_dir:
            ref_csv = args.ref_map_dir.expanduser().resolve() / f"{vid}.csv"
            if ref_csv.is_file():
                ref = load_btc_map(ref_csv)
                row["ref_covered_by_a"] = match_coverage(ref, va.frame_idx, args.match_window)
                row["ref_covered_by_b"] = match_coverage(ref, vb.frame_idx, args.match_window)
                row["a_covered_by_ref"] = match_coverage(va.frame_idx, ref, args.match_window)
                row["b_covered_by_ref"] = match_coverage(vb.frame_idx, ref, args.match_window)

        per_video.append(row)
        safe_print(
            f"{vid}: A n={sa['n']} kf/s={sa['kf_per_s']} mean_gap={sa['mean_gap']} "
            f"| B n={sb['n']} kf/s={sb['kf_per_s']} mean_gap={sb['mean_gap']} "
            f"| A⊂B={cov_ab['coverage']} B⊂A={cov_ba['coverage']}"
        )

    def _mean_key(rows: list[dict], side: str, key: str):
        vals = [r[side][key] for r in rows if r[side].get(key) is not None]
        return float(np.mean(vals)) if vals else None

    aggregate = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n_common_videos": len(common),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "match_window": args.match_window,
        "trake_gap": args.trake_gap,
        "mean_kf_per_s_a": _mean_key(per_video, "a", "kf_per_s"),
        "mean_kf_per_s_b": _mean_key(per_video, "b", "kf_per_s"),
        "mean_gap_a": _mean_key(per_video, "a", "mean_gap"),
        "mean_gap_b": _mean_key(per_video, "b", "mean_gap"),
        "mean_frac_gaps_gt_trake_a": _mean_key(per_video, "a", "frac_gaps_gt_trake"),
        "mean_frac_gaps_gt_trake_b": _mean_key(per_video, "b", "frac_gaps_gt_trake"),
        "mean_consec_dist_a": _mean_key(per_video, "a", "mean_consec_dist"),
        "mean_consec_dist_b": _mean_key(per_video, "b", "mean_consec_dist"),
        "mean_a_covered_by_b": float(
            np.mean([r["a_covered_by_b"]["coverage"] for r in per_video if r["a_covered_by_b"]["coverage"] is not None])
        )
        if per_video
        else None,
        "mean_b_covered_by_a": float(
            np.mean([r["b_covered_by_a"]["coverage"] for r in per_video if r["b_covered_by_a"]["coverage"] is not None])
        )
        if per_video
        else None,
    }

    report = {"aggregate": aggregate, "per_video": per_video}
    report_json = out_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    safe_print(f"Wrote {report_json}")

    try:
        plot_gap_histograms(
            out_dir / "gaps_hist.png",
            gaps_a_all,
            gaps_b_all,
            args.label_a,
            args.label_b,
            args.trake_gap,
        )
        plot_density_bars(out_dir / "density_bars.png", per_video, args.label_a, args.label_b)
        safe_print(f"Wrote plots under {out_dir}")
    except Exception as exc:  # noqa: BLE001
        safe_print(f"Plotting skipped/failed: {exc}")

    samples = build_manual_samples(
        paired_maps,
        match_window=args.match_window,
        n_gap=args.n_gap_samples,
        n_unmatched=args.n_unmatched_samples,
        n_random=args.n_random_samples,
        seed=args.seed,
    )
    html_path = out_dir / "index.html"
    write_html_report(
        html_path,
        label_a=args.label_a,
        label_b=args.label_b,
        samples=samples,
        summary=aggregate,
    )
    safe_print(f"Open manual review: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
