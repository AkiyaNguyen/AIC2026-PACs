"""Extract audio from videos, then Whisper ASR → JSONL segments."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

MODEL = "large-v3"
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


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
            if p.suffix.lower() not in VIDEO_EXTS:
                raise SystemExit(f"Not a video file: {p}")
            found.append(p)
            continue
        nested = sorted(
            q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in VIDEO_EXTS
        )
        if not nested:
            raise SystemExit(f"No videos under {p}")
        found.extend(nested)
    out: list[Path] = []
    seen: set[Path] = set()
    for v in found:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def video_to_audio(video_path: Path, wav_path: Path) -> None:
    """ffmpeg: 16 kHz mono WAV (what Whisper expects)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(f"ffmpeg failed for {video_path.name}: {err[-500:]}")


def _whisper_runtime(device: torch.device) -> tuple[str, str]:
    """faster-whisper device + compute_type. No MPS in CTranslate2."""
    if device.type == "cuda":
        return "cuda", "float16"
    if device.type == "mps":
        safe_print("faster-whisper has no MPS; using cpu")
    return "cpu", "int8"


def transcribe_wav(model, wav_path: Path) -> list[dict]:
    segments, _info = model.transcribe(str(wav_path), vad_filter=True)
    rows: list[dict] = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        rows.append(
            {
                "start": round(float(s.start), 3),
                "end": round(float(s.end), 3),
                "text": text,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_extract(
    inputs: list[Path],
    audio_dir: Path,
    out_dir: Path,
    device: torch.device,
) -> None:
    from faster_whisper import WhisperModel

    videos = discover_videos(inputs)
    audio_dir = audio_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    wdevice, compute = _whisper_runtime(device)
    safe_print(f"Loading Whisper {MODEL} device={wdevice} compute={compute} ...")
    model = WhisperModel(MODEL, device=wdevice, compute_type=compute)

    for i, video in enumerate(videos, start=1):
        vid = video.stem
        wav_path = audio_dir / f"{vid}.wav"
        jsonl_path = out_dir / f"{vid}.jsonl"
        safe_print(f"[{i}/{len(videos)}] {vid}")
        video_to_audio(video, wav_path)
        rows = transcribe_wav(model, wav_path)
        write_jsonl(jsonl_path, rows)
        safe_print(f"  {wav_path.name} → {jsonl_path.name} ({len(rows)} segments)")

    safe_print(f"Done: {len(videos)} video(s) → {out_dir}")
