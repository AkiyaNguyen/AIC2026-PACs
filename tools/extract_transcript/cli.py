"""argparse CLI: video → wav → Whisper large-v3 JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.extract_transcript.engine.transcript_extract import MODEL, run_extract
from tools.util import get_proper_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.extract_transcript",
        description=(
            "Extract 16 kHz WAV, then Whisper large-v3. "
            "Writes VIDEO_ID.wav under --audio-dir and VIDEO_ID.jsonl under --out-dir."
        ),
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Video file(s) and/or folder(s) for one batch (e.g. Videos_L21)",
    )
    p.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Where to write extracted WAV (e.g. features/audio/L21)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where to write JSONL (e.g. features/asr/L21)",
    )
    p.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="cpu (default) or gpu (CUDA, else MPS; Whisper uses CPU if MPS)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for p in args.inputs:
        if not p.exists():
            raise SystemExit(f"Input not found: {p}")
    device = get_proper_device(args.device)
    print(f"device={device} model={MODEL}")
    print(f"audio-dir={args.audio_dir.resolve()}")
    print(f"out-dir={args.out_dir.resolve()}")
    run_extract(args.inputs, args.audio_dir, args.out_dir, device)
    return 0
