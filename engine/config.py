""" load then refine config"""
# from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_features_root() -> Path:
    """FEATURES_ROOT from .env (absolute path to features/ tree)."""
    raw = os.getenv("FEATURES_ROOT")
    if not raw:
        raise EnvironmentError("FEATURES_ROOT is not set (e.g. in .env)")
    root = Path(raw)
    if not root.is_dir():
        raise FileNotFoundError(f"FEATURES_ROOT does not exist: {root}")
    return root


def get_keyframes_root() -> Path:
    """KEYFRAMES_ROOT containing flat VIDEO_ID/ keyframe folders."""
    raw = os.getenv("KEYFRAMES_ROOT")
    if not raw:
        raise EnvironmentError(
            "KEYFRAMES_ROOT is not set (e.g. media/keyframes in .env)"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"KEYFRAMES_ROOT does not exist: {root}")
    return root


def get_videos_root() -> Path:
    """VIDEOS_ROOT containing video_BATCH/VIDEO_ID.mp4 files."""
    raw = os.getenv("VIDEOS_ROOT")
    if not raw:
        raise EnvironmentError(
            "VIDEOS_ROOT is not set (e.g. media/videos in .env)"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VIDEOS_ROOT does not exist: {root}")
    return root
