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
