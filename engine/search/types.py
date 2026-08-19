"""Shared datatypes for hybrid keyframe search."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QueryFeatures:
    """Prepared per-query state: union pool + score vectors for instant rerank."""

    all_rows: np.ndarray
    s_vis: np.ndarray
    r_sem: np.ndarray
    r_bm25: np.ndarray
    source: dict[int, set[str]] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict) ## execution time


@dataclass(frozen=True)
class TranscriptProposals:
    """Text-channel keyframe candidates before union with visual pool."""

    bm25: dict[int, float] # index of keyframes, bm25 score of keyframes
    semantic: dict[int, float] # index of keyframes, semantic score of keyframes


@dataclass(frozen=True)
class CandidatePool:
    """Union of visual and text-only keyframe rows with aligned visual scores."""

    all_rows: np.ndarray
    s_vis: np.ndarray
    source: dict[int, set[str]]
    n_vis: int
    n_text: int
