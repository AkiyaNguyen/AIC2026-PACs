from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from engine.config import get_keyframes_root, get_videos_root

router = APIRouter(prefix="/media", tags=["media"])

_VIDEO_ID_PATTERN = re.compile(r"^L\d+_V\d+$")


def resolve_keyframe_path(
    root: Path, video_id: str, row_idx_in_video: int
) -> Path:
    """Resolve a keyframe without allowing a route value to escape media root."""
    if not _VIDEO_ID_PATTERN.fullmatch(video_id) or row_idx_in_video < 0:
        raise ValueError("Invalid video_id or keyframe row")

    root = root.resolve()
    video_dir = (root / video_id).resolve()
    if video_dir.parent != root:
        raise ValueError("Invalid video_id")
    return video_dir / f"{row_idx_in_video:06d}.webp"


def thumbnail_url_if_available(
    video_id: str, row_idx_in_video: int
) -> str | None:
    """Return an API-relative thumbnail URL only when the local still exists."""
    try:
        path = resolve_keyframe_path(
            get_keyframes_root(), video_id, row_idx_in_video
        )
    except (EnvironmentError, FileNotFoundError, ValueError):
        return None
    if not path.is_file():
        return None
    return f"/media/keyframes/{video_id}/{row_idx_in_video}"


def resolve_video_path(root: Path, video_id: str) -> Path:
    """Resolve VIDEO_ID to video_BATCH/VIDEO_ID.mp4 within media root."""
    if not _VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Invalid video_id")

    root = root.resolve()
    batch = video_id.split("_", 1)[0]
    batch_dir = (root / f"video_{batch}").resolve()
    if batch_dir.parent != root:
        raise ValueError("Invalid video_id")
    return batch_dir / f"{video_id}.mp4"


def video_url_if_available(video_id: str) -> str | None:
    """Return an API-relative video URL only when the local MP4 exists."""
    try:
        path = resolve_video_path(get_videos_root(), video_id)
    except (EnvironmentError, FileNotFoundError, ValueError):
        return None
    if not path.is_file():
        return None
    return f"/media/videos/{video_id}"


@router.get("/keyframes/{video_id}/{row_idx_in_video}")
def get_keyframe(video_id: str, row_idx_in_video: int) -> FileResponse:
    try:
        root = get_keyframes_root()
    except (EnvironmentError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        path = resolve_keyframe_path(root, video_id, row_idx_in_video)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Keyframe not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Keyframe not found")

    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/videos/{video_id}")
def get_video(video_id: str) -> FileResponse:
    try:
        root = get_videos_root()
    except (EnvironmentError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        path = resolve_video_path(root, video_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Video not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    # Starlette FileResponse handles byte-range requests, so browsers can seek
    # without downloading the complete source video first.
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=86400"},
    )
