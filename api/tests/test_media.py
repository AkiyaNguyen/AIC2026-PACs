from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.media import (
    router,
    resolve_keyframe_path,
    resolve_video_path,
    thumbnail_url_if_available,
    video_url_if_available,
)


class ResolveKeyframePathTests(TestCase):
    def test_resolves_zero_based_row_to_six_digit_webp(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                resolve_keyframe_path(root, "L21_V001", 158),
                root.resolve() / "L21_V001" / "000158.webp",
            )

    def test_rejects_unsafe_video_id(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                resolve_keyframe_path(Path(tmp), "../L21_V001", 0)

    def test_rejects_negative_row(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                resolve_keyframe_path(Path(tmp), "L21_V001", -1)

    def test_returns_url_only_when_keyframe_exists(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_dir = root / "L21_V001"
            video_dir.mkdir()
            (video_dir / "000158.webp").write_bytes(b"RIFFtestWEBP")
            with patch("api.media.get_keyframes_root", return_value=root):
                self.assertEqual(
                    thumbnail_url_if_available("L21_V001", 158),
                    "/media/keyframes/L21_V001/158",
                )
                self.assertIsNone(
                    thumbnail_url_if_available("L21_V001", 159)
                )


class ResolveVideoPathTests(TestCase):
    def test_resolves_video_to_batch_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                resolve_video_path(root, "L21_V001"),
                root.resolve() / "video_L21" / "L21_V001.mp4",
            )

    def test_rejects_unsafe_video_id(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                resolve_video_path(Path(tmp), "../L21_V001")

    def test_returns_url_only_when_video_exists(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "video_L21"
            batch_dir.mkdir()
            (batch_dir / "L21_V001.mp4").write_bytes(b"test-mp4")
            with patch("api.media.get_videos_root", return_value=root):
                self.assertEqual(
                    video_url_if_available("L21_V001"),
                    "/media/videos/L21_V001",
                )
                self.assertIsNone(video_url_if_available("L21_V002"))

    def test_video_endpoint_supports_byte_ranges(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "video_L21"
            batch_dir.mkdir()
            (batch_dir / "L21_V001.mp4").write_bytes(b"0123456789")

            app = FastAPI()
            app.include_router(router)
            with patch("api.media.get_videos_root", return_value=root):
                response = TestClient(app).get(
                    "/media/videos/L21_V001",
                    headers={"Range": "bytes=2-5"},
                )

            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, b"2345")
            self.assertEqual(response.headers["accept-ranges"], "bytes")
            self.assertEqual(response.headers["content-range"], "bytes 2-5/10")
            self.assertEqual(response.headers["content-type"], "video/mp4")
