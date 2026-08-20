from unittest import TestCase

from pydantic import ValidationError

from api.schemas import Hit


class HitSchemaTests(TestCase):
    def test_accepts_positive_source_fps(self):
        hit = Hit(
            rank=1,
            score=0.9,
            clip_row=0,
            video_id="L21_V001",
            pts_time=10.0,
            row_idx_in_video=1,
            frame_idx=300,
            fps=30.0,
        )
        self.assertEqual(hit.fps, 30.0)

    def test_rejects_non_positive_source_fps(self):
        with self.assertRaises(ValidationError):
            Hit(
                rank=1,
                score=0.9,
                clip_row=0,
                video_id="L21_V001",
                pts_time=10.0,
                row_idx_in_video=1,
                frame_idx=300,
                fps=0.0,
            )
