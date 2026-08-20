from unittest import TestCase

from pydantic import ValidationError

from api.schemas import Hit, SearchRequest


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
            source="bm25+sem+vis",
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
                source="vis",
            )


class SearchRequestSchemaTests(TestCase):
    def test_maps_hybrid_controls_to_search_config(self):
        request = SearchRequest(
            query="xuất khẩu gạo",
            num_candidates_visual=120,
            num_results=25,
            bm25_top_segments=80,
            sem_top_segments=70,
            delta=1.5,
            segment_min_gap=0.8,
            segment_pad=0.4,
            rrf_k=55,
        )

        config = request.to_search_config()
        self.assertEqual(config.num_candidates_vis, 120)
        self.assertEqual(config.num_results, 25)
        self.assertEqual(config.bm25_top_segments, 80)
        self.assertEqual(config.sem_top_segments, 70)
        self.assertEqual(config.delta, 1.5)
        self.assertEqual(config.segment_min_gap, 0.8)
        self.assertEqual(config.segment_pad, 0.4)
        self.assertEqual(config.rrf_k, 55)

    def test_rejects_all_zero_retrieval_weights(self):
        with self.assertRaises(ValidationError):
            SearchRequest(
                query="test",
                weight_visual=0.0,
                weight_transcript=0.0,
            )

    def test_rejects_empty_transcript_mix(self):
        with self.assertRaises(ValidationError):
            SearchRequest(
                query="test",
                weight_visual=0.5,
                weight_transcript=0.5,
                weight_sem_text=0.0,
                weight_bm25=0.0,
            )
