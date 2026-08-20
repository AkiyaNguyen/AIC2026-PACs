from unittest import TestCase
from unittest.mock import patch

from api.app import search
from api.schemas import SearchRequest
from engine.query_language import ResolvedQueries


class FakeSearchService:
    def __init__(self) -> None:
        self.call: tuple[tuple, dict] | None = None

    def search(self, *args, **kwargs) -> tuple[ResolvedQueries, list[dict]]:
        self.call = (args, kwargs)
        return ResolvedQueries(
            query_vi="xuất khẩu gạo",
            query_en="Vietnam rice export",
            query_en_source="user",
        ), [
            {
                "score": 0.91,
                "clip_row": 42,
                "video_id": "L21_V001",
                "pts_time": 10.0,
                "row_idx_in_video": 3,
                "frame_idx": 300,
                "fps": 30.0,
                "source": "bm25+sem+vis",
            }
        ]


class SearchEndpointContractTests(TestCase):
    def test_passes_hybrid_config_and_enriches_media_urls(self):
        service = FakeSearchService()
        request = SearchRequest(
            query_vi="xuất khẩu gạo",
            query_en="Vietnam rice export",
            num_candidates_visual=120,
            num_results=10,
            weight_visual=0.4,
            weight_transcript=0.6,
            weight_sem_text=0.3,
            weight_bm25=0.7,
            bm25_top_segments=80,
            sem_top_segments=70,
        )

        with (
            patch("api.app.search_service", service),
            patch(
                "api.app.thumbnail_url_if_available",
                return_value="/media/keyframes/L21_V001/3",
            ),
            patch(
                "api.app.video_url_if_available",
                return_value="/media/videos/L21_V001",
            ),
        ):
            response = search(request)

        self.assertIsNotNone(service.call)
        args, kwargs = service.call or ((), {})
        self.assertEqual(args, ("xuất khẩu gạo", "Vietnam rice export"))
        self.assertEqual(kwargs["cfg"].num_candidates_vis, 120)
        self.assertEqual(kwargs["cfg"].bm25_top_segments, 80)
        self.assertEqual(kwargs["cfg"].sem_top_segments, 70)
        self.assertEqual(kwargs["weight_visual"], 0.4)
        self.assertEqual(kwargs["weight_transcript"], 0.6)
        self.assertEqual(kwargs["weight_sem_text"], 0.3)
        self.assertEqual(kwargs["weight_bm25"], 0.7)

        self.assertEqual(response.query_vi, "xuất khẩu gạo")
        self.assertEqual(response.query_en, "Vietnam rice export")
        self.assertEqual(response.query_en_source, "user")

        hit = response.hits[0]
        self.assertEqual(hit.fps, 30.0)
        self.assertEqual(hit.source, "bm25+sem+vis")
        self.assertEqual(hit.thumbnail_url, "/media/keyframes/L21_V001/3")
        self.assertEqual(hit.video_url, "/media/videos/L21_V001")
