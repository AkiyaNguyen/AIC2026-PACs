"""Load-once search runtime: galleries, query encoders, hybrid pipeline."""

from __future__ import annotations

import sys

import faiss
import numpy as np
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from engine.config import get_features_root
from engine.device import get_proper_device
from engine.gallery.encoders import (
    CLIP_DIM,
    CLIP_MODEL_ID,
    ClipTextQueryEncoder,
    SIGLIP_DIM,
    SIGLIP_MODEL_ID,
    Siglip2TextQueryEncoder,
)
from engine.gallery.loaders import (
    assert_clip_siglip_aligned,
    build_asr_by_video,
    build_row_index_map_info,
    load_asr_gallery,
    load_gallery_map,
    load_visual_tower,
)
from engine.query_language import ResolvedQueries, resolve_queries
from engine.search import HybridSearcher
from engine.search.config import SearchConfig


class SearchService:
    """Composition root: load galleries + encoders once; run hybrid search per query."""

    def __init__(self, device: str = "gpu") -> None:
        self.device = get_proper_device(device)
        root = get_features_root()

        self.clip_encoder, self.clip_index = load_visual_tower(
            root / "clip",
            model_id=CLIP_MODEL_ID,
            dim=CLIP_DIM,
            model_cls=CLIPModel,
            processor_cls=CLIPProcessor,
            encoder_cls=ClipTextQueryEncoder,
            device=self.device,
        )
        self.siglip_encoder, self.siglip_index = load_visual_tower(
            root / "SigLIP2",
            model_id=SIGLIP_MODEL_ID,
            dim=SIGLIP_DIM,
            model_cls=AutoModel,
            processor_cls=AutoProcessor,
            encoder_cls=Siglip2TextQueryEncoder,
            device=self.device,
        )
        self.asr_encoder, self.asr_embed_mat = load_asr_gallery(
            root / "asr_emb", self.device
        )

        clip_map = load_gallery_map(root / "clip" / "gallery_map.csv")
        asr_map = load_gallery_map(root / "asr_emb" / "gallery_map.csv")
        self.row_index_map_info = build_row_index_map_info(
            clip_map, root / "maps", ntotal=self.clip_index.ntotal
        )
        self.asr_by_video = build_asr_by_video(asr_map, root)

        assert_clip_siglip_aligned(root, self.clip_index, self.siglip_index)
        self.hybrid = HybridSearcher.from_service(self)

    def ann(
        self,
        encoder: ClipTextQueryEncoder | Siglip2TextQueryEncoder,
        index: faiss.Index,
        query: str,
        k: int,
    ) -> np.ndarray:
        q = encoder.encode_query(query)
        _scores, ids = index.search(np.ascontiguousarray(q.astype(np.float32)), k)
        return ids[0]

    def search(
        self,
        query_vi: str,
        query_en: str | None = None,
        *,
        cfg: SearchConfig | None = None,
        weight_visual: float = 0.8,
        weight_transcript: float = 0.2,
        weight_sem_text: float = 0.6,
        weight_bm25: float = 0.4,
    ) -> tuple[ResolvedQueries, list[dict]]:
        resolved = resolve_queries(query_vi, query_en)
        raw = self.hybrid.search(
            resolved.query_vi,
            resolved.query_en,
            weight_visual=weight_visual,
            weight_transcript=weight_transcript,
            weight_sem_text=weight_sem_text,
            weight_bm25=weight_bm25,
            cfg=cfg,
        )
        keys = (
            "score",
            "clip_row",
            "video_id",
            "pts_time",
            "row_idx_in_video",
            "frame_idx",
            "fps",
            "source",
        )
        hits = [{k: hit[k] for k in keys} for hit in raw]
        return resolved, hits


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    service = SearchService(device="cpu")
    print("FEATURES_ROOT", get_features_root())
    print("clip ntotal", service.clip_index.ntotal, "d", service.clip_index.d)
    print("siglip ntotal", service.siglip_index.ntotal, "d", service.siglip_index.d)
    print("asr_mat", service.asr_embed_mat.shape)
    n_meta = len(service.row_index_map_info["video_ids"])
    print("clip meta rows", n_meta, "asr videos", len(service.asr_by_video))

    query_vi = "chương trình 60 giây đài truyền hình thành phố Hồ Chí Minh"
    resolved, hits = service.search(
        query_vi,
        cfg=SearchConfig(num_candidates_vis=50, num_results=5),
        weight_visual=0.8,
        weight_transcript=0.2,
    )
    print(
        f"query_vi={resolved.query_vi!r}\n"
        f"query_en={resolved.query_en!r} ({resolved.query_en_source})"
    )
    for rank, hit in enumerate(hits, start=1):
        print(
            f"  {rank:2d}  score={hit['score']:.4f}  {hit['video_id']}  "
            f"kf={hit['row_idx_in_video']}  pts={hit['pts_time']:.2f}s"
        )
