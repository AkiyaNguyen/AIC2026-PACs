from __future__ import annotations

import csv
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from engine.config import get_features_root
from engine.utils import asr_rows_for_clip_rows, get_proper_device, max_asr_scores

def _load_gallery_map(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}; run the merge-gallery notebook first: {path}"
        )
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def _read_asr_intervals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    starts, ends = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        starts.append(float(obj["start"]))
        ends.append(float(obj["end"]))
    return np.asarray(starts, dtype=np.float32), np.asarray(ends, dtype=np.float32)

# Hard-coded for now
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_DIM = 512
ASR_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ASR_DIM = 384


class CLIPQueryEncoder:
    """CLIP text tower only. Does not own the FAISS index."""

    def __init__(
        self,
        model: CLIPModel,
        processor: CLIPProcessor,
        device: torch.device,
        normalize_embeddings: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.normalize_embeddings = normalize_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            if self.normalize_embeddings:
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return feat.cpu().numpy().astype(np.float32)


class ASRQueryEncoder:
    """MiniLM text encoder only. Does not own the segment gallery."""

    def __init__(
        self,
        model: SentenceTransformer,
        device: torch.device,
        normalize_embeddings: bool = True,
    ):
        self.model = model
        self.device = device
        self.normalize_embeddings = normalize_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        feat = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(feat, dtype=np.float32)


class Embedder:
    """Load-once: CLIP encoder + full FAISS index; ASR encoder + segment matrix.

    Search (later): CLIP over the whole index; ASR only among CLIP candidates.
    No shared ABC — both encoders just implement encode_query().
    """

    def __init__(self, device: str = "gpu"):
        self.device = get_proper_device(device)
        self.clip_encoder, self.clip_index = self._load_clip()
        self.asr_encoder, self.asr_embed_mat = self._load_asr()
        self._load_keyframe_asr_maps()

    def _load_clip(self) -> tuple[CLIPQueryEncoder, faiss.Index]:
        clip_dir = get_features_root() / "clip"
        if not clip_dir.is_dir():
            raise FileNotFoundError(f"Features root missing clip/: {clip_dir}")

        model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device).eval()
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        encoder = CLIPQueryEncoder(model, processor, self.device, normalize_embeddings=True)

        index_path = clip_dir / "index.faiss"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing CLIP FAISS index: {index_path}")
        index = faiss.read_index(str(index_path))
        if index.d != CLIP_DIM:
            raise ValueError(f"Expected CLIP index dim {CLIP_DIM}, got {index.d}")

        return encoder, index

    def _load_asr(self) -> tuple[ASRQueryEncoder, np.ndarray]:
        asr_dir = get_features_root() / "asr_emb"
        if not asr_dir.is_dir():
            raise FileNotFoundError(f"Features root missing asr_emb/: {asr_dir}")

        model = SentenceTransformer(ASR_MODEL_ID, device=str(self.device))
        encoder = ASRQueryEncoder(model, self.device, normalize_embeddings=True)

        cache = asr_dir / "embeddings.npy"
        if cache.is_file():
            asr_embed_mat = np.load(cache)
        else:
            npy_paths = sorted(asr_dir.glob("*/*.npy"))
            if not npy_paths:
                npy_paths = sorted(p for p in asr_dir.glob("*.npy") if p.name != "embeddings.npy")
            npy_paths = [p for p in npy_paths if p.name != "embeddings.npy"]
            if not npy_paths:
                raise FileNotFoundError(f"No per-video .npy under {asr_dir}")
            blocks = [np.load(p).astype(np.float32, copy=False) for p in npy_paths]
            asr_embed_mat = np.concatenate(blocks, axis=0)
            np.save(cache, asr_embed_mat)

        if asr_embed_mat.ndim != 2 or asr_embed_mat.shape[-1] != ASR_DIM:
            raise ValueError(
                f"Expected ASR gallery (*, {ASR_DIM}), got {asr_embed_mat.shape}"
            )
        return encoder, asr_embed_mat

    def _load_keyframe_asr_maps(self) -> None:
        """CLIP FAISS row → (video_id, pts_time); video_id → ASR slice + intervals."""
        root = get_features_root()
        clip_map = _load_gallery_map(root / "clip" / "gallery_map.csv") # map vid_name -> start_row, n_rows in index matrix
        asr_map = _load_gallery_map(root / "asr_emb" / "gallery_map.csv") # map vid_name -> start_row, n_rows in asr matrix
        maps_dir = root / "maps"
        asr_dir = root / "asr_emb"

        video_ids: list[str] = []
        pts_list: list[float] = []
        n_clip = 0
        for row in clip_map:
            video_id = row["video_id"]
            n = int(row["n_rows"])
            map_csv = maps_dir / f"{video_id}.csv" # map vid_name -> pts_time for each keyframe
            if not map_csv.is_file():
                raise FileNotFoundError(f"Missing keyframe map: {map_csv}")
            with map_csv.open(encoding="utf-8", newline="") as f:
                pts = [float(r["pts_time"]) for r in csv.DictReader(f)]
            if len(pts) < n:
                raise ValueError(
                    f"{video_id}: map.csv has {len(pts)} rows, clip gallery n_rows={n}"
                )
            video_ids.extend([video_id] * n)
            pts_list.extend(pts[:n])
            n_clip += n

        assert n_clip == self.clip_index.ntotal, \
            f"clip gallery_map covers {n_clip} rows, FAISS ntotal={self.clip_index.ntotal}"

        self.clip_video_id = video_ids # i_th row of index matrix is from video_ids[i]
        self.clip_pts = np.asarray(pts_list, dtype=np.float32) # i_th row of index matrix is at pts_list[i] seconds

        asr_by_video: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
        for row in asr_map:
            video_id = row["video_id"]
            start_row = int(row["start_row"])
            n = int(row["n_rows"])

            jsonl = asr_dir / video_id.split("_")[0] / f"{video_id}.jsonl"

            if jsonl is None:
                starts = np.zeros(0, dtype=np.float32)
                ends = np.zeros(0, dtype=np.float32)
            else:
                starts, ends = _read_asr_intervals(jsonl)
            assert len(starts) == n and len(ends) == n, \
                f"{video_id}: asr jsonl segs={len(starts)}={len(ends)} gallery n_rows={n}"
            asr_by_video[video_id] = (start_row, starts, ends)
        self.asr_by_video = asr_by_video # map vid_name -> (start_row, list n of starts, list n of ends)

    def search(
        self,
        query: str,
        num_candidates: int = 500,
        num_results: int = 100,
        weight_clip: float = 0.8,
        weight_asr: float = 0.2,
        delta: float = 3.0,
    ) -> list[tuple[float, int]]:
        embed_for_clip = self.clip_encoder.encode_query(query)
        embed_for_asr = self.asr_encoder.encode_query(query)

        k = min(num_candidates, self.clip_index.ntotal)
        sv, ids = self.clip_index.search(
            np.ascontiguousarray(embed_for_clip.astype(np.float32)), k
        )
        clip_rows = ids[0]
        visual = sv[0]

        row_lists = asr_rows_for_clip_rows(
            clip_rows,
            self.clip_video_id,
            self.clip_pts,
            self.asr_by_video,
            delta=delta,
        )
        ra = max_asr_scores(embed_for_asr, self.asr_embed_mat, row_lists)
        total = weight_clip * visual + weight_asr * ra
        order = np.argsort(-total)[:num_results]
        return [(float(total[i]), int(clip_rows[i])) for i in order]

if __name__ == "__main__":
    embedder = Embedder(device="cpu")
    print("FEATURES_ROOT", get_features_root())
    print("clip ntotal", embedder.clip_index.ntotal, "d", embedder.clip_index.d)
    print("asr_mat", embedder.asr_embed_mat.shape)
    print("clip meta rows", len(embedder.clip_video_id), "asr videos", len(embedder.asr_by_video))
    assert len(embedder.clip_video_id) == embedder.clip_index.ntotal
    assert embedder.clip_pts.shape[0] == embedder.clip_index.ntotal

    query = "chương trình 60 giây đài truyền hình thành phố Hồ Chí Minh"
    print("\nQuery:", query)

    mixed = embedder.search(
        query, num_candidates=50, num_results=10, weight_clip=0.8, weight_asr=0.2
    )
    visual_only = embedder.search(
        query, num_candidates=50, num_results=10, weight_clip=1.0, weight_asr=0.0
    )

    print("\n#  mixed (0.8 clip + 0.2 asr)")
    for rank, (score, clip_row) in enumerate(mixed, start=1):
        vid = embedder.clip_video_id[clip_row]
        pts = float(embedder.clip_pts[clip_row])
        print(f"  {rank:2d}  score={score:.4f}  {vid}  pts={pts:.2f}s  clip_row={clip_row}")

    print("\n#  visual only (w_asr=0)")
    for rank, (score, clip_row) in enumerate(visual_only, start=1):
        vid = embedder.clip_video_id[clip_row]
        pts = float(embedder.clip_pts[clip_row])
        print(f"  {rank:2d}  score={score:.4f}  {vid}  pts={pts:.2f}s  clip_row={clip_row}")

    mixed_ids = [r for _, r in mixed]
    vis_ids = [r for _, r in visual_only]
    print("\norder changed vs visual-only:", mixed_ids != vis_ids)
