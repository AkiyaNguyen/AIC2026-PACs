"""Embed a tiny still tree into BTC-shaped VIDEO_ID.npy without loading CLIP."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.extract_features.engine.embed import embed_tree


class FakeEmbedder:
    name = "fake"

    def embed_pils(self, images: list) -> np.ndarray:
        n = len(images)
        if n == 0:
            return np.zeros((0, 4), dtype=np.float32)
        # Distinct rows so we can check order; already L2-normalizable.
        rows = np.eye(max(n, 4), dtype=np.float32)[:n, :4]
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.maximum(norms, 1e-12)


def _write_video_dir(root: Path, video_id: str, n_frames: int) -> None:
    folder = root / video_id
    folder.mkdir(parents=True)
    with (folder / "map.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        w.writeheader()
        for i in range(n_frames):
            Image.new("RGB", (8, 8), color=(i * 40, 0, 0)).save(
                folder / f"{i:06d}.webp", format="WEBP"
            )
            w.writerow(
                {
                    "n": i + 1,
                    "pts_time": f"{i * 0.4:.6f}",
                    "fps": 25.0,
                    "frame_idx": i * 10,
                }
            )


class EmbedTreeTest(unittest.TestCase):
    def test_writes_npy_with_one_row_per_map_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "stills"
            out = Path(tmp) / "gallery"
            _write_video_dir(src, "L21_V001", 3)
            summary = embed_tree(src, out, FakeEmbedder(), batch_size=2)
            npy = out / "L21_V001.npy"
            self.assertTrue(npy.is_file())
            feats = np.load(npy)
            self.assertEqual(feats.shape, (3, 4))
            self.assertEqual(feats.dtype, np.float32)
            norms = np.linalg.norm(feats, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)
            self.assertEqual(summary["n_videos"], 1)
            self.assertEqual(summary["videos"][0]["n_rows"], 3)

    def test_empty_root_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "empty"
            src.mkdir()
            out = Path(tmp) / "gallery"
            with self.assertRaises(SystemExit):
                embed_tree(src, out, FakeEmbedder())

    def test_copy_embeddings_writes_flat_npy_without_embedder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "stills"
            out = Path(tmp) / "gallery"
            _write_video_dir(src, "L21_V001", 3)
            src_emb = np.eye(3, 8, dtype=np.float32)
            np.save(src / "L21_V001" / "embeddings.npy", src_emb)
            summary = embed_tree(
                src, out, embedder=None, copy_embeddings=True
            )
            got = np.load(out / "L21_V001.npy")
            self.assertEqual(got.shape, (3, 8))
            self.assertEqual(summary["copied"], True)
            self.assertEqual(summary["videos"][0]["source"], "embeddings.npy")

    def test_copy_embeddings_missing_file_exits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "stills"
            out = Path(tmp) / "gallery"
            _write_video_dir(src, "L21_V001", 3)
            with self.assertRaises(SystemExit):
                embed_tree(src, out, embedder=None, copy_embeddings=True)


if __name__ == "__main__":
    unittest.main()
