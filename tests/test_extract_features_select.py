"""Keep/drop selection: first kept, near-dup dropped, large change kept."""

from __future__ import annotations

import unittest

import numpy as np

from tools.extract_features.engine.extract import select_keyframes


def _l2(rows: list[list[float]]) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-12)).astype(np.float32)


class SelectKeyframesTest(unittest.TestCase):
    def test_empty_returns_empty(self) -> None:
        self.assertEqual(select_keyframes(np.zeros((0, 4), dtype=np.float32), 0.15), [])

    def test_first_candidate_always_kept(self) -> None:
        emb = _l2([[1.0, 0.0, 0.0]])
        self.assertEqual(select_keyframes(emb, 0.15), [0])

    def test_near_duplicate_is_dropped(self) -> None:
        emb = _l2([[1.0, 0.0, 0.0], [1.0, 0.01, 0.0]])
        self.assertEqual(select_keyframes(emb, 0.15), [0])

    def test_large_change_is_kept(self) -> None:
        emb = _l2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(select_keyframes(emb, 0.15), [0, 1])

    def test_compares_against_last_kept_not_previous_candidate(self) -> None:
        # middle is a near-dup of first (dropped); third differs from first (kept)
        emb = _l2(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.01, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.assertEqual(select_keyframes(emb, 0.15), [0, 2])


class ParseDeviceNameTest(unittest.TestCase):
    def test_cpu(self) -> None:
        from tools.extract_features.engine.clip import parse_device_name

        self.assertEqual(parse_device_name("cpu"), "cpu")

    def test_gpu_and_cuda_map_to_cuda(self) -> None:
        from tools.extract_features.engine.clip import parse_device_name

        self.assertEqual(parse_device_name("gpu"), "cuda")
        self.assertEqual(parse_device_name("cuda"), "cuda")
        self.assertEqual(parse_device_name("GPU"), "cuda")

    def test_invalid_raises(self) -> None:
        from tools.extract_features.engine.clip import parse_device_name

        with self.assertRaises(SystemExit):
            parse_device_name("tpu")


if __name__ == "__main__":
    unittest.main()
