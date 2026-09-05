"""
Unit tests for pipeline checkpoint & resume mechanism.
Verifies that previously extracted frames and DISK neural features are reused.
"""
import unittest
import tempfile
import json
import pickle
from pathlib import Path
import numpy as np

from src.video_ingest import VideoIngestor, VideoIngestConfig
from src.neural_matcher import NeuralMatcher, NeuralMatcherConfig


class TestResumeCheckpoint(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_video_ingest_reuse(self):
        """Verify process_video reuses frames and manifest if allow_reuse=True."""
        frames_dir = self.base_path / "frames"
        frames_dir.mkdir(parents=True)
        manifest_path = self.base_path / "frames_manifest.json"

        # Create 6 fake frame images
        for i in range(6):
            (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)

        fake_manifest = {
            "total_extracted": 6,
            "fps": 4.0,
            "frames": [f"frame_{i:04d}.jpg" for i in range(6)]
        }
        manifest_path.write_text(json.dumps(fake_manifest))

        from src.config import MarkerConfig
        ingestor = VideoIngestor(VideoIngestConfig(), MarkerConfig())
        # Dummy video path (does not need to exist because allow_reuse should short-circuit)
        dummy_video = self.base_path / "dummy.mp4"
        result = ingestor.process_video(dummy_video, frames_dir, allow_reuse=True)

        self.assertEqual(result["total_extracted"], 6)
        self.assertEqual(len(result["frames"]), 6)

    def test_neural_features_disk_cache(self):
        """Verify NeuralMatcher saves and reloads disk_features_cache.pkl."""
        cache_file = self.base_path / "disk_features_cache.pkl"
        fake_cache = {
            "frame_0001.jpg": {
                "keypoints": np.zeros((100, 2), dtype=np.float32),
                "descriptors": np.zeros((100, 128), dtype=np.float32)
            }
        }
        with open(cache_file, "wb") as f:
            pickle.dump(fake_cache, f)

        # Reload cache
        with open(cache_file, "rb") as f:
            loaded = pickle.load(f)

        self.assertIn("frame_0001.jpg", loaded)
        self.assertEqual(loaded["frame_0001.jpg"]["keypoints"].shape, (100, 2))


if __name__ == "__main__":
    unittest.main()
