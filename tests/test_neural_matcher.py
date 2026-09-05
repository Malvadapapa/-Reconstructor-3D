"""
Unit test for NeuralMatcher (DISK + LightGlue).
"""
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.config import NeuralMatcherConfig
from src.neural_matcher import NeuralMatcher


class TestNeuralMatcher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("output/test_neural_scratch")
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Create two textured images with a relative shift
        np.random.seed(42)
        base_texture = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        # Add high-contrast circles and markers
        for _ in range(30):
            cx, cy = np.random.randint(40, 260, 2)
            r = np.random.randint(8, 25)
            col = tuple(int(c) for c in np.random.randint(0, 255, 3))
            cv2.circle(base_texture, (cx, cy), r, col, -1)

        # Image 0
        cls.img0_path = cls.test_dir / "img0.jpg"
        cv2.imwrite(str(cls.img0_path), base_texture)

        # Image 1 (shifted by 15px with homography/affine)
        M = np.float32([[1, 0, 15], [0, 1, 10]])
        shifted = cv2.warpAffine(base_texture, M, (300, 300))
        cls.img1_path = cls.test_dir / "img1.jpg"
        cv2.imwrite(str(cls.img1_path), shifted)

    def test_disk_extraction_and_lightglue_matching(self):
        config = NeuralMatcherConfig(enabled=True, device="cpu", filter_threshold=0.1)
        matcher = NeuralMatcher(config)

        # 1. Feature extraction
        feats0 = matcher.extract_frame_features(self.img0_path)
        feats1 = matcher.extract_frame_features(self.img1_path)

        self.assertIn("keypoints", feats0)
        self.assertIn("descriptors", feats0)
        self.assertGreater(len(feats0["keypoints"]), 10)
        self.assertGreater(len(feats1["keypoints"]), 10)

        # 2. Matching
        matches = matcher.match_pair(feats0, feats1)
        self.assertIsInstance(matches, np.ndarray)
        self.assertGreater(len(matches), 5, f"Expected > 5 matches, got {len(matches)}")
        print(f"\n[TestNeuralMatcher] DISK kpts: {len(feats0['keypoints'])}, {len(feats1['keypoints'])} | LightGlue matches: {len(matches)}")

    def test_generate_image_pairs_and_device_property(self):
        config = NeuralMatcherConfig(enabled=True, device="cpu")
        matcher = NeuralMatcher(config)

        # Verify device property works
        self.assertIsNotNone(matcher.device)
        self.assertEqual(matcher.device.type, "cpu")

        # Test small set (exhaustive)
        small_names = [f"frame_{i:03d}.jpg" for i in range(5)]
        pairs_small = matcher.generate_image_pairs(small_names)
        self.assertEqual(len(pairs_small), 10)  # 5*4/2 = 10

        # Test larger set on CPU (sliding window + loop closure)
        large_names = [f"frame_{i:03d}.jpg" for i in range(60)]
        pairs_large = matcher.generate_image_pairs(large_names)
        self.assertGreater(len(pairs_large), 50)
        self.assertLess(len(pairs_large), 1770)  # should be ~300 pairs, not 1770
        print(f"[TestNeuralMatcher] Generated {len(pairs_large)} pairs for 60 frames on CPU.")


if __name__ == "__main__":
    unittest.main()

