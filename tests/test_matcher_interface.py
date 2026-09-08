"""
Unit Tests for Matcher Interface and Decoupling.
Verifies that:
1. BaseMatcher interface is properly implemented by SIFTMatcher and DISKLightGlueMatcher.
2. MatchingResult dataclass contains all required diagnostic fields.
"""
import unittest
from pathlib import Path

from src.matcher_interface import BaseMatcher, MatchingResult
from src.matcher import SIFTMatcher, DISKLightGlueMatcher, NeuralMatcher
from src.config import MatcherConfig, NeuralMatcherConfig, CameraConfig


class TestMatcherInterface(unittest.TestCase):

    def test_sift_matcher_is_base_matcher(self):
        cfg = MatcherConfig(engine="sift")
        matcher = SIFTMatcher(cfg)
        self.assertIsInstance(matcher, BaseMatcher)

    def test_neural_matcher_is_base_matcher(self):
        cfg = NeuralMatcherConfig(enabled=True)
        matcher = DISKLightGlueMatcher(cfg)
        self.assertIsInstance(matcher, BaseMatcher)
        self.assertIs(DISKLightGlueMatcher, NeuralMatcher)

    def test_matching_result_structure(self):
        res = MatchingResult(
            engine_name="SIFT",
            num_images=10,
            total_pairs_processed=45,
            valid_pairs=40,
            inliers_mean=120.5,
            inliers_median=110.0,
            inliers_p10=30.0,
            inliers_p90=200.0,
            database_path=Path("output/test/database.db"),
            metadata={}
        )
        self.assertEqual(res.engine_name, "SIFT")
        self.assertEqual(res.num_images, 10)
        self.assertEqual(res.valid_pairs, 40)
        self.assertEqual(res.inliers_mean, 120.5)


if __name__ == "__main__":
    unittest.main()
