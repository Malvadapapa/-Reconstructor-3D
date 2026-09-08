"""
Unit Tests for CameraConfig and Intrinsics Parsing.
Verifies that:
1. CameraConfig defaults to standard SIMPLE_RADIAL and shared single_camera.
2. Intrinsics parsing accurately extracts fx, fy, cx, cy and distortion.
"""
import unittest
import numpy as np

from src.config import CameraConfig
from src.sfm_reconstruction import parse_camera_intrinsics


class TestCameraConfig(unittest.TestCase):

    def test_camera_config_defaults(self):
        cam_cfg = CameraConfig()
        self.assertEqual(cam_cfg.model, "SIMPLE_RADIAL")
        self.assertTrue(cam_cfg.single_camera)
        self.assertTrue(cam_cfg.refine_focal_length)
        self.assertFalse(cam_cfg.refine_principal_point)
        self.assertFalse(cam_cfg.refine_extra_params)
        self.assertIsNone(cam_cfg.fx)
        self.assertIsNone(cam_cfg.cx)

    def test_parse_simple_radial_intrinsics(self):
        cam_dict = {
            "model_id": 2,
            "model_name": "SIMPLE_RADIAL",
            "width": 1920,
            "height": 1080,
            "params": np.array([1250.5, 960.0, 540.0, -0.015])
        }
        parsed = parse_camera_intrinsics(cam_dict)
        self.assertEqual(parsed["model_id"], 2)
        self.assertEqual(parsed["fx"], 1250.5)
        self.assertEqual(parsed["fy"], 1250.5)
        self.assertEqual(parsed["cx"], 960.0)
        self.assertEqual(parsed["cy"], 540.0)
        self.assertEqual(len(parsed["distortion"]), 1)
        self.assertAlmostEqual(parsed["distortion"][0], -0.015)

    def test_parse_pinhole_intrinsics(self):
        cam_dict = {
            "model_id": 1,
            "model_name": "PINHOLE",
            "width": 1280,
            "height": 720,
            "params": np.array([1000.0, 1005.0, 640.0, 360.0])
        }
        parsed = parse_camera_intrinsics(cam_dict)
        self.assertEqual(parsed["fx"], 1000.0)
        self.assertEqual(parsed["fy"], 1005.0)
        self.assertEqual(parsed["cx"], 640.0)
        self.assertEqual(parsed["cy"], 360.0)


if __name__ == "__main__":
    unittest.main()
