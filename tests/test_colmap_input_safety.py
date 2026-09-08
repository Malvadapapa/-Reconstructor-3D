"""
Unit Tests for COLMAP Input Safety and Anti-Contamination Verification.
Verifies that:
1. Only valid frame images are processed.
2. Any subdirectory (annotated/, debug/) or debug file triggers an immediate abort.
3. image_list.txt is correctly generated and populated.
"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

from src.sfm_reconstruction import validate_sfm_image_inputs


class TestColmapInputSafety(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.frames_dir = self.root_path / "frames"
        self.sfm_dir = self.root_path / "sfm"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.sfm_dir.mkdir(parents=True, exist_ok=True)

        # Create 5 dummy valid frames
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(5):
            cv2.imwrite(str(self.frames_dir / f"frame_{i:04d}.jpg"), dummy_img)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_frames_accepted(self):
        """Clean frames directory must validate cleanly and generate image_list.txt."""
        valid_names = validate_sfm_image_inputs(self.frames_dir, self.sfm_dir)
        self.assertEqual(len(valid_names), 5)
        self.assertEqual(valid_names[0], "frame_0000.jpg")

        image_list_file = self.sfm_dir / "image_list.txt"
        self.assertTrue(image_list_file.exists())
        with open(image_list_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(lines, [f"frame_{i:04d}.jpg" for i in range(5)])

    def test_subdirectory_contamination_aborts(self):
        """Presence of subdirectories (like annotated/) must raise ValueError."""
        annotated_sub = self.frames_dir / "annotated"
        annotated_sub.mkdir()
        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(annotated_sub / "frame_0000_detected.jpg"), dummy_img)

        with self.assertRaises(ValueError) as ctx:
            validate_sfm_image_inputs(self.frames_dir, self.sfm_dir)
        self.assertIn("Unauthorized subdirectories", str(ctx.exception))

    def test_annotated_debug_file_aborts(self):
        """Presence of annotated or debug files in top-level frames/ must raise ValueError."""
        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(self.frames_dir / "frame_annotated_debug.jpg"), dummy_img)

        with self.assertRaises(ValueError) as ctx:
            validate_sfm_image_inputs(self.frames_dir, self.sfm_dir)
        self.assertIn("Unauthorized debug or invalid files", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
