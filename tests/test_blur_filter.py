"""
Unit test for Laplacian sharpness / blur filter.
"""
import unittest
import cv2
import numpy as np
from src.video_ingest import VideoIngestor


class TestBlurFilter(unittest.TestCase):

    def test_sharp_vs_blurred_image(self):
        # Create a sharp image with high frequency edges (checkerboard pattern)
        sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
        sharp_img[::20, :, :] = 255
        sharp_img[:, ::20, :] = 255

        # Create heavily blurred version
        blurred_img = cv2.GaussianBlur(sharp_img, (25, 25), 10.0)

        sharp_var = VideoIngestor.compute_sharpness(sharp_img)
        blurred_var = VideoIngestor.compute_sharpness(blurred_img)

        print(f"\n[TestBlurFilter] Sharp Var: {sharp_var:.1f} | Blurred Var: {blurred_var:.1f}")
        self.assertGreater(sharp_var, blurred_var)
        self.assertGreater(sharp_var, 80.0)
        self.assertLess(blurred_var, 50.0)


if __name__ == "__main__":
    unittest.main()
