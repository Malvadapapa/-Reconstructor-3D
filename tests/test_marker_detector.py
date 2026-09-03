"""
Unit test for AprilTag / ArUco marker detection and corner extraction.
"""
import unittest
import cv2
import numpy as np
from src.marker_detector import MarkerDetector
from src.marker_generator import generate_apriltag_matrix_36h11


class TestMarkerDetector(unittest.TestCase):

    def test_apriltag_detection(self):
        # Generate clean tag36h11 id 0
        tag_bits = generate_apriltag_matrix_36h11(tag_id=0)
        tag_resized = cv2.resize(tag_bits, (200, 200), interpolation=cv2.INTER_NEAREST)
        
        # Place in larger image with white border
        canvas = np.full((400, 400, 3), 255, dtype=np.uint8)
        if len(tag_resized.shape) == 2:
            tag_resized = cv2.cvtColor(tag_resized, cv2.COLOR_GRAY2BGR)
        canvas[100:300, 100:300] = tag_resized

        detector = MarkerDetector(family="tag36h11")
        detections = detector.detect(canvas)

        print(f"\n[TestMarkerDetector] Detected {len(detections)} markers.")
        self.assertGreaterEqual(len(detections), 1)
        self.assertEqual(detections[0].marker_id, 0)
        self.assertEqual(detections[0].corners.shape, (4, 2))

        # Check center is approximately (200, 200)
        center = detections[0].center
        self.assertAlmostEqual(center[0], 200.0, delta=5.0)
        self.assertAlmostEqual(center[1], 200.0, delta=5.0)


if __name__ == "__main__":
    unittest.main()
