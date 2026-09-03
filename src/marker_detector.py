"""
Marker detection module supporting AprilTag (tag36h11) and ArUco dictionaries.
Extracts 2D corners, IDs, and centers from images for metric calibration.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from pupil_apriltags import Detector as AprilTagDetector
    PUPIL_APRILTAGS_AVAILABLE = True
except ImportError:
    PUPIL_APRILTAGS_AVAILABLE = False


@dataclass
class MarkerDetection:
    """Represents a detected fiducial marker in an image."""
    marker_id: int
    family: str
    corners: np.ndarray  # Shape (4, 2) in [top-left, top-right, bottom-right, bottom-left] order
    center: np.ndarray   # Shape (2,) [x, y]
    margin: float = 0.0  # Decision margin / confidence


class MarkerDetector:
    """Detects AprilTag and ArUco markers with sub-pixel precision."""

    def __init__(self, family: str = "tag36h11", search_aruco_dict: str = "DICT_APRILTAG_36h11"):
        self.family = family
        self.search_aruco_dict = search_aruco_dict
        self._init_detector()

    def _init_detector(self) -> None:
        self.pupil_detector = None
        if PUPIL_APRILTAGS_AVAILABLE:
            try:
                self.pupil_detector = AprilTagDetector(
                    families=self.family,
                    nthreads=2,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0
                )
            except Exception as e:
                print(f"[MarkerDetector] pupil_apriltags init warning: {e}. Falling back to cv2.aruco.")

        # OpenCV ArUco detector setup (fallback or standard)
        try:
            dict_attr = getattr(cv2.aruco, self.search_aruco_dict, cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
            self.aruco_params = cv2.aruco.DetectorParameters()
            # Refine corner detection for sub-pixel accuracy
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except Exception as e:
            print(f"[MarkerDetector] cv2.aruco init warning: {e}")
            self.aruco_detector = None

    def detect(self, image_bgr: np.ndarray) -> List[MarkerDetection]:
        """
        Detect fiducial markers in a BGR image.
        Returns a list of MarkerDetection instances.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        detections: List[MarkerDetection] = []

        # 1. Try pupil_apriltags first
        if self.pupil_detector is not None:
            try:
                results = self.pupil_detector.detect(gray)
                for res in results:
                    corners = np.array(res.corners, dtype=np.float32)
                    center = np.array(res.center, dtype=np.float32)
                    detections.append(MarkerDetection(
                        marker_id=int(res.tag_id),
                        family=self.family,
                        corners=corners,
                        center=center,
                        margin=float(getattr(res, 'decision_margin', 0.0))
                    ))
                if detections:
                    return detections
            except Exception as e:
                pass  # Fall through to aruco

        # 2. Fallback to cv2.aruco
        if self.aruco_detector is not None:
            try:
                corners_list, ids, _ = self.aruco_detector.detectMarkers(gray)
                if ids is not None and len(ids) > 0:
                    for i, cid in enumerate(ids.flatten()):
                        c = corners_list[i][0]  # shape (4, 2)
                        center = np.mean(c, axis=0)
                        detections.append(MarkerDetection(
                            marker_id=int(cid),
                            family=self.search_aruco_dict,
                            corners=c.astype(np.float32),
                            center=center.astype(np.float32),
                            margin=1.0
                        ))
            except Exception as e:
                print(f"[MarkerDetector] ArUco detection error: {e}")

        return detections

    def draw_detections(self, image_bgr: np.ndarray, detections: List[MarkerDetection]) -> np.ndarray:
        """Draw bounding boxes and IDs on the image for visual verification."""
        canvas = image_bgr.copy()
        for det in detections:
            pts = det.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Draw corner 0 with distinct red dot (Top-Left)
            c0 = tuple(pts[0][0])
            cv2.circle(canvas, c0, 5, (0, 0, 255), -1)
            
            # Text label
            center = tuple(det.center.astype(np.int32))
            label = f"ID: {det.marker_id}"
            cv2.putText(canvas, label, (center[0] - 20, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return canvas
