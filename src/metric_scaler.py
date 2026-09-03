"""
Metric Scaling and Coordinate Alignment Module.
Triangulates fiducial marker corners in 3D across camera poses, calculates scale factor in millimeters (mm),
and aligns the coordinate system so the marker plane is at Z=0 with normal pointing +Z.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import MarkerConfig
from src.sfm_reconstruction import ColmapModelParser


@dataclass
class MetricCalibrationResult:
    """Contains scale factor, transformation matrix, and calibration statistics."""
    scale_factor: float          # Multiplier to convert raw SfM units to millimeters (mm)
    marker_size_mm: float        # Known physical marker side length (mm)
    measured_size_sfm: float     # Reconstructed side length in raw SfM units
    scale_error_mm: float        # Discrepancy between orthogonal sides in mm
    corners_3d_metric: np.ndarray# 4x3 array of metric marker corner coordinates
    center_3d_metric: np.ndarray # 3x1 center position of marker
    transform_matrix: np.ndarray # 4x4 transformation matrix [R*S | t]


class MetricScaler:
    """Calculates 1:1 metric scale and aligns 3D coordinates to the physical marker plane."""

    def __init__(self, marker_config: MarkerConfig):
        self.config = marker_config

    @staticmethod
    def _get_camera_matrix(camera_dict: Dict) -> np.ndarray:
        """Construct 3x3 intrinsic matrix K from COLMAP camera parameters."""
        params = camera_dict["params"]
        model_id = camera_dict["model_id"]
        w = camera_dict["width"]
        h = camera_dict["height"]

        # Models: 0: SIMPLE_PINHOLE, 1: PINHOLE, 2: SIMPLE_RADIAL, 3: RADIAL
        if model_id in [0, 2]:  # [f, cx, cy, ...]
            f, cx, cy = params[0], params[1], params[2]
            fx, fy = f, f
        elif model_id in [1, 3]:  # [fx, fy, cx, cy, ...]
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            fx = fy = params[0]
            cx, cy = w / 2.0, h / 2.0

        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def _triangulate_point_multiview(projection_matrices: List[np.ndarray], points_2d: List[np.ndarray]) -> np.ndarray:
        """
        Triangulate a 3D point from 2D observations in multiple cameras using SVD (DLT algorithm).
        """
        A = []
        for P, pt in zip(projection_matrices, points_2d):
            x, y = pt[0], pt[1]
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        A = np.array(A)

        _, _, vh = np.linalg.svd(A)
        X_hom = vh[-1]
        if abs(X_hom[3]) < 1e-8:
            return X_hom[:3]
        return (X_hom[:3] / X_hom[3]).astype(np.float64)

    def calibrate_and_align(
        self,
        sfm_result: Dict,
        frames_manifest: Dict
    ) -> MetricCalibrationResult:
        """
        Triangulates AprilTag corners across registered SfM cameras, calculates scale factor,
        and computes the 4x4 alignment matrix.
        """
        cameras = sfm_result["cameras"]
        images = sfm_result["images"]
        frames = frames_manifest.get("frames", [])

        # Count detections per marker ID across registered images
        tag_counts: Dict[int, int] = {}
        for f in frames:
            name = f["file_name"]
            # Only count frames that were actually registered in SfM
            if not any(img_data["name"] == name for img_data in images.values()):
                continue
            for d in f.get("detections", []):
                mid = d["marker_id"]
                tag_counts[mid] = tag_counts.get(mid, 0) + 1

        # Pick the marker ID with the most registered camera detections
        best_tag_id = self.config.marker_id
        if tag_counts:
            best_tag_id = max(tag_counts, key=tag_counts.get)
            print(f"[MetricScaler] Detecciones por Tag: {tag_counts} -> Seleccionado Tag ID {best_tag_id}")

        # Peripheral tags (1, 2, 3, 4) are 25mm on the target sheet, central is 30mm (or configured)
        effective_marker_size_mm = 25.0 if best_tag_id in [1, 2, 3, 4] else self.config.marker_size_mm

        # Map image filename to frame detection dict for the chosen tag
        detections_by_name: Dict[str, List] = {}
        for f in frames:
            name = f["file_name"]
            for d in f.get("detections", []):
                if d["marker_id"] == best_tag_id:
                    detections_by_name[name] = d["corners_2d"]
                    break

        # Collect camera projection matrices and 2D corner observations
        # Corner index 0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left
        corner_proj_mats: List[List[np.ndarray]] = [[], [], [], []]
        corner_pts_2d: List[List[np.ndarray]] = [[], [], [], []]

        for img_id, img_data in images.items():
            img_name = img_data["name"]
            if img_name not in detections_by_name:
                continue

            cam_id = img_data["camera_id"]
            if cam_id not in cameras:
                continue

            K = self._get_camera_matrix(cameras[cam_id])
            R = ColmapModelParser.qvec2rotmat(img_data["qvec"])
            t = img_data["tvec"].reshape(3, 1)

            # Projection Matrix P = K * [R | t]
            extrinsic = np.hstack([R, t])
            P = K @ extrinsic

            corners_2d = np.array(detections_by_name[img_name], dtype=np.float64)
            for c_idx in range(4):
                corner_proj_mats[c_idx].append(P)
                corner_pts_2d[c_idx].append(corners_2d[c_idx])

        # Verify we have sufficient observations
        num_views = len(corner_proj_mats[0])
        print(f"[MetricScaler] Tag ID {best_tag_id} triangulado en {num_views} vistas de cámara.")

        if num_views < 2:
            print("[MetricScaler] Warning: Marcador AprilTag no detectado en al menos 2 vistas.")
            print("[MetricScaler] Centrando nube de puntos en el origen (0, 0) y alineando base Z=0...")
            xyz = sfm_result.get("xyz", np.empty((0, 3)))
            if len(xyz) > 0:
                center = np.median(xyz, axis=0)
                min_z = np.percentile(xyz[:, 2], 2)
                t = np.array([-center[0], -center[1], -min_z])
            else:
                t = np.zeros(3)

            default_scale = 1.0
            transform_mat = np.eye(4, dtype=np.float64)
            transform_mat[:3, 3] = t
            return MetricCalibrationResult(
                scale_factor=default_scale,
                marker_size_mm=self.config.marker_size_mm,
                measured_size_sfm=1.0,
                scale_error_mm=0.0,
                corners_3d_metric=np.zeros((4, 3)),
                center_3d_metric=np.zeros(3),
                transform_matrix=transform_mat
            )

        # Triangulate 4 corners in raw SfM space
        raw_corners_3d = []
        for c_idx in range(4):
            pt_3d = self._triangulate_point_multiview(corner_proj_mats[c_idx], corner_pts_2d[c_idx])
            raw_corners_3d.append(pt_3d)
        raw_corners = np.array(raw_corners_3d)  # Shape (4, 3)

        # Side lengths in raw SfM units
        s01 = np.linalg.norm(raw_corners[1] - raw_corners[0])
        s12 = np.linalg.norm(raw_corners[2] - raw_corners[1])
        s23 = np.linalg.norm(raw_corners[3] - raw_corners[2])
        s30 = np.linalg.norm(raw_corners[0] - raw_corners[3])

        avg_side_sfm = float((s01 + s12 + s23 + s30) / 4.0)
        scale_factor = float(effective_marker_size_mm / avg_side_sfm) if avg_side_sfm > 1e-6 else 1.0

        # Discrepancy between sides (aspect ratio error)
        max_diff_sfm = max(abs(s01 - s23), abs(s12 - s30))
        scale_error_mm = float(max_diff_sfm * scale_factor)

        # Center in raw space
        raw_center = np.mean(raw_corners, axis=0)

        # Coordinate axes based on marker:
        # X-axis along top edge: P0 -> P1
        vx = (raw_corners[1] - raw_corners[0])
        vx = vx / (np.linalg.norm(vx) + 1e-12)

        # Y-temp along left edge: P0 -> P3
        vy_temp = (raw_corners[3] - raw_corners[0])
        vy_temp = vy_temp / (np.linalg.norm(vy_temp) + 1e-12)

        # Normal vector (Z-axis pointing up from marker board)
        vz = np.cross(vx, vy_temp)
        vz = vz / (np.linalg.norm(vz) + 1e-12)

        # Ensure Z points upwards towards the camera centroid
        if images:
            cam_centers_raw = []
            for img in images.values():
                R_cam = ColmapModelParser.qvec2rotmat(img["qvec"])
                C_cam = -R_cam.T @ img["tvec"]
                cam_centers_raw.append(C_cam)
            mean_cam = np.mean(cam_centers_raw, axis=0)
            if np.dot(vz, mean_cam - raw_center) < 0:
                vz = -vz

        # Orthogonal Y-axis
        vy = np.cross(vz, vx)
        vy = vy / (np.linalg.norm(vy) + 1e-12)

        # 3x3 Rotation matrix to align [vx, vy, vz] to standard axes [X, Y, Z]
        R_align = np.vstack([vx, vy, vz])  # Shape (3, 3)

        # Known tag offset on the calibration board (mm) from center (0, 0, 0):
        # Tag 0: Center (0, 0), Tag 1: Norte (0, +72), Tag 2: Este (+72, 0), Tag 3: Sur (0, -72), Tag 4: Oeste (-72, 0)
        TAG_OFFSETS_MM = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([0.0, 72.0, 0.0]),
            2: np.array([72.0, 0.0, 0.0]),
            3: np.array([0.0, -72.0, 0.0]),
            4: np.array([-72.0, 0.0, 0.0])
        }
        tag_offset_mm = TAG_OFFSETS_MM.get(best_tag_id, np.array([0.0, 0.0, 0.0]))

        # Build 4x4 combined transformation matrix: S * R * (X - raw_center) - tag_offset
        T_matrix = np.eye(4, dtype=np.float64)
        T_matrix[:3, :3] = scale_factor * R_align
        T_matrix[:3, 3] = -scale_factor * (R_align @ raw_center) - tag_offset_mm

        # Compute metric corners
        metric_corners = (T_matrix[:3, :3] @ raw_corners.T + T_matrix[:3, 3:4]).T
        metric_center = np.mean(metric_corners, axis=0)

        print(f"[MetricScaler] Tag ID {best_tag_id} alineado al centro del tablero (offset: {tag_offset_mm[:2]} mm).")
        print(f"[MetricScaler] Scale factor: {scale_factor:.4f} mm/unit | Mean Side: {avg_side_sfm:.4f} sfm -> {avg_side_sfm * scale_factor:.2f} mm | Side Error: ±{scale_error_mm:.2f} mm")

        return MetricCalibrationResult(
            scale_factor=scale_factor,
            marker_size_mm=self.config.marker_size_mm,
            measured_size_sfm=avg_side_sfm,
            scale_error_mm=scale_error_mm,
            corners_3d_metric=metric_corners,
            center_3d_metric=metric_center,
            transform_matrix=T_matrix
        )

    @staticmethod
    def apply_transform_to_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Apply 4x4 transformation matrix to an Nx3 array of points."""
        if points.shape[0] == 0:
            return points
        hom_points = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
        transformed = (transform_matrix @ hom_points.T).T
        return transformed[:, :3]
