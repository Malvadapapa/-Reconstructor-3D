"""
End-to-End Integration Test with Synthetic 3D Video.
Generates an orbiting video of a textured cylinder over an AprilTag target,
runs the complete VideoTo3DPipeline, and verifies STL and measurement outputs.
"""
import math
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.config import PipelineConfig, MarkerConfig, VideoIngestConfig, SfMConfig, MeshConfig, SliceConfig
from src.pipeline import VideoTo3DPipeline
from src.marker_generator import generate_apriltag_matrix_36h11


class TestEndToEndSynthetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.work_dir = Path("output/test_e2e")
        cls.work_dir.mkdir(parents=True, exist_ok=True)
        cls.video_path = cls.work_dir / "synthetic_bottle_orbit.mp4"

        # 1. Generate synthetic orbiting video
        cls._create_synthetic_video(cls.video_path, num_frames=36)

    @classmethod
    def _create_synthetic_video(cls, output_video_path: Path, num_frames: int = 36):
        """Create an orbiting video of a 3D textured cylinder on top of an AprilTag base."""
        w, h = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, 10.0, (w, h))

        tag_bits = generate_apriltag_matrix_36h11(0)
        tag_img = cv2.resize(tag_bits, (120, 120), interpolation=cv2.INTER_NEAREST)

        # 3D points of a cylinder with checkerboard texture
        r_cyl = 30.0  # mm
        h_cyl = 100.0 # mm
        
        # Camera intrinsics
        fx = fy = 500.0
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # 2. Draw rich textured background and cylinder body with high-frequency SIFT features
        np.random.seed(42)
        base_noise = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        for i in range(num_frames):
            angle = (2.0 * math.pi * i) / num_frames
            cam_dist = 220.0
            cam_height = 130.0
            
            cam_x = cam_dist * math.cos(angle)
            cam_y = cam_dist * math.sin(angle)
            cam_z = cam_height
            cam_pos = np.array([cam_x, cam_y, cam_z])
            target_pos = np.array([0.0, 0.0, 30.0])

            # Look-at rotation
            forward = (target_pos - cam_pos)
            forward = forward / np.linalg.norm(forward)
            up_world = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up_world)
            right = right / (np.linalg.norm(right) + 1e-8)
            up_cam = np.cross(right, forward)

            # Camera coordinates: X right, Y down, Z forward
            R_world_to_cam = np.vstack([right, -up_cam, forward])
            t_world_to_cam = -R_world_to_cam @ cam_pos

            # Canvas
            frame = np.full((h, w, 3), 235, dtype=np.uint8)

            # Draw textured ground board (Z=0, size=160x160mm)
            board_size = 140.0
            board_corners_3d = np.array([
                [-board_size/2, -board_size/2, 0],
                [ board_size/2, -board_size/2, 0],
                [ board_size/2,  board_size/2, 0],
                [-board_size/2,  board_size/2, 0]
            ])
            b_cam = (R_world_to_cam @ board_corners_3d.T + t_world_to_cam.reshape(3, 1)).T
            b_2d = (K @ b_cam.T).T
            b_2d = b_2d[:, :2] / b_2d[:, 2:3]

            H_b, _ = cv2.findHomography(
                np.array([[0,0], [200,0], [200,200], [0,200]], dtype=np.float32),
                b_2d.astype(np.float32)
            )
            if H_b is not None:
                warped_b = cv2.warpPerspective(base_noise, H_b, (w, h))
                mask_b = (warped_b > 0).astype(np.uint8)
                frame = np.where(mask_b > 0, warped_b, frame)

            # Overlay AprilTag quad on center
            tag_size = 50.0
            tag_corners_3d = np.array([
                [-tag_size/2, -tag_size/2, 0],
                [ tag_size/2, -tag_size/2, 0],
                [ tag_size/2,  tag_size/2, 0],
                [-tag_size/2,  tag_size/2, 0]
            ])
            tag_cam = (R_world_to_cam @ tag_corners_3d.T + t_world_to_cam.reshape(3, 1)).T
            tag_2d = (K @ tag_cam.T).T
            tag_2d = tag_2d[:, :2] / tag_2d[:, 2:3]

            H, _ = cv2.findHomography(
                np.array([[0,0], [120,0], [120,120], [0,120]], dtype=np.float32),
                tag_2d.astype(np.float32)
            )
            if H is not None:
                warped_tag = cv2.warpPerspective(tag_img, H, (w, h))
                mask = (warped_tag > 0).astype(np.uint8)
                if len(warped_tag.shape) == 2:
                    warped_tag = cv2.cvtColor(warped_tag, cv2.COLOR_GRAY2BGR)
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                frame = np.where(mask > 0, warped_tag, frame)

            # Project 3D textured cylinder points
            for z_val in np.linspace(5, h_cyl, 20):
                for theta_idx in range(36):
                    th = (2.0 * math.pi * theta_idx) / 36.0
                    px = r_cyl * math.cos(th)
                    py = r_cyl * math.sin(th)
                    pz = z_val
                    pt3d = np.array([px, py, pz])
                    
                    pt_cam = R_world_to_cam @ pt3d + t_world_to_cam
                    if pt_cam[2] > 1.0:
                        pt2d = K @ pt_cam
                        u, v = int(pt2d[0] / pt2d[2]), int(pt2d[1] / pt2d[2])
                        if 0 <= u < w and 0 <= v < h:
                            col = (int((z_val * 2) % 255), int((theta_idx * 7) % 255), int((z_val * 3 + theta_idx * 5) % 255))
                            cv2.circle(frame, (u, v), 4, col, -1)

            out.write(frame)

        out.release()
        print(f"[TestE2E] Generated synthetic video: {output_video_path}")

    def test_pipeline_execution(self):
        output_scan_dir = self.work_dir / "scan_results"
        config = PipelineConfig(
            video_path=self.video_path,
            output_dir=output_scan_dir,
            marker=MarkerConfig(marker_id=0, marker_size_mm=50.0),
            ingest=VideoIngestConfig(target_fps=10.0, min_laplacian_var=10.0),
            sfm=SfMConfig(matcher_type="exhaustive", use_gpu=False, max_features=4096),
            mesh=MeshConfig(poisson_depth=8, filter_statistical_outliers=False),
            slice=SliceConfig(step_height_mm=10.0, min_height_mm=5.0)
        )

        pipeline = VideoTo3DPipeline(config)
        result = pipeline.run()

        self.assertTrue(result.success)
        self.assertTrue(result.stl_model_path.exists())
        self.assertTrue(result.measurements_json_path.exists())
        self.assertGreater(result.num_registered_cameras, 5)
        self.assertGreater(result.num_sparse_points, 50)
        print(f"\n[TestE2E] Reconstructed STL: {result.stl_model_path}")
        print(f"[TestE2E] Processing Time: {result.total_time_sec:.2f}s")


if __name__ == "__main__":
    unittest.main()
