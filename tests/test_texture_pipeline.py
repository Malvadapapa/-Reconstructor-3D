"""
Unit test for TexturePipeline (xatlas UV parameterization and multi-view texture baking).
"""
import unittest
from pathlib import Path
import cv2
import numpy as np
import trimesh

from src.config import TextureConfig
from src.texture_pipeline import TexturePipeline


class TestTexturePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("output/test_texture_scratch")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.frames_dir = cls.test_dir / "frames"
        cls.frames_dir.mkdir(exist_ok=True)

        # 1. Create a synthetic textured image frame
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Checkerboard pattern
        for y in range(0, 480, 40):
            for x in range(0, 640, 40):
                if (x // 40 + y // 40) % 2 == 0:
                    img[y:y+40, x:x+40] = (200, 100, 50)
                else:
                    img[y:y+40, x:x+40] = (50, 180, 220)

        cls.frame_name = "frame_0000.jpg"
        cv2.imwrite(str(cls.frames_dir / cls.frame_name), img)

        # 2. Create a simple 3D mesh (planar quad facing Z)
        # 2 triangles in XY plane at Z=0
        vertices = np.array([
            [-50.0, -50.0, 0.0],
            [ 50.0, -50.0, 0.0],
            [ 50.0,  50.0, 0.0],
            [-50.0,  50.0, 0.0]
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.uint32)

        cls.mesh_path = cls.test_dir / "quad_mesh.stl"
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(str(cls.mesh_path))

        # 3. Create camera looking at (0, 0, 0) from (0, 0, 200)
        # In COLMAP coordinates:
        # X right, Y down, Z forward
        # If camera is at (0, 0, 200) looking along -Z:
        # Camera coords: X_c = X_w, Y_c = -Y_w, Z_c = -Z_w + 200
        # R = [[1, 0, 0], [0, -1, 0], [0, 0, -1]] -> qvec
        # t = [0, 0, 200]
        # For simplicity: identity rotation looking along +Z:
        # Camera at (0, 0, -200) looking along +Z at (0, 0, 0):
        # R = I, t = [0, 0, 200]
        # Camera at (0, 0, 200) looking along -Z at (0, 0, 0)
        # 180 deg rotation around X axis: R = diag(1, -1, -1), t = [0, 0, 200]
        # qvec for 180 deg around X: [0.0, 1.0, 0.0, 0.0]
        cls.sfm_result = {
            "cameras": {
                1: {
                    "camera_id": 1,
                    "model_id": 2,
                    "width": 640,
                    "height": 480,
                    "params": np.array([500.0, 320.0, 240.0, 0.0])
                }
            },
            "images": {
                1: {
                    "image_id": 1,
                    "camera_id": 1,
                    "name": cls.frame_name,
                    "qvec": [0.0, 1.0, 0.0, 0.0],
                    "tvec": [0.0, 0.0, 200.0]
                }
            }
        }

    def test_texture_baking_pipeline(self):
        config = TextureConfig(
            enabled=True,
            atlas_resolution=512, # 512px for fast test
            seam_blending=True,
            export_glb=True
        )
        pipeline = TexturePipeline(config)

        output_dir = self.test_dir / "texture_out"
        result = pipeline.bake_texture(
            mesh_path=self.mesh_path,
            sfm_result=self.sfm_result,
            frames_dir=self.frames_dir,
            output_dir=output_dir,
            base_name="quad_test"
        )

        self.assertTrue(result.success)
        self.assertTrue(result.atlas_png_path.exists())
        self.assertTrue(result.obj_model_path.exists())
        self.assertTrue(result.mtl_path.exists())
        self.assertIsNotNone(result.glb_model_path)
        self.assertTrue(result.glb_model_path.exists())

        # Check atlas image size
        atlas_img = cv2.imread(str(result.atlas_png_path))
        self.assertEqual(atlas_img.shape[:2], (512, 512))

        # Check that atlas is not blank
        self.assertGreater(np.count_nonzero(atlas_img), 100)
        print(f"\n[TestTexturePipeline] Atlas non-zero px: {np.count_nonzero(atlas_img)} / {512*512}")
        print(f"[TestTexturePipeline] OBJ: {result.obj_model_path}")
        print(f"[TestTexturePipeline] GLB: {result.glb_model_path}")


if __name__ == "__main__":
    unittest.main()
