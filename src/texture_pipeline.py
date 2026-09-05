"""
UV Parametrization and Multi-View Texture Baking module.
Uses xatlas for chart parameterization and projects high-resolution keyframes
onto the UV atlas, exporting textured OBJ/MTL/PNG and interactive GLB for Three.js.
"""
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import trimesh

from src.config import TextureConfig


@dataclass
class TextureBakingResult:
    """Artifacts produced by the texture baking pipeline."""
    success: bool
    atlas_png_path: Path
    obj_model_path: Path
    mtl_path: Path
    glb_model_path: Optional[Path]
    atlas_resolution: int
    num_uv_vertices: int
    num_uv_faces: int
    baking_time_sec: float


class TexturePipeline:
    """
    Parametrizes a 3D triangle mesh with xatlas and bakes real multi-view photographic
    textures from calibrated camera poses onto a unified 2D texture atlas.
    """

    def __init__(self, config: TextureConfig):
        self.config = config

    @staticmethod
    def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
        """Convert COLMAP quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
        qw, qx, qy, qz = qvec
        return np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ], dtype=np.float64)

    def bake_texture(
        self,
        mesh_path: Path,
        sfm_result: Dict,
        frames_dir: Path,
        output_dir: Path,
        base_name: str = "model",
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> TextureBakingResult:
        """
        Execute UV unwrapping, camera projection scoring, texture atlas rasterization,
        seam inpainting, and multi-format export (OBJ + MTL + PNG + GLB).
        """
        start_time = time.time()
        mesh_path = Path(mesh_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = Path(frames_dir)

        print(f"\n[TexturePipeline] Starting UV parameterization and Texture Baking on '{mesh_path.name}'...")
        if progress_callback: progress_callback(5, "Cargando malla 3D para parametrización UV...")

        # 1. Load mesh with Trimesh
        mesh = trimesh.load(str(mesh_path), process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            # Try to convert if scene
            mesh = mesh.dump().sum()

        vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

        print(f"[TexturePipeline] Mesh has {len(vertices)} vertices, {len(faces)} faces")

        # 2. UV Parameterization via xatlas
        if progress_callback: progress_callback(20, "Generando desplegado UV paramétrico con xatlas...")
        print("[TexturePipeline] Computing UV atlas charts with xatlas...")

        import xatlas
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

        # Unpacked geometry with UV coordinates
        unpacked_vertices = vertices[vmapping]
        unpacked_faces = indices
        num_new_verts = len(unpacked_vertices)
        num_new_faces = len(unpacked_faces)
        print(f"[TexturePipeline] xatlas unwrapped to {num_new_verts} UV vertices, {num_new_faces} faces")

        # 3. Setup Calibrated Cameras
        if progress_callback: progress_callback(40, "Calculando proyección y visibilidad de cámaras multivista...")
        print("[TexturePipeline] Preparing camera models and poses...")

        # Cameras and images dict from sfm_result
        colmap_cameras = sfm_result.get("cameras", {})
        colmap_images = sfm_result.get("images", {})

        # Build camera projection information for each registered image
        cams_info = []
        for img_id, img_data in colmap_images.items():
            img_name = img_data["name"]
            img_file = frames_dir / img_name
            if not img_file.exists():
                continue

            qvec = img_data["qvec"]
            tvec = img_data["tvec"]
            R = self._qvec2rotmat(qvec)
            t = np.array(tvec, dtype=np.float64).reshape(3, 1)

            # Camera center in world coords: C = -R^T * t
            cam_center = (-R.T @ t).flatten()

            cam_id = img_data["camera_id"]
            cam_meta = colmap_cameras.get(cam_id, {})
            c_params = cam_meta.get("params", np.array([800.0, 320.0, 240.0, 0.0]))
            fx = fy = c_params[0]
            cx = c_params[1] if len(c_params) > 1 else 320.0
            cy = c_params[2] if len(c_params) > 2 else 240.0

            K = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            cams_info.append({
                "name": img_name,
                "file": img_file,
                "R": R,
                "t": t.flatten(),
                "center": cam_center,
                "K": K,
                "width": cam_meta.get("width", int(cx * 2)),
                "height": cam_meta.get("height", int(cy * 2))
            })

        print(f"[TexturePipeline] Total active calibrated cameras: {len(cams_info)}")

        # 4. View Selection per Face
        # Compute face normals and face centers
        v0 = unpacked_vertices[unpacked_faces[:, 0]]
        v1 = unpacked_vertices[unpacked_faces[:, 1]]
        v2 = unpacked_vertices[unpacked_faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3.0
        face_normals = np.cross(v1 - v0, v2 - v0)
        norm_lens = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
        face_normals = face_normals / norm_lens

        # Score cameras for each face
        # Best camera index for each face
        best_cam_per_face = np.full(num_new_faces, -1, dtype=np.int32)
        best_scores = np.full(num_new_faces, -1.0, dtype=np.float32)

        for c_idx, cam in enumerate(cams_info):
            # Vector from face center to camera
            vec_to_cam = cam["center"] - face_centers
            dist_to_cam = np.linalg.norm(vec_to_cam, axis=1) + 1e-6
            dir_to_cam = vec_to_cam / dist_to_cam[:, None]

            # Cosine with face normal
            cos_angles = np.sum(face_normals * dir_to_cam, axis=1)

            # Valid only if facing camera
            valid_mask = cos_angles > 0.05
            if not np.any(valid_mask):
                continue

            # Project centers to check if inside frame
            # X_c = R * X + t
            pts_cam = (cam["R"] @ face_centers[valid_mask].T + cam["t"][:, None]).T
            in_front = pts_cam[:, 2] > 0.1
            valid_indices = np.where(valid_mask)[0][in_front]

            if len(valid_indices) == 0:
                continue

            pts_proj = (cam["K"] @ pts_cam[in_front].T).T
            uv_proj = pts_proj[:, :2] / pts_proj[:, 2:3]

            in_bounds = (
                (uv_proj[:, 0] >= 5) & (uv_proj[:, 0] < cam["width"] - 5) &
                (uv_proj[:, 1] >= 5) & (uv_proj[:, 1] < cam["height"] - 5)
            )
            final_indices = valid_indices[in_bounds]
            scores = (cos_angles[final_indices] / dist_to_cam[final_indices]).astype(np.float32)

            better = scores > best_scores[final_indices]
            best_cam_per_face[final_indices[better]] = c_idx
            best_scores[final_indices[better]] = scores[better]

        # 5. Rasterize onto Texture Atlas
        res = self.config.atlas_resolution
        print(f"[TexturePipeline] Rasterizing texture atlas ({res}x{res} px)...")
        if progress_callback: progress_callback(60, f"Horneando atlas de textura fotográfica ({res}x{res} px)...")

        atlas = np.zeros((res, res, 3), dtype=np.uint8)
        painted_mask = np.zeros((res, res), dtype=np.uint8)

        # Convert UVs (0..1) to pixel coordinates on the atlas
        # Note: standard UV (0,0) is bottom-left, image (0,0) is top-left
        uv_px = np.empty_like(uvs)
        uv_px[:, 0] = np.clip(uvs[:, 0] * (res - 1), 0, res - 1)
        uv_px[:, 1] = np.clip((1.0 - uvs[:, 1]) * (res - 1), 0, res - 1)

        # Process faces grouped by camera
        for c_idx, cam in enumerate(cams_info):
            assigned_face_ids = np.where(best_cam_per_face == c_idx)[0]
            if len(assigned_face_ids) == 0:
                continue

            # Load source frame
            src_img = cv2.imread(str(cam["file"]))
            if src_img is None:
                continue

            # Project face vertices into camera
            for f_id in assigned_face_ids:
                idx_tri = unpacked_faces[f_id]
                tri_pts_3d = unpacked_vertices[idx_tri] # (3, 3)

                # Camera projection
                tri_cam = (cam["R"] @ tri_pts_3d.T + cam["t"][:, None]).T
                if np.any(tri_cam[:, 2] <= 0.05):
                    continue

                tri_2d = (cam["K"] @ tri_cam.T).T
                src_pts = (tri_2d[:, :2] / tri_2d[:, 2:3]).astype(np.float32)

                dst_pts = uv_px[idx_tri].astype(np.float32)

                # Bounding box on atlas
                min_dst = np.maximum(np.floor(np.min(dst_pts, axis=0)).astype(int), 0)
                max_dst = np.minimum(np.ceil(np.max(dst_pts, axis=0)).astype(int), res - 1)

                if (max_dst[0] <= min_dst[0]) or (max_dst[1] <= min_dst[1]):
                    continue

                # Local triangle in cropped atlas patch
                dst_local = (dst_pts - min_dst).astype(np.float32)
                patch_w = int(max_dst[0] - min_dst[0] + 1)
                patch_h = int(max_dst[1] - min_dst[1] + 1)

                # Source bounding box
                min_src = np.maximum(np.floor(np.min(src_pts, axis=0)).astype(int), 0)
                max_src = np.minimum(np.ceil(np.max(src_pts, axis=0)).astype(int), [cam["width"] - 1, cam["height"] - 1])

                if (max_src[0] <= min_src[0]) or (max_src[1] <= min_src[1]):
                    continue

                src_local = (src_pts - min_src).astype(np.float32)
                cropped_src = src_img[min_src[1]:max_src[1]+1, min_src[0]:max_src[0]+1]

                if cropped_src.size == 0:
                    continue

                # Compute affine warp from source to destination patch
                try:
                    warp_mat = cv2.getAffineTransform(src_local[:3], dst_local[:3])
                    warped_patch = cv2.warpAffine(
                        cropped_src, warp_mat, (patch_w, patch_h),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                    )

                    # Create triangle mask on dst patch
                    tri_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
                    cv2.fillConvexPoly(tri_mask, dst_local.astype(np.int32), 255)

                    # Paint on atlas
                    atlas_roi = atlas[min_dst[1]:max_dst[1]+1, min_dst[0]:max_dst[0]+1]
                    mask_roi = painted_mask[min_dst[1]:max_dst[1]+1, min_dst[0]:max_dst[0]+1]

                    paint_area = (tri_mask > 0)
                    atlas_roi[paint_area] = warped_patch[paint_area]
                    mask_roi[paint_area] = 255
                except cv2.error:
                    continue

        # 6. Seam Inpainting & Bleed Dilation
        if self.config.seam_blending:
            if progress_callback: progress_callback(85, "Difuminando costuras y dilatando bordes UV...")
            print("[TexturePipeline] Dilating and inpainting texture seams to avoid border artifacts...")
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated_mask = cv2.dilate(painted_mask, kernel, iterations=2)
            seam_border = (dilated_mask > 0) & (painted_mask == 0)

            if np.any(seam_border):
                inpaint_mask = (painted_mask == 0).astype(np.uint8)
                atlas = cv2.inpaint(atlas, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # 7. Save Texture Image
        atlas_png_path = output_dir / f"{base_name}_texture.png"
        cv2.imwrite(str(atlas_png_path), atlas, [int(cv2.IMWRITE_PNG_COMPRESSION), 4])
        print(f"[TexturePipeline] Saved texture atlas: {atlas_png_path}")

        # 8. Export Wavefront OBJ + MTL
        if progress_callback: progress_callback(92, "Exportando modelo texturizado OBJ y MTL...")
        obj_path = output_dir / f"{base_name}_textured.obj"
        mtl_path = output_dir / f"{base_name}_textured.mtl"

        with open(mtl_path, "w", encoding="utf-8") as f_mtl:
            f_mtl.write("# Reconstructor 3D Metric Model Material\n")
            f_mtl.write(f"newmtl Material_Atlas\n")
            f_mtl.write("Ka 0.200 0.200 0.200\n")
            f_mtl.write("Kd 0.900 0.900 0.900\n")
            f_mtl.write("Ks 0.100 0.100 0.100\n")
            f_mtl.write("d 1.0\n")
            f_mtl.write(f"map_Kd {atlas_png_path.name}\n")

        with open(obj_path, "w", encoding="utf-8") as f_obj:
            f_obj.write(f"# Reconstructor 3D Metric Model with Photorealistic UV Atlas\n")
            f_obj.write(f"mtllib {mtl_path.name}\n")
            f_obj.write(f"o {base_name}\n\n")

            # Vertices
            for v in unpacked_vertices:
                f_obj.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

            # UV coordinates
            for uv in uvs:
                f_obj.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            f_obj.write("\nusemtl Material_Atlas\ns 1\n")
            # Faces: 1-based indexing for vertices and UVs (v/vt)
            for f_idx in unpacked_faces:
                f_obj.write(f"f {f_idx[0]+1}/{f_idx[0]+1} {f_idx[1]+1}/{f_idx[1]+1} {f_idx[2]+1}/{f_idx[2]+1}\n")

        print(f"[TexturePipeline] Saved textured OBJ: {obj_path}")

        # 9. Export GLB (glTF binary) with embedded texture for Three.js
        glb_path = None
        if self.config.export_glb:
            if progress_callback: progress_callback(96, "Empaquetando modelo interactivo GLB para Three.js...")
            glb_path = output_dir / f"{base_name}_textured.glb"
            try:
                rgb_atlas = cv2.cvtColor(atlas, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_atlas)
                
                visual = trimesh.visual.TextureVisuals(
                    uv=uvs,
                    image=pil_img
                )
                textured_mesh = trimesh.Trimesh(
                    vertices=unpacked_vertices,
                    faces=unpacked_faces,
                    visual=visual,
                    process=False
                )
                textured_mesh.export(str(glb_path), file_type="glb")
                print(f"[TexturePipeline] Saved interactive GLB: {glb_path}")
            except Exception as e:
                print(f"[TexturePipeline] Warning: GLB export failed: {e}")
                glb_path = None

        total_time = time.time() - start_time
        print(f"[TexturePipeline] Texture Baking completed in {total_time:.1f} s [OK]")
        if progress_callback: progress_callback(100, f"Texturizado completado [OK] ({res}x{res} px)")

        return TextureBakingResult(
            success=True,
            atlas_png_path=atlas_png_path,
            obj_model_path=obj_path,
            mtl_path=mtl_path,
            glb_model_path=glb_path,
            atlas_resolution=res,
            num_uv_vertices=num_new_verts,
            num_uv_faces=num_new_faces,
            baking_time_sec=total_time
        )
