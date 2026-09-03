"""
Mesh Generation and Surface Solidification Module.
Uses Open3D Poisson Reconstruction to generate clean, watertight STL and OBJ 3D models.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import open3d as o3d
import trimesh

from src.config import MeshConfig


@dataclass
class MeshGenerationResult:
    """Contains paths and statistics for the reconstructed 3D mesh."""
    stl_path: Path
    obj_path: Path
    ply_path: Path
    num_vertices: int
    num_triangles: int
    bounding_box_min_mm: np.ndarray
    bounding_box_max_mm: np.ndarray
    dimensions_mm: np.ndarray  # [dx, dy, dz] in mm
    is_watertight: bool
    volume_cm3: float


class MeshGenerator:
    """Reconstructs, filters, and exports watertight 3D meshes from metric point clouds."""

    def __init__(self, config: MeshConfig):
        self.config = config

    def generate_mesh(
        self,
        points_xyz_metric: np.ndarray,
        colors_rgb: Optional[np.ndarray],
        output_dir: Path,
        base_name: str = "reconstructed_model"
    ) -> MeshGenerationResult:
        """
        Processes point cloud and creates clean 3D solid mesh.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if points_xyz_metric.shape[0] < 100:
            raise ValueError(f"Insufficient 3D points ({points_xyz_metric.shape[0]}) to generate mesh.")

        print(f"[MeshGenerator] Processing {points_xyz_metric.shape[0]} metric points...")

        # 1. Build Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz_metric)
        if colors_rgb is not None and colors_rgb.shape[0] == points_xyz_metric.shape[0]:
            # Normalize to 0..1 if uint8
            if colors_rgb.max() > 1.0:
                colors_rgb = colors_rgb / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        # 2. Region of Interest (ROI) Crop (Isolate object from floor/table/background)
        if self.config.crop_to_roi:
            pts = np.asarray(pcd.points)
            r_sq = pts[:, 0]**2 + pts[:, 1]**2
            # Keep points within cylinder around board center and above the sheet surface (Z >= 2.0 mm)
            valid_mask = (r_sq <= self.config.roi_radius_mm**2) & \
                         (pts[:, 2] >= 2.0) & \
                         (pts[:, 2] <= self.config.roi_height_mm)
            if np.sum(valid_mask) > 50:
                pcd = pcd.select_by_index(np.where(valid_mask)[0])
                print(f"[MeshGenerator] ROI crop: retained {len(pcd.points)} points inside radius {self.config.roi_radius_mm} mm and above Z=2.0 mm.")

        # 3. Statistical Outlier Removal
        if self.config.filter_statistical_outliers and len(pcd.points) > 50:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.config.nb_neighbors,
                std_ratio=self.config.std_ratio
            )
            print(f"[MeshGenerator] Outlier filter: {len(pcd.points)} points remaining.")

        # 4. Scale-Invariant Normal Estimation & Orientation
        # Use KNN search so normal estimation is robust regardless of physical scale
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=min(25, max(8, len(pcd.points) - 1)))
        )
        pcd.orient_normals_consistent_tangent_plane(k=min(15, max(5, len(pcd.points) - 1)))

        # 5. Poisson Surface Reconstruction
        # For sparse point clouds (<1500 pts), depth 7-8 avoids over-fitting/ballooning
        actual_depth = min(self.config.poisson_depth, 8) if len(pcd.points) < 1500 else self.config.poisson_depth
        print(f"[MeshGenerator] Running Poisson Surface Reconstruction (depth={actual_depth}, points={len(pcd.points)})...")
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=actual_depth, linear_fit=True
        )

        # 6. Density Trimming (remove loose enclosing bubble / boundary artifacts)
        densities = np.asarray(densities)
        if len(densities) > 0:
            trim_pct = max(self.config.density_trim_percentile, 10.0 if len(pcd.points) < 1500 else 5.0)
            density_threshold = np.percentile(densities, trim_pct)
            vertices_to_remove = densities < density_threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

        # 7. Post-processing and smoothing
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_non_manifold_edges()

        if self.config.smooth_iterations > 0 and len(mesh_o3d.vertices) > 0:
            mesh_o3d = mesh_o3d.filter_smooth_taubin(number_of_iterations=self.config.smooth_iterations)

        # Filter any NaN vertices produced by Taubin or degenerate geometry
        verts = np.asarray(mesh_o3d.vertices)
        if len(verts) > 0 and np.isnan(verts).any():
            nan_mask = np.isnan(verts).any(axis=1)
            mesh_o3d.remove_vertices_by_mask(nan_mask)

        mesh_o3d.compute_vertex_normals()

        # 8. Export via Trimesh for watertight verification
        stl_path = output_dir / f"{base_name}.stl"
        obj_path = output_dir / f"{base_name}.obj"
        ply_path = output_dir / f"{base_name}.ply"

        o3d.io.write_triangle_mesh(str(obj_path), mesh_o3d)
        o3d.io.write_triangle_mesh(str(ply_path), mesh_o3d)

        # Load into trimesh to inspect watertightness and volume
        t_mesh = trimesh.load(str(obj_path), force='mesh')
        
        # Save as binary STL
        t_mesh.export(str(stl_path))

        # Calculate bounding box and dimensions
        bbox_min = np.array(t_mesh.bounds[0])
        bbox_max = np.array(t_mesh.bounds[1])
        dims = bbox_max - bbox_min
        volume_cm3 = (t_mesh.volume / 1000.0) if t_mesh.is_watertight and t_mesh.volume > 0 else 0.0

        print(f"[MeshGenerator] Mesh saved:")
        print(f"  -> STL: {stl_path.name}")
        print(f"  -> Vertices: {len(t_mesh.vertices)}, Faces: {len(t_mesh.faces)}")
        print(f"  -> Dimensions (X, Y, Z): {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")
        print(f"  -> Watertight: {t_mesh.is_watertight} | Volume: {volume_cm3:.1f} cm³")

        return MeshGenerationResult(
            stl_path=stl_path,
            obj_path=obj_path,
            ply_path=ply_path,
            num_vertices=len(t_mesh.vertices),
            num_triangles=len(t_mesh.faces),
            bounding_box_min_mm=bbox_min,
            bounding_box_max_mm=bbox_max,
            dimensions_mm=dims,
            is_watertight=bool(t_mesh.is_watertight),
            volume_cm3=float(volume_cm3)
        )
