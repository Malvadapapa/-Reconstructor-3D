"""
Geometric Slicing and Cross-Sectional Analysis Module.
Extracts perimeters, diameters, cross-sectional areas, and heights along the Z-axis for physical validation.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from src.config import SliceConfig


@dataclass
class SectionMeasurement:
    """Measurement values for a single cross-section slice."""
    height_z_mm: float
    perimeter_mm: float
    area_mm2: float
    equivalent_diameter_mm: float  # Diameter of equivalent circle: 2 * sqrt(Area / pi)
    bounding_diameter_x_mm: float
    bounding_diameter_y_mm: float
    is_closed: bool


@dataclass
class MeshAnalysisReport:
    """Full geometric report containing dimensions and slices."""
    total_height_mm: float
    max_diameter_mm: float
    min_diameter_mm: float
    mean_diameter_mm: float
    total_volume_cm3: float
    num_slices: int
    slices: List[SectionMeasurement]

    def to_dict(self) -> Dict:
        return asdict(self)


class MeshAnalyzer:
    """Performs geometric slicing and calculates cross-sectional metrics on 3D meshes."""

    def __init__(self, config: SliceConfig):
        self.config = config

    def analyze_mesh(self, mesh_path: Path, output_json_path: Optional[Path] = None) -> MeshAnalysisReport:
        """
        Slices the mesh along the Z-axis at regular intervals and computes geometric metrics.
        """
        mesh_path = Path(mesh_path)
        mesh = trimesh.load(str(mesh_path), force='mesh')

        bounds = mesh.bounds
        z_min, z_max = float(bounds[0][2]), float(bounds[1][2])
        total_height = max(0.0, z_max - z_min)

        start_z = max(z_min + self.config.min_height_mm, z_min + 2.0)
        end_z = self.config.max_height_mm if self.config.max_height_mm else (z_max - 2.0)

        if end_z <= start_z:
            end_z = z_max

        slice_heights = np.arange(start_z, end_z, self.config.step_height_mm)
        if len(slice_heights) == 0:
            slice_heights = np.array([(start_z + end_z) / 2.0])

        print(f"[MeshAnalyzer] Slicing '{mesh_path.name}' along Z: {start_z:.1f} mm -> {end_z:.1f} mm ({len(slice_heights)} slices)...")

        slices_data: List[SectionMeasurement] = []
        all_diams: List[float] = []

        # Limit number of slices to at most 50 for efficiency
        if len(slice_heights) > 50:
            step_idx = max(1, len(slice_heights) // 40)
            slice_heights = slice_heights[::step_idx]

        for h in slice_heights:
            # Cross section 2D path
            plane_origin = [0.0, 0.0, float(h)]
            plane_normal = [0.0, 0.0, 1.0]

            slice_2d = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

            if slice_2d is None:
                continue

            # Convert 3D planar path to 2D
            path_2d, _ = slice_2d.to_2D()
            
            # Polygons inside slice (use polygons_closed which does not require rtree)
            try:
                polys = list(path_2d.polygons_closed)
            except Exception:
                polys = []

            if len(polys) == 0:
                continue

            # Largest polygon (primary outer contour)
            main_poly = max(polys, key=lambda p: p.area if hasattr(p, 'area') else 0)
            if not hasattr(main_poly, 'area') or main_poly.area <= 0:
                continue
            
            area = float(main_poly.area)
            perimeter = float(main_poly.length)
            eq_diam = float(2.0 * np.sqrt(area / np.pi)) if area > 0 else 0.0

            # Bounding box in 2D
            minx, miny, maxx, maxy = main_poly.bounds
            dx = float(maxx - minx)
            dy = float(maxy - miny)

            all_diams.append(eq_diam)

            measurement = SectionMeasurement(
                height_z_mm=round(float(h), 2),
                perimeter_mm=round(perimeter, 2),
                area_mm2=round(area, 2),
                equivalent_diameter_mm=round(eq_diam, 2),
                bounding_diameter_x_mm=round(dx, 2),
                bounding_diameter_y_mm=round(dy, 2),
                is_closed=bool(main_poly.is_valid and not main_poly.is_empty)
            )
            slices_data.append(measurement)

        volume_cm3 = float(mesh.volume / 1000.0) if mesh.is_watertight and mesh.volume > 0 else 0.0
        max_diam = float(np.max(all_diams)) if all_diams else 0.0
        min_diam = float(np.min(all_diams)) if all_diams else 0.0
        mean_diam = float(np.mean(all_diams)) if all_diams else 0.0

        report = MeshAnalysisReport(
            total_height_mm=round(total_height, 2),
            max_diameter_mm=round(max_diam, 2),
            min_diameter_mm=round(min_diam, 2),
            mean_diameter_mm=round(mean_diam, 2),
            total_volume_cm3=round(volume_cm3, 2),
            num_slices=len(slices_data),
            slices=slices_data
        )

        if output_json_path:
            output_json_path = Path(output_json_path)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"[MeshAnalyzer] Geometric report saved to: {output_json_path}")

        return report

    @staticmethod
    def compare_with_ground_truth(
        report: MeshAnalysisReport,
        physical_caliper_measurements: Dict[str, float]
    ) -> Dict:
        """
        Compare 3D reconstructed slices against manual physical caliper measurements.
        Example physical dict:
          {"height": 220.0, "body_diameter": 65.0, "neck_diameter": 28.0}
        """
        results = {}
        if "height" in physical_caliper_measurements:
            real_h = physical_caliper_measurements["height"]
            meas_h = report.total_height_mm
            diff_h = meas_h - real_h
            err_pct = (abs(diff_h) / real_h) * 100.0 if real_h > 0 else 0
            results["height_comparison"] = {
                "physical_mm": real_h,
                "scanned_3d_mm": meas_h,
                "error_mm": round(diff_h, 2),
                "error_pct": round(err_pct, 2)
            }

        return results
