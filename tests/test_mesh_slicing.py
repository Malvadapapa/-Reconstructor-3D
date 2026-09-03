"""
Unit test for 3D mesh slicing, perimeter calculation, and geometric accuracy.
Creates a synthetic ground-truth cylinder and validates extracted measurements.
"""
import unittest
from pathlib import Path
import numpy as np
import trimesh

from src.config import SliceConfig
from src.mesh_analysis import MeshAnalyzer


class TestMeshSlicing(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("output/test_scratch")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_path = self.test_dir / "synthetic_cylinder.stl"

        # Create synthetic cylinder with known radius = 30 mm, height = 120 mm
        self.real_radius = 30.0
        self.real_height = 120.0
        self.real_perimeter = 2.0 * np.pi * self.real_radius # ~188.496 mm
        self.real_diameter = 2.0 * self.real_radius          # 60.0 mm
        self.real_area = np.pi * (self.real_radius ** 2)     # ~2827.43 mm²

        cylinder = trimesh.creation.cylinder(
            radius=self.real_radius,
            height=self.real_height,
            sections=64
        )
        # Translate so base is at Z = 0
        cylinder.apply_translation([0, 0, self.real_height / 2.0])
        cylinder.export(str(self.mesh_path))

    def test_slicing_geometric_precision(self):
        config = SliceConfig(step_height_mm=10.0, min_height_mm=10.0, max_height_mm=110.0)
        analyzer = MeshAnalyzer(config)
        report = analyzer.analyze_mesh(self.mesh_path, output_json_path=self.test_dir / "report.json")

        print(f"\n[TestMeshSlicing] Cylinder Ground Truth: D={self.real_diameter}mm, P={self.real_perimeter:.2f}mm, H={self.real_height}mm")
        print(f"[TestMeshSlicing] Reconstructed Slices: Mean D={report.mean_diameter_mm:.2f}mm, Total H={report.total_height_mm:.2f}mm")

        self.assertGreater(report.num_slices, 5)

        for s in report.slices:
            # Check perimeter is within 1% error
            self.assertAlmostEqual(s.perimeter_mm, self.real_perimeter, delta=self.real_perimeter * 0.02)
            # Check diameter is within 1 mm
            self.assertAlmostEqual(s.equivalent_diameter_mm, self.real_diameter, delta=1.0)
            self.assertTrue(s.is_closed)


if __name__ == "__main__":
    unittest.main()
