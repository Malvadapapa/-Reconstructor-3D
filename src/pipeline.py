"""
End-to-End Pipeline Orchestrator for Video to 3D Metric Model.
Connects Video Ingestion -> SfM (COLMAP) -> Metric Calibration -> Open3D Poisson Meshing -> Slicing Analysis.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.config import PipelineConfig
from src.video_ingest import VideoIngestor
from src.sfm_reconstruction import SfMReconstructor
from src.metric_scaler import MetricScaler, MetricCalibrationResult
from src.mesh_generator import MeshGenerator, MeshGenerationResult
from src.mesh_analysis import MeshAnalyzer, MeshAnalysisReport
from src.neural_matcher import NeuralMatcher
from src.texture_pipeline import TexturePipeline, TextureBakingResult


@dataclass
class PipelineResult:
    """Overall results of the 3D scanning and metric reconstruction pipeline."""
    success: bool
    total_time_sec: float
    video_frames_extracted: int
    num_registered_cameras: int
    num_sparse_points: int
    scale_factor_mm: float
    stl_model_path: Path
    obj_model_path: Path
    measurements_json_path: Path
    summary_report: Dict
    textured_obj_path: Optional[Path] = None
    textured_glb_path: Optional[Path] = None
    texture_atlas_path: Optional[Path] = None


class VideoTo3DPipeline:
    """Main execution engine for converting a video to a calibrated 3D model."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.ensure_dirs()

        self.ingestor = VideoIngestor(config.ingest, config.marker)
        self.sfm = SfMReconstructor(config.sfm)
        self.scaler = MetricScaler(config.marker)
        self.mesher = MeshGenerator(config.mesh)
        self.analyzer = MeshAnalyzer(config.slice)
        self.neural_matcher = NeuralMatcher(config.neural, colmap_bin=self.sfm.colmap_bin) if config.neural.enabled else None
        self.texture_pipeline = TexturePipeline(config.texture) if config.texture.enabled else None

    def run(self) -> PipelineResult:
        """Execute all stages sequentially."""
        start_time = time.time()
        print("\n" + "=" * 80)
        print("  VIDEO TO 3D METRIC MODEL PIPELINE - START")
        print("=" * 80)
        print(f"Video input:    {self.config.video_path}")
        print(f"Output dir:     {self.config.output_dir}")
        print(f"Marker ID:      {self.config.marker.marker_id} ({self.config.marker.marker_size_mm} mm)")
        print(f"Neural Matcher: {'ENABLED (DISK + LightGlue)' if self.config.neural.enabled else 'DISABLED (SIFT)'}")
        print(f"UV Texturing:   {'ENABLED (xatlas + Baking)' if self.config.texture.enabled else 'DISABLED'}")
        print("-" * 80)

        # STAGE 1: Video Ingestion & Keyframe Filtering
        print("\n--- STAGE 1: Video Ingest & Frame Quality Filter ---")
        frames_dir = self.config.output_dir / "frames"
        manifest = self.ingestor.process_video(self.config.video_path, frames_dir)
        total_extracted = manifest["total_extracted"]

        if total_extracted < 5:
            raise ValueError(f"Demasiados pocos frames nítidos extraídos ({total_extracted}). Por favor grabe un video un poco más largo o con mejor iluminación.")

        # STAGE 2: Structure from Motion (COLMAP)
        print("\n--- STAGE 2: Structure from Motion (SfM) ---")
        sfm_dir = self.config.output_dir / "sfm"
        database_path = sfm_dir / "database.db"

        if self.config.neural.enabled and self.neural_matcher:
            print("[Pipeline] Running Neural Feature Extraction (DISK) & Matching (LightGlue)...")
            self.neural_matcher.run_neural_sfm_prep(frames_dir, database_path)
            sfm_result = self.sfm.run_mapper(frames_dir, sfm_dir, database_path)
        else:
            sfm_result = self.sfm.run_reconstruction(frames_dir, sfm_dir)

        if sfm_result["num_registered_images"] < 3:
            raise RuntimeError(f"COLMAP registered only {sfm_result['num_registered_images']} cameras. Reconstruction failed.")

        # STAGE 3: Metric Calibration & Coordinate Alignment
        print("\n--- STAGE 3: Metric Scaling & Marker Alignment ---")
        calibration: MetricCalibrationResult = self.scaler.calibrate_and_align(sfm_result, manifest)
        
        # Apply transformation matrix (Scale + Alignment) to 3D point cloud
        raw_xyz = sfm_result["xyz"]
        metric_xyz = MetricScaler.apply_transform_to_points(raw_xyz, calibration.transform_matrix)
        rgb_colors = sfm_result["rgb"]

        # STAGE 4: Poisson Surface Reconstruction & STL Generation
        print("\n--- STAGE 4: Mesh Generation & Solidification ---")
        mesh_dir = self.config.output_dir / "mesh"
        mesh_result: MeshGenerationResult = self.mesher.generate_mesh(
            points_xyz_metric=metric_xyz,
            colors_rgb=rgb_colors,
            output_dir=mesh_dir,
            base_name=self.config.video_path.stem
        )

        # STAGE 4B: UV Parametrization & Texture Baking (Optional)
        textured_obj_path = None
        textured_glb_path = None
        texture_atlas_path = None

        if self.config.texture.enabled and self.texture_pipeline:
            print("\n--- STAGE 4B: UV Parametrization & Texture Baking ---")
            texture_dir = self.config.output_dir / "texture"
            baking_result = self.texture_pipeline.bake_texture(
                mesh_path=mesh_result.stl_path,
                sfm_result=sfm_result,
                frames_dir=frames_dir,
                output_dir=texture_dir,
                base_name=self.config.video_path.stem
            )
            textured_obj_path = baking_result.obj_model_path
            textured_glb_path = baking_result.glb_model_path
            texture_atlas_path = baking_result.atlas_png_path

        # STAGE 5: Geometric Slicing & Measurement Extraction
        print("\n--- STAGE 5: Slicing & Geometric Profile Analysis ---")
        reports_dir = self.config.output_dir / "reports"
        report_json_path = reports_dir / "measurements.json"
        analysis_report: MeshAnalysisReport = self.analyzer.analyze_mesh(
            mesh_path=mesh_result.stl_path,
            output_json_path=report_json_path
        )

        total_duration = time.time() - start_time

        # Build final summary dictionary
        summary = {
            "pipeline_status": "SUCCESS",
            "duration_seconds": round(total_duration, 2),
            "video_frames": total_extracted,
            "registered_cameras": sfm_result["num_registered_images"],
            "sparse_points_3d": sfm_result["num_points3d"],
            "scale_factor_applied": calibration.scale_factor,
            "calibration_error_mm": calibration.scale_error_mm,
            "features_engine": "DISK + LightGlue" if self.config.neural.enabled else "SIFT",
            "mesh_dimensions_mm": {
                "width_x": round(float(mesh_result.dimensions_mm[0]), 2),
                "depth_y": round(float(mesh_result.dimensions_mm[1]), 2),
                "height_z": round(float(mesh_result.dimensions_mm[2]), 2),
            },
            "mesh_stats": {
                "num_vertices": mesh_result.num_vertices,
                "num_triangles": mesh_result.num_triangles,
                "is_watertight": mesh_result.is_watertight,
                "volume_cm3": mesh_result.volume_cm3,
            },
            "slices_extracted": analysis_report.num_slices,
            "texture": {
                "enabled": self.config.texture.enabled,
                "atlas_resolution": self.config.texture.atlas_resolution if self.config.texture.enabled else None,
                "atlas_path": str(texture_atlas_path) if texture_atlas_path else None,
                "textured_obj": str(textured_obj_path) if textured_obj_path else None,
                "textured_glb": str(textured_glb_path) if textured_glb_path else None,
            },
            "paths": {
                "stl_model": str(mesh_result.stl_path),
                "obj_model": str(mesh_result.obj_path),
                "ply_model": str(mesh_result.ply_path),
                "textured_obj": str(textured_obj_path) if textured_obj_path else None,
                "textured_glb": str(textured_glb_path) if textured_glb_path else None,
                "measurements_json": str(report_json_path)
            }
        }

        # Save summary report
        summary_path = reports_dir / "pipeline_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("  PIPELINE COMPLETE - SUMMARY OF RESULTS")
        print("=" * 80)
        print(f"Total processing time: {total_duration:.1f} s")
        print(f"Watertight Solid STL:  {mesh_result.stl_path}")
        if textured_glb_path:
            print(f"Textured 3D GLB:       {textured_glb_path}")
        print(f"Dimensions (X x Y x Z): {mesh_result.dimensions_mm[0]:.1f} x {mesh_result.dimensions_mm[1]:.1f} x {mesh_result.dimensions_mm[2]:.1f} mm")
        print(f"Estimated Volume:      {mesh_result.volume_cm3:.1f} cm³")
        print(f"Cross-sections Sliced: {analysis_report.num_slices}")
        print("=" * 80 + "\n")

        return PipelineResult(
            success=True,
            total_time_sec=total_duration,
            video_frames_extracted=total_extracted,
            num_registered_cameras=sfm_result["num_registered_images"],
            num_sparse_points=sfm_result["num_points3d"],
            scale_factor_mm=calibration.scale_factor,
            stl_model_path=mesh_result.stl_path,
            obj_model_path=mesh_result.obj_path,
            measurements_json_path=report_json_path,
            summary_report=summary,
            textured_obj_path=textured_obj_path,
            textured_glb_path=textured_glb_path,
            texture_atlas_path=texture_atlas_path
        )
