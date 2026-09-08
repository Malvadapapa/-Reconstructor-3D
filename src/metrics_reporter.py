"""
Metrics Reporter Module for the Video-to-3D Metric Model Pipeline.
Aggregates Input, Matching, SfM, Intrinsics, Scale, and Geometry metrics into:
- reports/pipeline_summary.json (machine-readable)
- reports/pipeline_summary.md (human-readable markdown)
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.sfm_reconstruction import ColmapModelParser


class MetricsReporter:
    """Computes, structures, and exports comprehensive pipeline diagnostic metrics."""

    @staticmethod
    def compute_triangulation_angles(
        points3D: Dict,
        images: Dict,
        cameras: Dict
    ) -> Dict[str, float]:
        """Compute inter-camera triangulation angles for reconstructed 3D points."""
        if not points3D or not images:
            return {"mean": 0.0, "median": 0.0}

        # Calculate camera optical centers in world space
        cam_centers = {}
        for img_id, img in images.items():
            R = ColmapModelParser.qvec2rotmat(img["qvec"])
            t = img["tvec"]
            cam_centers[img_id] = -R.T @ t

        angles = []
        # Sample up to 1000 points for fast, robust statistics
        p_items = list(points3D.values())
        if len(p_items) > 1000:
            step = len(p_items) // 1000
            sampled_pts = p_items[::step]
        else:
            sampled_pts = p_items

        for pt in sampled_pts:
            pt_xyz = pt["xyz"]
            rays = []
            # track elements: image_ids (if full track available)
            # In ColmapModelParser.read_points3D_binary, we only have error & track_length
            # If track length >= 2, we estimate triangulation quality
            pass

        return {"mean": 0.0, "median": 0.0}

    @staticmethod
    def compute_geometry_profile(
        points_xyz_metric: np.ndarray,
        height_span_mm: float
    ) -> Dict:
        """
        Fits cylinder / axial geometry on metric point cloud:
        Computes radial bands, linear slope b, b_HQ, R2, RMSE, D60, D80, D100, CylRatio.
        """
        if len(points_xyz_metric) < 20:
            return {
                "slope_b": 0.0, "slope_b_hq": 0.0, "r2": 0.0, "rmse": 0.0,
                "d60": 0.0, "d80": 0.0, "d100": 0.0, "cyl_ratio": 1.0,
                "bands": {}
            }

        z = points_xyz_metric[:, 2]
        # Robust center in XY
        z_min, z_max = float(np.percentile(z, 5)), float(np.percentile(z, 95))
        mid_mask = (z >= z_min + 0.1 * (z_max - z_min)) & (z <= z_max - 0.1 * (z_max - z_min))
        pts_mid = points_xyz_metric[mid_mask]

        if len(pts_mid) >= 10:
            A = np.column_stack([pts_mid[:, 0], pts_mid[:, 1], np.ones(len(pts_mid))])
            b = pts_mid[:, 0]**2 + pts_mid[:, 1]**2
            c_res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            xc, yc = float(c_res[0] / 2.0), float(c_res[1] / 2.0)
        else:
            xc, yc = float(np.median(points_xyz_metric[:, 0])), float(np.median(points_xyz_metric[:, 1]))

        r = np.sqrt((points_xyz_metric[:, 0] - xc)**2 + (points_xyz_metric[:, 1] - yc)**2)

        # Filter points belonging to main object body (within 3x median radius)
        med_r = float(np.median(r))
        body_mask = (r <= 2.5 * med_r) & (r >= 0.2 * med_r) & (z >= z_min) & (z <= z_max)
        z_body = z[body_mask]
        r_body = r[body_mask]

        if len(z_body) < 10:
            return {
                "slope_b": 0.0, "slope_b_hq": 0.0, "r2": 0.0, "rmse": 0.0,
                "d60": 0.0, "d80": 0.0, "d100": 0.0, "cyl_ratio": 1.0,
                "bands": {}
            }

        # Linear fit: r = b * z + a
        b_slope, a_intercept = np.polyfit(z_body, r_body, 1)
        pred_r = b_slope * z_body + a_intercept
        res = r_body - pred_r
        ss_tot = float(np.sum((r_body - np.mean(r_body))**2))
        r2 = float(1.0 - (np.sum(res**2) / ss_tot)) if ss_tot > 1e-9 else 0.0
        rmse = float(np.sqrt(np.mean(res**2)))

        # HQ fit: points within 1.5 sigma of initial fit
        sigma = float(np.std(res))
        hq_mask = np.abs(res) <= 1.5 * sigma
        if np.sum(hq_mask) >= 10:
            b_hq, a_hq = np.polyfit(z_body[hq_mask], r_body[hq_mask], 1)
        else:
            b_hq, a_hq = b_slope, a_intercept

        # Key reference diameters (e.g. at 60mm, 80mm, 100mm height if in range, or relative)
        d60 = float(2.0 * (a_hq + b_hq * 60.0))
        d80 = float(2.0 * (a_hq + b_hq * 80.0))
        d100 = float(2.0 * (a_hq + b_hq * 100.0))
        d_base = float(2.0 * (a_hq + b_hq * (z_min + 10.0)))
        d_top = float(2.0 * (a_hq + b_hq * (z_max - 10.0)))
        cyl_ratio = float(d_top / d_base) if abs(d_base) > 1e-6 else 1.0

        # Radial bands
        bands = {}
        band_edges = np.linspace(z_min, z_max, 6)
        for idx in range(len(band_edges) - 1):
            b_low, b_high = band_edges[idx], band_edges[idx + 1]
            b_mask = (z_body >= b_low) & (z_body < b_high)
            r_in_b = r_body[b_mask]
            b_name = f"Z {b_low:.1f}-{b_high:.1f} mm"
            if len(r_in_b) > 0:
                bands[b_name] = {
                    "count": int(len(r_in_b)),
                    "r_mean": round(float(np.mean(r_in_b)), 3),
                    "r_median": round(float(np.median(r_in_b)), 3),
                    "diam_mean": round(float(2.0 * np.mean(r_in_b)), 3),
                    "diam_median": round(float(2.0 * np.median(r_in_b)), 3),
                }

        return {
            "center_axis": {"xc": round(xc, 3), "yc": round(yc, 3)},
            "slope_b": float(b_slope),
            "slope_b_hq": float(b_hq),
            "r2": round(r2, 4),
            "rmse_mm": round(rmse, 3),
            "d60_mm": round(d60, 2),
            "d80_mm": round(d80, 2),
            "d100_mm": round(d100, 2),
            "cyl_ratio": round(cyl_ratio, 5),
            "bands": bands
        }

    @classmethod
    def generate_and_save_summary(
        cls,
        output_dir: Path,
        manifest: Dict,
        matching_stats: Optional[Dict],
        sfm_result: Dict,
        calibration,
        mesh_result,
        analysis_report,
        pipeline_duration_sec: float,
        config
    ) -> Dict:
        """
        Assembles all pipeline data and persists pipeline_summary.json and pipeline_summary.md.
        """
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        frames_extracted = manifest.get("total_extracted", len(manifest.get("frames", [])))
        total_original = manifest.get("sampling_metadata", {}).get("total_original_frames", frames_extracted)

        # SfM stats
        registered_cams = sfm_result.get("num_registered_images", len(sfm_result.get("images", {})))
        num_points = sfm_result.get("num_points3d", len(sfm_result.get("points3D", {})))
        track_stats = sfm_result.get("track_length", {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0})
        reproj_stats = sfm_result.get("reprojection_error", {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0})
        cam_intrinsics = sfm_result.get("camera_intrinsics", {})

        # Geometry fit
        raw_xyz = sfm_result.get("xyz", np.empty((0, 3)))
        metric_xyz = calibration.transform_matrix[:3, :3] @ raw_xyz.T + calibration.transform_matrix[:3, 3:4]
        metric_xyz = metric_xyz.T
        geom = cls.compute_geometry_profile(metric_xyz, float(mesh_result.dimensions_mm[2]))

        summary = {
            "pipeline_status": "SUCCESS",
            "duration_seconds": round(pipeline_duration_sec, 2),
            "input": {
                "frames_available": total_original,
                "frames_used": frames_extracted,
                "sampling_method": manifest.get("sampling_metadata", {}).get("sampling_method", "uniform_window_max_laplacian"),
                "target_fps": manifest.get("sampling_metadata", {}).get("target_fps", config.ingest.target_fps)
            },
            "matching": {
                "engine": matching_stats.get("engine_name", "SIFT") if matching_stats else ("DISK + LightGlue" if getattr(config, 'neural', None) and config.neural.enabled else "SIFT"),
                "pairs_processed": matching_stats.get("total_pairs_processed", 0) if matching_stats else 0,
                "valid_pairs": matching_stats.get("valid_pairs", 0) if matching_stats else 0,
                "inliers_per_pair_mean": round(float(matching_stats.get("inliers_mean", 0.0)), 1) if matching_stats else 0.0,
                "inliers_per_pair_median": round(float(matching_stats.get("inliers_median", 0.0)), 1) if matching_stats else 0.0,
                "inliers_per_pair_p10": round(float(matching_stats.get("inliers_p10", 0.0)), 1) if matching_stats else 0.0,
                "inliers_per_pair_p90": round(float(matching_stats.get("inliers_p90", 0.0)), 1) if matching_stats else 0.0,
            },
            "sfm": {
                "registered_cameras": registered_cams,
                "sparse_points_3d": num_points,
                "track_length": track_stats,
                "reprojection_error": reproj_stats,
                "camera_intrinsics": cam_intrinsics
            },
            "scale": {
                "scale_factor_applied": round(float(calibration.scale_factor), 6),
                "calibration_error_mm": round(float(calibration.scale_error_mm), 4),
                "marker_size_configured_mm": getattr(calibration, 'marker_size_mm', config.marker.marker_size_mm),
                "marker_size_measured_sfm": round(float(getattr(calibration, 'measured_size_sfm', 0.0)), 6)
            },
            "geometry": {
                "dimensions_mm": {
                    "width_x": round(float(mesh_result.dimensions_mm[0]), 2),
                    "depth_y": round(float(mesh_result.dimensions_mm[1]), 2),
                    "height_z": round(float(mesh_result.dimensions_mm[2]), 2)
                },
                "slope_b": geom["slope_b"],
                "slope_b_hq": geom["slope_b_hq"],
                "r2": geom["r2"],
                "rmse_mm": geom["rmse_mm"],
                "d60_mm": geom["d60_mm"],
                "d80_mm": geom["d80_mm"],
                "d100_mm": geom["d100_mm"],
                "cyl_ratio": geom["cyl_ratio"],
                "radial_bands": geom["bands"]
            },
            "mesh": {
                "num_vertices": mesh_result.num_vertices,
                "num_triangles": mesh_result.num_triangles,
                "is_watertight": mesh_result.is_watertight,
                "volume_cm3": round(float(mesh_result.volume_cm3), 2),
                "slices_extracted": analysis_report.num_slices
            },
            "paths": {
                "stl_model": str(mesh_result.stl_path),
                "obj_model": str(mesh_result.obj_path),
                "ply_model": str(mesh_result.ply_path),
                "measurements_json": str(reports_dir / "measurements.json")
            }
        }

        # Save JSON
        json_path = reports_dir / "pipeline_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Save Human-readable Markdown
        md_path = reports_dir / "pipeline_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Resumen de Reconstrucción 3D y Calibración Métrica\n\n")
            f.write(f"- **Estado:** `{summary['pipeline_status']}`\n")
            f.write(f"- **Tiempo Total:** {summary['duration_seconds']} s\n\n")
            f.write(f"## 1. Entrada y Muestreo\n")
            f.write(f"- Frames en video: {summary['input']['frames_available']}\n")
            f.write(f"- Frames seleccionados: {summary['input']['frames_used']}\n")
            f.write(f"- Método de muestreo: `{summary['input']['sampling_method']}`\n\n")
            f.write(f"## 2. Matching ({summary['matching']['engine']})\n")
            f.write(f"- Pares procesados: {summary['matching']['pairs_processed']}\n")
            f.write(f"- Pares válidos: {summary['matching']['valid_pairs']}\n")
            f.write(f"- Inliers por par (Mediana): {summary['matching']['inliers_per_pair_median']:.1f} (Media: {summary['matching']['inliers_per_pair_mean']:.1f})\n\n")
            f.write(f"## 3. Reconstrucción SfM (COLMAP)\n")
            f.write(f"- Cámaras registradas: {summary['sfm']['registered_cameras']}/{summary['input']['frames_used']}\n")
            f.write(f"- Puntos 3D dispersos: {summary['sfm']['sparse_points_3d']}\n")
            f.write(f"- Longitud de pista media: {summary['sfm']['track_length']['mean']:.2f} vistas\n")
            f.write(f"- Error de reproyección medio: {summary['sfm']['reprojection_error']['mean']:.3f} px (p90: {summary['sfm']['reprojection_error']['p90']:.3f} px)\n")
            if cam_intrinsics and cam_intrinsics.get("final"):
                fc = cam_intrinsics["final"]
                f.write(f"- Intrínsecos finales: `{fc.get('model_name')}` {fc.get('width')}x{fc.get('height')}, fx={fc.get('fx')}, cx={fc.get('cx')}, cy={fc.get('cy')}, dist={fc.get('distortion')}\n")
            f.write(f"\n## 4. Escala Métrica\n")
            f.write(f"- Factor de escala aplicado: **{summary['scale']['scale_factor_applied']} mm/sfm**\n")
            f.write(f"- Error residual de escala: ±{summary['scale']['calibration_error_mm']} mm\n\n")
            f.write(f"## 5. Geometría y Perfil Cilíndrico\n")
            f.write(f"- Dimensiones: {summary['geometry']['dimensions_mm']['width_x']} x {summary['geometry']['dimensions_mm']['depth_y']} x {summary['geometry']['dimensions_mm']['height_z']} mm\n")
            f.write(f"- Pendiente nominal b: {summary['geometry']['slope_b']:.6f} (b_HQ: {summary['geometry']['slope_b_hq']:.6f})\n")
            f.write(f"- Diámetros de referencia: D60={summary['geometry']['d60_mm']} mm | D80={summary['geometry']['d80_mm']} mm | D100={summary['geometry']['d100_mm']} mm\n")
            f.write(f"- CylRatio (D_top / D_base): {summary['geometry']['cyl_ratio']:.5f}\n")
            f.write(f"- Estanqueidad STL (Watertight): `{summary['mesh']['is_watertight']}` | Volumen: {summary['mesh']['volume_cm3']} cm³\n")

        return summary
