"""
SIFT Feature Extractor and Matcher Module.
Wraps COLMAP's SIFT feature extraction and geometric matching under BaseMatcher interface.
"""
from pathlib import Path
import shutil
import sqlite3
import subprocess
from typing import Callable, Dict, List, Optional
import numpy as np

from src.matcher_interface import BaseMatcher, MatchingResult
from src.config import CameraConfig, MatcherConfig


def query_two_view_geometry_stats(database_path: Path) -> Dict[str, float]:
    """Query COLMAP SQLite database for verified two-view geometry inlier statistics."""
    if not database_path.exists():
        return {
            "total_pairs": 0, "valid_pairs": 0,
            "inliers_mean": 0.0, "inliers_median": 0.0,
            "inliers_p10": 0.0, "inliers_p90": 0.0
        }
    conn = sqlite3.connect(str(database_path))
    cur = conn.cursor()
    cur.execute("SELECT rows FROM two_view_geometries")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()

    total_pairs = len(rows)
    valid_rows = [r for r in rows if r > 0]
    valid_pairs = len(valid_rows)

    if valid_pairs > 0:
        return {
            "total_pairs": total_pairs,
            "valid_pairs": valid_pairs,
            "inliers_mean": float(np.mean(valid_rows)),
            "inliers_median": float(np.median(valid_rows)),
            "inliers_p10": float(np.percentile(valid_rows, 10)),
            "inliers_p90": float(np.percentile(valid_rows, 90))
        }
    return {
        "total_pairs": total_pairs, "valid_pairs": 0,
        "inliers_mean": 0.0, "inliers_median": 0.0,
        "inliers_p10": 0.0, "inliers_p90": 0.0
    }


class SIFTMatcher(BaseMatcher):
    """SIFT feature extraction and matching engine wrapping COLMAP CLI."""

    def __init__(self, config: MatcherConfig, colmap_bin: str = "colmap"):
        self.config = config
        if colmap_bin == "colmap":
            local_tools = Path("tools/colmap")
            possible_exes = list(local_tools.glob("**/colmap.exe")) + list(local_tools.glob("**/COLMAP.bat"))
            if possible_exes:
                colmap_bin = str(possible_exes[0].resolve())
            else:
                sys_colmap = shutil.which("colmap")
                if sys_colmap:
                    colmap_bin = sys_colmap
        self.colmap_bin = colmap_bin

    def extract_and_match(
        self,
        image_dir: Path,
        database_path: Path,
        camera_config: CameraConfig,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> MatchingResult:
        """
        Executes SIFT extraction and matching using an explicit image list to guarantee zero contamination.
        """
        from src.sfm_reconstruction import validate_sfm_image_inputs

        image_dir = Path(image_dir).resolve()
        database_path = Path(database_path).resolve()
        output_sfm_dir = database_path.parent
        output_sfm_dir.mkdir(parents=True, exist_ok=True)

        # 1. Strict validation of images and generation of explicit image_list.txt
        valid_image_names = validate_sfm_image_inputs(image_dir, output_sfm_dir)
        image_list_path = output_sfm_dir / "image_list.txt"

        if database_path.exists():
            database_path.unlink()

        # Check CUDA support
        has_cuda = False
        try:
            help_res = subprocess.run([self.colmap_bin, "help"], capture_output=True, text=True)
            has_cuda = "without CUDA" not in (help_res.stdout + help_res.stderr) and "CUDA" in (help_res.stdout + help_res.stderr)
        except Exception:
            has_cuda = False

        use_gpu_flag = "1" if (self.config.use_gpu and has_cuda) else "0"

        # 2. Feature Extractor
        cmd_extract = [
            self.colmap_bin, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--image_list_path", str(image_list_path),
            "--ImageReader.camera_model", camera_config.model,
            "--ImageReader.single_camera", "1" if camera_config.single_camera else "0",
            "--SiftExtraction.use_gpu", use_gpu_flag,
            "--SiftExtraction.max_num_features", str(self.config.max_features)
        ]

        if camera_config.fx is not None and camera_config.cx is not None and camera_config.cy is not None:
            if camera_config.model == "SIMPLE_RADIAL":
                k1 = camera_config.distortion_params[0] if camera_config.distortion_params else 0.0
                params_str = f"{camera_config.fx},{camera_config.cx},{camera_config.cy},{k1}"
            elif camera_config.model in ["PINHOLE", "RADIAL"]:
                fy = camera_config.fy if camera_config.fy is not None else camera_config.fx
                params_str = f"{camera_config.fx},{fy},{camera_config.cx},{camera_config.cy}"
            else:
                params_str = f"{camera_config.fx},{camera_config.cx},{camera_config.cy}"
            cmd_extract.extend(["--ImageReader.camera_params", params_str])

        print(f"[SIFTMatcher] Running SIFT feature extraction on {len(valid_image_names)} images (GPU: {use_gpu_flag})...")
        if progress_callback: progress_callback(15, f"Extrayendo features SIFT de {len(valid_image_names)} frames...")
        res = subprocess.run(cmd_extract, capture_output=True, text=True)
        if res.returncode != 0 and use_gpu_flag == "1":
            print("[SIFTMatcher] GPU extraction failed, retrying on CPU...")
            use_gpu_flag = "0"
            cmd_extract[cmd_extract.index("--SiftExtraction.use_gpu") + 1] = "0"
            res = subprocess.run(cmd_extract, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"COLMAP feature extraction failed:\n{res.stderr or res.stdout}")

        # 3. Matcher
        if self.config.sift_type == "exhaustive":
            matcher_cmd = "exhaustive_matcher"
            cmd_match = [
                self.colmap_bin, matcher_cmd,
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", use_gpu_flag
            ]
        else:
            matcher_cmd = "sequential_matcher"
            cmd_match = [
                self.colmap_bin, matcher_cmd,
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", use_gpu_flag,
                "--SequentialMatching.overlap", "10",
                "--SequentialMatching.quadratic_overlap", "1"
            ]

        print(f"[SIFTMatcher] Running {matcher_cmd} (GPU: {use_gpu_flag})...")
        if progress_callback: progress_callback(40, f"Emparejando pares con {matcher_cmd}...")
        res = subprocess.run(cmd_match, capture_output=True, text=True)
        if res.returncode != 0 and use_gpu_flag == "1":
            print(f"[SIFTMatcher] GPU {matcher_cmd} failed, retrying on CPU...")
            use_gpu_flag = "0"
            cmd_match[cmd_match.index("--SiftMatching.use_gpu") + 1] = "0"
            res = subprocess.run(cmd_match, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"COLMAP matching failed:\n{res.stderr or res.stdout}")

        stats = query_two_view_geometry_stats(database_path)
        print(f"[SIFTMatcher] Matching completed: {stats['valid_pairs']}/{stats['total_pairs']} valid pairs, Mean Inliers = {stats['inliers_mean']:.1f}")

        return MatchingResult(
            engine_name="SIFT",
            num_images=len(valid_image_names),
            total_pairs_processed=stats["total_pairs"],
            valid_pairs=stats["valid_pairs"],
            inliers_mean=stats["inliers_mean"],
            inliers_median=stats["inliers_median"],
            inliers_p10=stats["inliers_p10"],
            inliers_p90=stats["inliers_p90"],
            database_path=database_path,
            metadata={"matcher_cmd": matcher_cmd, "max_features": self.config.max_features}
        )
