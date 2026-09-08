"""
Structure-from-Motion (SfM) module using COLMAP.
Orchestrates feature extraction, matching, bundle adjustment, and exports camera poses + sparse point cloud.
"""
import os
import shutil
import sqlite3
import struct
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import requests

from src.config import SfMConfig, CameraConfig


# Download URL for official prebuilt COLMAP release on Windows (CPU/CUDA)
COLMAP_WINDOWS_NOCUDA_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/colmap-x64-windows-nocuda.zip"
COLMAP_WINDOWS_CUDA_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/colmap-x64-windows-cuda.zip"


def validate_sfm_image_inputs(image_dir: Path, output_sfm_dir: Path) -> List[str]:
    """
    Strictly validates the SfM input directory:
    1. Validates only frame_*.jpg (or allowed image) files exist at top-level.
    2. Explicitly verifies no subdirectories (e.g. annotated/, debug/) exist inside image_dir.
    3. Aborts with clear error if any unauthorized, debug, or annotated image/folder is found.
    4. Writes and returns an explicit image_list.txt for COLMAP.
    """
    image_dir = Path(image_dir).resolve()
    output_sfm_dir = Path(output_sfm_dir).resolve()
    output_sfm_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"[SfM Validation Error] Image directory does not exist: {image_dir}")

    # 1. Check for unauthorized subdirectories
    subdirs = [d.name for d in image_dir.iterdir() if d.is_dir()]
    if subdirs:
        raise ValueError(
            f"[SfM Security Abort] Unauthorized subdirectories detected inside image directory: {subdirs}. "
            f"No subdirectories (especially 'annotated/' or 'debug/') are permitted inside {image_dir}."
        )

    # 2. Check for unauthorized files or non-frame images
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    valid_names = []
    invalid_files = []

    for item in sorted(image_dir.iterdir()):
        if item.is_file():
            name_lower = item.name.lower()
            if "annotated" in name_lower or "debug" in name_lower:
                invalid_files.append(item.name)
            elif item.suffix.lower() in allowed_exts:
                valid_names.append(item.name)
            else:
                invalid_files.append(item.name)

    if invalid_files:
        raise ValueError(
            f"[SfM Security Abort] Unauthorized debug or invalid files found in image directory: {invalid_files}. "
            f"Frames directory must only contain clean, selected input frames."
        )

    if len(valid_names) < 3:
        raise ValueError(
            f"[SfM Security Abort] Insufficient valid images in {image_dir}: found {len(valid_names)}, minimum required is 3."
        )

    print(f"[SfM Validation] Validated {len(valid_names)} clean frame images in {image_dir}.")
    print(f"[SfM Validation] Verified: zero subdirectories, zero annotated/debug contamination.")

    # Write explicit image_list.txt
    image_list_path = output_sfm_dir / "image_list.txt"
    with open(image_list_path, "w", encoding="utf-8") as f_list:
        for fname in valid_names:
            f_list.write(f"{fname}\n")

    return valid_names


def parse_camera_intrinsics(camera_dict: Dict) -> Dict:
    """Parse camera intrinsics parameters from COLMAP camera dictionary."""
    model_id = camera_dict.get("model_id", -1)
    model_name = camera_dict.get("model_name", "UNKNOWN")
    params = list(camera_dict.get("params", []))
    w = camera_dict.get("width", 0)
    h = camera_dict.get("height", 0)

    # Mapping models: 0: SIMPLE_PINHOLE, 1: PINHOLE, 2: SIMPLE_RADIAL, 3: RADIAL, 4: OPENCV
    distortion = []
    if model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
        fx = fy = params[0] if len(params) > 0 else 0.0
        cx = params[1] if len(params) > 1 else 0.0
        cy = params[2] if len(params) > 2 else 0.0
    elif model_id == 1: # PINHOLE: fx, fy, cx, cy
        fx = params[0] if len(params) > 0 else 0.0
        fy = params[1] if len(params) > 1 else 0.0
        cx = params[2] if len(params) > 2 else 0.0
        cy = params[3] if len(params) > 3 else 0.0
    elif model_id == 2: # SIMPLE_RADIAL: f, cx, cy, k1
        fx = fy = params[0] if len(params) > 0 else 0.0
        cx = params[1] if len(params) > 1 else 0.0
        cy = params[2] if len(params) > 2 else 0.0
        distortion = [params[3]] if len(params) > 3 else []
    elif model_id == 3: # RADIAL: f, cx, cy, k1, k2
        fx = fy = params[0] if len(params) > 0 else 0.0
        cx = params[1] if len(params) > 1 else 0.0
        cy = params[2] if len(params) > 2 else 0.0
        distortion = params[3:5] if len(params) > 4 else []
    elif model_id == 4: # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        fx = params[0] if len(params) > 0 else 0.0
        fy = params[1] if len(params) > 1 else 0.0
        cx = params[2] if len(params) > 2 else 0.0
        cy = params[3] if len(params) > 3 else 0.0
        distortion = params[4:] if len(params) > 4 else []
    else:
        fx = fy = params[0] if len(params) > 0 else 0.0
        cx = params[1] if len(params) > 1 else 0.0
        cy = params[2] if len(params) > 2 else 0.0

    return {
        "model_id": model_id,
        "model_name": model_name,
        "width": w,
        "height": h,
        "fx": round(float(fx), 3),
        "fy": round(float(fy), 3),
        "cx": round(float(cx), 3),
        "cy": round(float(cy), 3),
        "distortion": [round(float(k), 6) for k in distortion],
        "raw_params": [round(float(p), 6) for p in params]
    }


class ColmapModelParser:
    """Parses binary or text COLMAP output files (cameras, images, points3D)."""

    @staticmethod
    def read_cameras_binary(path_to_model_file: Path) -> Dict:
        """Read cameras.bin."""
        cameras = {}
        with open(path_to_model_file, "rb") as fid:
            num_cameras = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_cameras):
                camera_id, model_id, width, height = struct.unpack("<iiQQ", fid.read(24))
                # Number of params depends on model_id
                num_params_map = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5, 6: 8, 7: 12, 8: 4, 9: 5, 10: 1}
                num_params = num_params_map.get(model_id, 4)
                params = struct.unpack(f"<{num_params}d", fid.read(8 * num_params))
                model_names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV"}
                cameras[camera_id] = {
                    "camera_id": camera_id,
                    "model_id": model_id,
                    "model_name": model_names.get(model_id, "UNKNOWN"),
                    "width": width,
                    "height": height,
                    "params": np.array(params)
                }
        return cameras

    @staticmethod
    def read_images_binary(path_to_model_file: Path) -> Dict:
        """Read images.bin."""
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_reg_images):
                image_id = struct.unpack("<i", fid.read(4))[0]
                qvec = np.array(struct.unpack("<4d", fid.read(32)))
                tvec = np.array(struct.unpack("<3d", fid.read(24)))
                camera_id = struct.unpack("<i", fid.read(4))[0]
                image_name = ""
                char = fid.read(1)
                while char != b"\x00":
                    image_name += char.decode("utf-8", errors="replace")
                    char = fid.read(1)
                num_points2D = struct.unpack("<Q", fid.read(8))[0]
                fid.seek(num_points2D * 24, 1) # Skip 2D points data
                images[image_id] = {
                    "image_id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": image_name
                }
        return images

    @staticmethod
    def read_points3D_binary(path_to_model_file: Path) -> Dict:
        """Read points3D.bin."""
        points3D = {}
        with open(path_to_model_file, "rb") as fid:
            num_points = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_points):
                point3D_id = struct.unpack("<Q", fid.read(8))[0]
                xyz = np.array(struct.unpack("<3d", fid.read(24)))
                rgb = np.array(struct.unpack("<3B", fid.read(3)))
                error = struct.unpack("<d", fid.read(8))[0]
                track_length = struct.unpack("<Q", fid.read(8))[0]
                fid.seek(track_length * 8, 1) # Skip track data
                points3D[point3D_id] = {
                    "id": point3D_id,
                    "xyz": xyz,
                    "rgb": rgb,
                    "error": error,
                    "track_length": track_length
                }
        return points3D

    @staticmethod
    def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = qvec
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])


class SfMReconstructor:
    """Manages COLMAP Structure from Motion execution."""

    def __init__(self, config: SfMConfig):
        self.config = config
        self.colmap_bin = self._find_colmap_binary()

    def _find_colmap_binary(self) -> str:
        """Locates COLMAP binary on Windows."""
        # 1. Configured path
        if self.config.colmap_binary_path and Path(self.config.colmap_binary_path).exists():
            return str(Path(self.config.colmap_binary_path).resolve())

        # 2. System PATH
        system_colmap = shutil.which("colmap")
        if system_colmap:
            return system_colmap

        # 3. Local tools folder
        local_tools = Path("tools/colmap")
        possible_exes = list(local_tools.glob("**/colmap.exe")) + list(local_tools.glob("**/COLMAP.bat"))
        if possible_exes:
            return str(possible_exes[0].resolve())

        return "colmap"

    @staticmethod
    def auto_download_colmap(dest_dir: Path = Path("tools/colmap"), use_cuda: bool = False) -> Optional[Path]:
        """Download and extract prebuilt COLMAP release for Windows if needed."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / "colmap.zip"
        download_url = COLMAP_WINDOWS_CUDA_URL if use_cuda else COLMAP_WINDOWS_NOCUDA_URL

        print(f"[SfM] Downloading COLMAP from {download_url}...")
        try:
            resp = requests.get(download_url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            print("[SfM] Extracting COLMAP zip...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
            if zip_path.exists():
                zip_path.unlink()

            possible_exes = list(dest_dir.glob("**/colmap.exe")) + list(dest_dir.glob("**/COLMAP.bat"))
            if possible_exes:
                print(f"[SfM] COLMAP successfully installed at: {possible_exes[0]}")
                return possible_exes[0]
        except Exception as e:
            print(f"[SfM] Error downloading COLMAP: {e}")
        return None

    def run_reconstruction(
        self,
        image_dir: Path,
        output_sfm_dir: Path,
        camera_config: Optional[CameraConfig] = None,
        progress_callback=None
    ) -> Dict:
        """
        Execute COLMAP pipeline:
        1. Feature extraction (SIFT) with strict image_list.txt validation
        2. Feature matching
        3. Sparse Mapper (Bundle Adjustment)
        4. Model conversion to PLY
        """
        image_dir = Path(image_dir).resolve()
        output_sfm_dir = Path(output_sfm_dir).resolve()
        output_sfm_dir.mkdir(parents=True, exist_ok=True)

        database_path = output_sfm_dir / "database.db"

        from src.sift_matcher import SIFTMatcher
        from src.config import MatcherConfig

        matcher_cfg = MatcherConfig(
            engine="sift",
            sift_type=self.config.matcher_type,
            max_features=self.config.max_features,
            use_gpu=self.config.use_gpu
        )
        matcher = SIFTMatcher(matcher_cfg, colmap_bin=self.colmap_bin)
        cam_cfg = camera_config or CameraConfig(model=self.config.camera_model)

        matching_result = matcher.extract_and_match(
            image_dir=image_dir,
            database_path=database_path,
            camera_config=cam_cfg,
            progress_callback=progress_callback
        )

        sfm_res = self.run_mapper(image_dir, output_sfm_dir, database_path, camera_config=cam_cfg, progress_callback=progress_callback)
        sfm_res["matching_result"] = matching_result
        return sfm_res

    def run_mapper(
        self,
        image_dir: Path,
        output_sfm_dir: Path,
        database_path: Path,
        camera_config: Optional[CameraConfig] = None,
        progress_callback=None,
        allow_reuse: bool = True
    ) -> Dict:
        """Run COLMAP mapper (Bundle Adjustment) on an already matched database."""
        image_dir = Path(image_dir).resolve()
        output_sfm_dir = Path(output_sfm_dir).resolve()
        database_path = Path(database_path).resolve()

        sparse_dir = output_sfm_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        # Extract initial camera intrinsics from SQLite database prior to Mapper
        initial_cam_intrinsics = None
        try:
            if database_path.exists():
                conn = sqlite3.connect(str(database_path))
                cur = conn.cursor()
                cur.execute("SELECT model, width, height, params FROM cameras WHERE camera_id=1")
                row = cur.fetchone()
                conn.close()
                if row:
                    model_id, w, h, p_blob = row
                    num_p = len(p_blob) // 8
                    p_vals = list(struct.unpack(f"<{num_p}d", p_blob))
                    model_names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV"}
                    initial_cam_intrinsics = parse_camera_intrinsics({
                        "model_id": model_id,
                        "model_name": model_names.get(model_id, "UNKNOWN"),
                        "width": w,
                        "height": h,
                        "params": p_vals
                    })
        except Exception as e:
            print(f"[SfM] Notice: Could not read initial camera from db: {e}")

        def get_model_size(mdir: Path) -> int:
            bin_file = mdir / "images.bin"
            if bin_file.exists():
                try:
                    return len(ColmapModelParser.read_images_binary(bin_file))
                except Exception:
                    pass
            return 0

        # Check if completed sparse reconstruction already exists
        existing_subdirs = [d for d in sparse_dir.iterdir() if d.is_dir()] if sparse_dir.exists() else []
        reused = False
        if allow_reuse and existing_subdirs:
            primary_cand = max(existing_subdirs, key=get_model_size)
            if get_model_size(primary_cand) >= 3:
                print(f"[SfM] Reusing existing sparse reconstruction from '{primary_cand.name}' with {get_model_size(primary_cand)} registered cameras.")
                if progress_callback: progress_callback(90, f"⚡ Reutilizando mapa 3D previo ({get_model_size(primary_cand)} cámaras)...")
                reused = True

        if not reused:
            # 3. Mapper (Bundle Adjustment)
            refine_focal = "1" if (not camera_config or camera_config.refine_focal_length) else "0"
            refine_pp = "1" if (camera_config and camera_config.refine_principal_point) else "0"
            refine_extra = "1" if (camera_config and camera_config.refine_extra_params) else "0"

            cmd_map = [
                self.colmap_bin, "mapper",
                "--database_path", str(database_path),
                "--image_path", str(image_dir),
                "--output_path", str(sparse_dir),
                "--Mapper.min_num_matches", "10",
                "--Mapper.init_min_num_inliers", "15",
                "--Mapper.abs_pose_min_num_inliers", "15",
                "--Mapper.init_min_tri_angle", "4.0",
                "--Mapper.ba_refine_focal_length", refine_focal,
                "--Mapper.ba_refine_principal_point", refine_pp,
                "--Mapper.ba_refine_extra_params", refine_extra
            ]
            print(f"[SfM] Running sparse mapping (BA refine: focal={refine_focal}, pp={refine_pp}, extra={refine_extra})...")
            if progress_callback: progress_callback(65, "Ejecutando Bundle Adjustment (mapper)...")
            res = subprocess.run(cmd_map, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"COLMAP mapper failed:\n{res.stderr or res.stdout}")
            if progress_callback: progress_callback(90, "Reconstrucción 3D completada [OK]")

        # Check model index and pick the LARGEST submodel (most registered images)
        model_subdirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
        if not model_subdirs:
            raise RuntimeError("COLMAP did not produce any reconstructed sparse submodel.")

        primary_model_dir = max(model_subdirs, key=get_model_size)
        print(f"[SfM] Selected primary submodel '{primary_model_dir.name}' with {get_model_size(primary_model_dir)} registered cameras.")

        # 4. Export PLY point cloud
        ply_output_path = output_sfm_dir / "sparse_points.ply"
        cmd_convert = [
            self.colmap_bin, "model_converter",
            "--input_path", str(primary_model_dir),
            "--output_path", str(ply_output_path),
            "--output_type", "PLY"
        ]
        subprocess.run(cmd_convert, capture_output=True, text=True)

        # 5. Parse cameras, images, points3D
        cameras = {}
        images = {}
        points3D = {}
        if (primary_model_dir / "cameras.bin").exists():
            cameras = ColmapModelParser.read_cameras_binary(primary_model_dir / "cameras.bin")
            images = ColmapModelParser.read_images_binary(primary_model_dir / "images.bin")
            points3D = ColmapModelParser.read_points3D_binary(primary_model_dir / "points3D.bin")

        xyz_coords = np.array([p["xyz"] for p in points3D.values()]) if points3D else np.empty((0, 3))
        rgb_colors = np.array([p["rgb"] for p in points3D.values()]) if points3D else np.empty((0, 3))

        # Parse final optimized camera intrinsics
        final_cam_intrinsics = None
        if cameras:
            prim_cam = cameras.get(1, next(iter(cameras.values())))
            final_cam_intrinsics = parse_camera_intrinsics(prim_cam)

        # Compute point cloud & tracking statistics
        track_lengths = [int(p["track_length"]) for p in points3D.values()] if points3D else []
        reproj_errors = [float(p["error"]) for p in points3D.values()] if points3D else []

        track_stats = {
            "mean": float(np.mean(track_lengths)) if track_lengths else 0.0,
            "median": float(np.median(track_lengths)) if track_lengths else 0.0,
            "p10": float(np.percentile(track_lengths, 10)) if track_lengths else 0.0,
            "p90": float(np.percentile(track_lengths, 90)) if track_lengths else 0.0,
        }
        reproj_stats = {
            "mean": float(np.mean(reproj_errors)) if reproj_errors else 0.0,
            "median": float(np.median(reproj_errors)) if reproj_errors else 0.0,
            "p10": float(np.percentile(reproj_errors, 10)) if reproj_errors else 0.0,
            "p90": float(np.percentile(reproj_errors, 90)) if reproj_errors else 0.0,
        }

        print(f"[SfM] Reconstruction success! Registered {len(images)} cameras, {len(points3D)} 3D points.")
        if final_cam_intrinsics:
            print(f"[SfM] Final Intrinsics: fx={final_cam_intrinsics['fx']}, fy={final_cam_intrinsics['fy']}, cx={final_cam_intrinsics['cx']}, cy={final_cam_intrinsics['cy']}, dist={final_cam_intrinsics['distortion']}")

        return {
            "model_dir": str(primary_model_dir),
            "ply_path": str(ply_output_path),
            "num_registered_images": len(images),
            "num_points3d": len(points3D),
            "cameras": cameras,
            "images": images,
            "points3D": points3D,
            "xyz": xyz_coords,
            "rgb": rgb_colors,
            "camera_intrinsics": {
                "initial": initial_cam_intrinsics,
                "final": final_cam_intrinsics,
                "refinement_flags": {
                    "ba_refine_focal_length": bool(camera_config.refine_focal_length) if camera_config else True,
                    "ba_refine_principal_point": bool(camera_config.refine_principal_point) if camera_config else False,
                    "ba_refine_extra_params": bool(camera_config.refine_extra_params) if camera_config else False,
                }
            },
            "track_length": track_stats,
            "reprojection_error": reproj_stats
        }
