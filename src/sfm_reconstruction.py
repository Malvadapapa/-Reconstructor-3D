"""
Structure-from-Motion (SfM) module using COLMAP.
Orchestrates feature extraction, matching, bundle adjustment, and exports camera poses + sparse point cloud.
"""
import os
import shutil
import struct
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import requests

from src.config import SfMConfig


# Download URL for official prebuilt COLMAP release on Windows (CPU/CUDA)
COLMAP_WINDOWS_NOCUDA_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/colmap-x64-windows-nocuda.zip"
COLMAP_WINDOWS_CUDA_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/colmap-x64-windows-cuda.zip"


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
                cameras[camera_id] = {
                    "camera_id": camera_id,
                    "model_id": model_id,
                    "width": width,
                    "height": height,
                    "params": np.array(params)
                }
        return cameras

    @staticmethod
    def read_images_binary(path_to_model_file: Path) -> Dict:
        """Read images.bin (quaternions, translation, 2D points)."""
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_reg_images):
                image_id = struct.unpack("<i", fid.read(4))[0]
                qvec = struct.unpack("<4d", fid.read(32))
                tvec = struct.unpack("<3d", fid.read(24))
                camera_id = struct.unpack("<i", fid.read(4))[0]
                image_name = ""
                current_char = fid.read(1)
                while current_char != b"\x00":
                    image_name += current_char.decode("utf-8", errors="ignore")
                    current_char = fid.read(1)
                num_points2D = struct.unpack("<Q", fid.read(8))[0]
                x_y_id_s = struct.unpack(f"<{2*num_points2D}d", fid.read(16 * num_points2D))
                point3D_ids = struct.unpack(f"<{num_points2D}q", fid.read(8 * num_points2D))
                
                xys = np.column_stack([x_y_id_s[0::2], x_y_id_s[1::2]])
                images[image_id] = {
                    "image_id": image_id,
                    "qvec": np.array(qvec),
                    "tvec": np.array(tvec),
                    "camera_id": camera_id,
                    "name": image_name,
                    "xys": xys,
                    "point3D_ids": np.array(point3D_ids)
                }
        return images

    @staticmethod
    def read_points3D_binary(path_to_model_file: Path) -> Dict:
        """Read points3D.bin (3D coordinates, RGB colors, reprojection error)."""
        points3D = {}
        with open(path_to_model_file, "rb") as fid:
            num_points = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_points):
                point3D_id = struct.unpack("<Q", fid.read(8))[0]
                xyz = struct.unpack("<3d", fid.read(24))
                rgb = struct.unpack("<3B", fid.read(3))
                error = struct.unpack("<d", fid.read(8))[0]
                track_length = struct.unpack("<Q", fid.read(8))[0]
                fid.read(8 * track_length) # skip track elements
                points3D[point3D_id] = {
                    "point3D_id": point3D_id,
                    "xyz": np.array(xyz),
                    "rgb": np.array(rgb),
                    "error": error
                }
        return points3D

    @staticmethod
    def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
        """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
        qw, qx, qy, qz = qvec
        return np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])


class SfMReconstructor:
    """Manages COLMAP execution and model data parsing."""

    def __init__(self, config: SfMConfig):
        self.config = config
        self.colmap_bin = self._find_or_setup_colmap()

    def _find_or_setup_colmap(self) -> str:
        """Locate COLMAP executable or check local tools directory."""
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

    def run_reconstruction(self, image_dir: Path, output_sfm_dir: Path, progress_callback=None) -> Dict:
        """
        Execute COLMAP pipeline:
        1. Feature extraction (SIFT)
        2. Feature matching
        3. Sparse Mapper (Bundle Adjustment)
        4. Model conversion to PLY
        """
        image_dir = Path(image_dir).resolve()
        output_sfm_dir = Path(output_sfm_dir).resolve()
        output_sfm_dir.mkdir(parents=True, exist_ok=True)

        database_path = output_sfm_dir / "database.db"
        sparse_dir = output_sfm_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        if database_path.exists():
            database_path.unlink()

        # Auto-detect if COLMAP binary has CUDA support
        has_cuda = False
        try:
            help_res = subprocess.run([self.colmap_bin, "help"], capture_output=True, text=True)
            has_cuda = "without CUDA" not in (help_res.stdout + help_res.stderr) and "CUDA" in (help_res.stdout + help_res.stderr)
        except Exception:
            has_cuda = False

        use_gpu_flag = "1" if (self.config.use_gpu and has_cuda) else "0"

        # 1. Feature Extractor
        cmd_extract = [
            self.colmap_bin, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", self.config.camera_model,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", use_gpu_flag,
            "--SiftExtraction.max_num_features", str(self.config.max_features)
        ]
        print(f"[SfM] Running feature extraction (GPU: {use_gpu_flag})...")
        if progress_callback: progress_callback(15, "Extrayendo features SIFT de cada frame...")
        res = subprocess.run(cmd_extract, capture_output=True, text=True)
        if res.returncode != 0 and use_gpu_flag == "1":
            print("[SfM] GPU extraction failed, retrying on CPU...")
            use_gpu_flag = "0"
            cmd_extract[cmd_extract.index("--SiftExtraction.use_gpu") + 1] = "0"
            res = subprocess.run(cmd_extract, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"COLMAP feature extraction failed:\n{res.stderr or res.stdout}")
        if progress_callback: progress_callback(35, "Features extraidos [OK] Emparejando imagenes...")

        # 2. Matcher (Sequential matcher is optimal for sequential video frames)
        if self.config.matcher_type == "exhaustive":
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

        print(f"[SfM] Running {matcher_cmd} (GPU: {use_gpu_flag})...")
        if progress_callback: progress_callback(40, f"Emparejando con {matcher_cmd}...")
        res = subprocess.run(cmd_match, capture_output=True, text=True)
        if res.returncode != 0 and use_gpu_flag == "1":
            print(f"[SfM] GPU {matcher_cmd} failed, retrying on CPU...")
            use_gpu_flag = "0"
            cmd_match[cmd_match.index("--SiftMatching.use_gpu") + 1] = "0"
            res = subprocess.run(cmd_match, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"COLMAP matching failed:\n{res.stderr or res.stdout}")
        if progress_callback: progress_callback(60, "Matching completado [OK] Construyendo mapa 3D...")

        return self.run_mapper(image_dir, output_sfm_dir, database_path, progress_callback)

    def run_mapper(
        self,
        image_dir: Path,
        output_sfm_dir: Path,
        database_path: Path,
        progress_callback=None,
        allow_reuse: bool = True
    ) -> Dict:
        """Run COLMAP mapper (Bundle Adjustment) on an already matched database."""
        sparse_dir = output_sfm_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

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
            cmd_map = [
                self.colmap_bin, "mapper",
                "--database_path", str(database_path),
                "--image_path", str(image_dir),
                "--output_path", str(sparse_dir),
                "--Mapper.min_num_matches", "10",
                "--Mapper.init_min_num_inliers", "15",
                "--Mapper.abs_pose_min_num_inliers", "15",
                "--Mapper.init_min_tri_angle", "4.0",
                "--Mapper.ba_refine_extra_params", "0"
            ]
            print(f"[SfM] Running sparse mapping...")
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

        print(f"[SfM] Reconstruction success! Registered {len(images)} cameras, {len(points3D)} 3D points.")

        return {
            "model_dir": str(primary_model_dir),
            "ply_path": str(ply_output_path),
            "num_registered_images": len(images),
            "num_points3d": len(points3D),
            "cameras": cameras,
            "images": images,
            "points3D": points3D,
            "xyz": xyz_coords,
            "rgb": rgb_colors
        }
