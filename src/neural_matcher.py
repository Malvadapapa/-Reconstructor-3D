"""
Neural Feature Extraction and Matching module using DISK and LightGlue.
Generates robust keypoints and deep feature correspondences, writing directly
to COLMAP-compatible SQLite database and running geometric verification for SfM.
"""
import math
import os
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import NeuralMatcherConfig


class NeuralMatcher:
    """
    Neural Feature Extraction and Matching engine using DISK + LightGlue.
    Optimized for biological textures, smooth surfaces, and monocular video orbits.
    """

    def __init__(self, config: NeuralMatcherConfig, colmap_bin: str = "colmap"):
        self.config = config
        self.colmap_bin = colmap_bin
        self._device = None
        self._extractor = None
        self._matcher = None

    def _init_models(self):
        """Lazy load PyTorch, DISK, and LightGlue models onto configured device."""
        if self._extractor is not None and self._matcher is not None:
            return

        import torch
        from lightglue import DISK, LightGlue

        if self.config.device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = self.config.device

        self._device = torch.device(device_str)
        print(f"[NeuralMatcher] Initializing DISK (extractor) + LightGlue (matcher) on {self._device}...")

        # Initialize DISK (dense feature detector + descriptor)
        self._extractor = DISK(max_num_keypoints=2048).eval().to(self._device)

        # Initialize LightGlue (deep transformer matcher)
        self._matcher = LightGlue(
            features="disk",
            filter_threshold=self.config.filter_threshold,
            depth_confidence=-1,
            width_confidence=-1,
        ).eval().to(self._device)

    def extract_frame_features(self, image_path: Path) -> Dict[str, np.ndarray]:
        """
        Extract DISK keypoints and descriptors for a single image.
        Returns dictionary with 'keypoints' (N, 2), 'descriptors' (N, 128),
        'width', 'height'.
        """
        self._init_models()
        import torch
        from lightglue.utils import load_image, rbd

        image_tensor = load_image(str(image_path)).to(self._device)
        if image_tensor.dim() == 4:
            _, _, h, w = image_tensor.shape
        else:
            _, h, w = image_tensor.shape

        with torch.inference_mode():
            feats = self._extractor.extract(image_tensor)
            feats = rbd(feats)

        kpts = feats["keypoints"].cpu().numpy().astype(np.float32)
        descs = feats["descriptors"].cpu().numpy().astype(np.float32)

        return {
            "keypoints": kpts,
            "descriptors": descs,
            "width": int(w),
            "height": int(h)
        }

    def match_pair(self, feats0: Dict, feats1: Dict) -> np.ndarray:
        """
        Match precomputed features between two frames using LightGlue.
        Returns array of match index pairs: shape (M, 2).
        """
        self._init_models()
        import torch

        t_feats0 = {
            "keypoints": torch.from_numpy(feats0["keypoints"]).unsqueeze(0).to(self._device),
            "descriptors": torch.from_numpy(feats0["descriptors"]).unsqueeze(0).to(self._device),
            "image_size": torch.tensor([[feats0["width"], feats0["height"]]], device=self._device)
        }
        t_feats1 = {
            "keypoints": torch.from_numpy(feats1["keypoints"]).unsqueeze(0).to(self._device),
            "descriptors": torch.from_numpy(feats1["descriptors"]).unsqueeze(0).to(self._device),
            "image_size": torch.tensor([[feats1["width"], feats1["height"]]], device=self._device)
        }

        with torch.inference_mode():
            matches_dict = self._matcher({"image0": t_feats0, "image1": t_feats1})
            matches = matches_dict["matches"][0].cpu().numpy()

        return matches.astype(np.uint32)

    def generate_image_pairs(self, image_names: List[str]) -> List[Tuple[int, int]]:
        """
        Generate image index pairs (i < j).
        Uses exhaustive matching for <= max_frames_exhaustive,
        or sequential sliding window (k=8) + circular loop closure for longer videos.
        """
        n = len(image_names)
        pairs: List[Tuple[int, int]] = []

        if n <= self.config.max_frames_exhaustive:
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((i, j))
        else:
            window = 8
            pair_set = set()
            for i in range(n):
                for w in range(1, window + 1):
                    j = (i + w) % n
                    idx1, idx2 = min(i, j), max(i, j)
                    if idx1 != idx2 and (idx1, idx2) not in pair_set:
                        pair_set.add((idx1, idx2))
                        pairs.append((idx1, idx2))

        return pairs

    def run_neural_sfm_prep(
        self,
        image_dir: Path,
        database_path: Path,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict:
        """
        Full pipeline for neural feature extraction and matching into a COLMAP database:
        1. Initialize COLMAP SQLite database schema with single shared camera.
        2. Extract DISK features for all images and store keypoints in SQLite.
        3. Match image pairs with LightGlue.
        4. Write raw match list and execute COLMAP 'matches_importer' for TwoViewGeometry verification.
        """
        self._init_models()
        image_dir = Path(image_dir)
        database_path = Path(database_path)

        # Collect image files sorted
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = sorted([
            f for f in image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])

        if len(image_files) < 3:
            raise ValueError(f"[NeuralMatcher] Need at least 3 images, found {len(image_files)}")

        if progress_callback: progress_callback(12, f"Inicializando base de datos COLMAP ({len(image_files)} frames)...")

        # 1. Read first image to determine camera resolution
        sample_img = cv2.imread(str(image_files[0]))
        if sample_img is None:
            raise RuntimeError(f"Cannot read image: {image_files[0]}")
        h, w = sample_img.shape[:2]

        # 2. Setup COLMAP SQLite database
        if database_path.exists():
            database_path.unlink()

        conn = sqlite3.connect(str(database_path))
        cur = conn.cursor()

        # Create standard COLMAP tables
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                model INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                params BLOB,
                prior_focal_length INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name TEXT NOT NULL UNIQUE,
                camera_id INTEGER NOT NULL,
                prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
                prior_tx REAL, prior_ty REAL, prior_tz REAL,
                CONSTRAINT image_id_check CHECK(image_id >= 0)
            );

            CREATE TABLE IF NOT EXISTS keypoints (
                image_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB
            );

            CREATE TABLE IF NOT EXISTS descriptors (
                image_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB
            );

            CREATE TABLE IF NOT EXISTS matches (
                pair_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB
            );

            CREATE TABLE IF NOT EXISTS two_view_geometries (
                pair_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                config INTEGER NOT NULL,
                F BLOB, E BLOB, H BLOB,
                qvec BLOB, tvec BLOB
            );
        """)

        # Insert Camera 1: SIMPLE_RADIAL (model_id 2) -> params: [f, cx, cy, k]
        focal_prior = 1.2 * max(w, h)
        cx, cy = w / 2.0, h / 2.0
        k = 0.0
        camera_params = np.array([focal_prior, cx, cy, k], dtype=np.float64)
        cur.execute(
            "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
            (1, 2, w, h, camera_params.tobytes(), 1)
        )

        # 3. Extract DISK features for all images
        features_cache: List[Dict] = []
        num_images = len(image_files)

        for idx, img_path in enumerate(image_files):
            img_id = idx + 1
            cur.execute(
                "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)",
                (img_id, img_path.name, 1)
            )

            # Extract DISK keypoints
            pct = 15 + int(20 * (idx + 1) / num_images)
            if progress_callback:
                progress_callback(pct, f"Extrayendo puntos neuronales DISK [{idx + 1}/{num_images}]...")

            feats = self.extract_frame_features(img_path)
            features_cache.append(feats)

            # Insert keypoints: shape (N, 2), float32
            kpts = feats["keypoints"]
            cur.execute(
                "INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (img_id, kpts.shape[0], 2, kpts.tobytes())
            )

        conn.commit()
        conn.close()

        # 4. Generate pairs and match with LightGlue
        pairs = self.generate_image_pairs([f.name for f in image_files])
        num_pairs = len(pairs)
        print(f"[NeuralMatcher] Matching {num_pairs} image pairs with LightGlue...")

        # Prepare match list text file for COLMAP's matches_importer
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f_match:
            match_file_path = Path(f_match.name)

            for p_idx, (i, j) in enumerate(pairs):
                if p_idx % max(1, num_pairs // 20) == 0:
                    pct = 35 + int(25 * (p_idx + 1) / num_pairs)
                    if progress_callback:
                        progress_callback(pct, f"Emparejamiento neuronal LightGlue [{p_idx + 1}/{num_pairs}]...")

                matches = self.match_pair(features_cache[i], features_cache[j])
                if matches.shape[0] >= self.config.min_inliers:
                    # Write block
                    name_i = image_files[i].name
                    name_j = image_files[j].name
                    f_match.write(f"{name_i} {name_j}\n")
                    for m in matches:
                        f_match.write(f"{m[0]} {m[1]}\n")
                    f_match.write("\n")

        # 5. Run COLMAP matches_importer to perform geometric verification (RANSAC)
        if progress_callback: progress_callback(60, "Verificando geometría epipolar 3D con COLMAP...")
        print("[NeuralMatcher] Importing matches and verifying two-view geometries...")

        cmd_import = [
            self.colmap_bin, "matches_importer",
            "--database_path", str(database_path),
            "--match_list_path", str(match_file_path),
            "--match_type", "raw"
        ]

        res = subprocess.run(cmd_import, capture_output=True, text=True)
        if match_file_path.exists():
            match_file_path.unlink()

        if res.returncode != 0:
            raise RuntimeError(f"COLMAP matches_importer failed:\n{res.stderr or res.stdout}")

        print("[NeuralMatcher] Neural matching & geometric verification completed successfully [OK]")
        if progress_callback: progress_callback(65, "Matching neuronal completado [OK] Construyendo mapa 3D...")

        return {
            "num_images": num_images,
            "num_pairs_matched": num_pairs,
            "database_path": database_path
        }
