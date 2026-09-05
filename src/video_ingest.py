"""
Video Ingestion and Adaptive Keyframe Selection Module.
Extracts uniform, crystal-sharp keyframes spanning 100% of the video duration.
Within each temporal window, it automatically selects the sharpest frame (highest Laplacian variance)
to eliminate motion blur and ensure a complete 360° multiview reconstruction.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import json
import numpy as np
from tqdm import tqdm

from src.config import VideoIngestConfig, MarkerConfig
from src.marker_detector import MarkerDetector


class VideoIngestor:
    """Manages video frame extraction, quality scoring, and fiducial marker detection."""

    def __init__(self, config: VideoIngestConfig, marker_config: MarkerConfig):
        self.config = config
        self.marker_config = marker_config
        self.detector = MarkerDetector(family=marker_config.family)

    @staticmethod
    def compute_sharpness(image_bgr: np.ndarray) -> float:
        """
        Compute image sharpness using the variance of the Laplacian filter.
        Higher values indicate sharper edges and less motion blur.
        """
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def process_video(
        self,
        video_path: Path,
        output_frames_dir: Path,
        allow_reuse: bool = True
    ) -> Dict:
        """
        Processes video file in a single fast pass:
        1. Divides the full video duration into N uniform temporal windows (up to max_frames).
        2. In each window, selects the frame with highest sharpness variance.
        3. Detects AprilTags and saves high-resolution keyframes + manifest.json.
        """
        video_path = Path(video_path)
        output_frames_dir = Path(output_frames_dir)
        output_frames_dir.mkdir(parents=True, exist_ok=True)

        # Check if frames already exist and can be reused
        manifest_path = output_frames_dir.parent / "frames_manifest.json"
        existing_frames = list(output_frames_dir.glob("frame_*.jpg"))
        if allow_reuse and manifest_path.exists() and len(existing_frames) >= 5:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f_m:
                    manifest = json.load(f_m)
                if manifest.get("total_extracted", 0) == len(existing_frames):
                    print(f"[VideoIngestor] Reusing {len(existing_frames)} existing frames from {output_frames_dir}")
                    return manifest
            except Exception as e:
                print(f"[VideoIngestor] Warning reading manifest: {e}")

        # Clear existing frames
        for f in output_frames_dir.glob("frame_*.jpg"):
            f.unlink(missing_ok=True)
        # Ensure any legacy annotated subfolder inside frames is removed
        legacy_annotated = output_frames_dir / "annotated"
        if legacy_annotated.exists():
            import shutil
            shutil.rmtree(legacy_annotated, ignore_errors=True)

        # Keep annotated debug frames completely isolated outside of frames/
        debug_dir = output_frames_dir.parent / "annotated_debug"
        if debug_dir.exists():
            for f in debug_dir.glob("*.jpg"):
                f.unlink(missing_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        duration_sec = total_frames / fps if total_frames > 0 else 0.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Target number of keyframes across entire 360 orbit (e.g. 50-60 frames)
        target_frames = min(self.config.max_frames, max(25, int(duration_sec * self.config.target_fps)))
        target_frames = min(target_frames, max(1, total_frames))
        window_size = total_frames / float(target_frames)

        print(f"[VideoIngest] Video: {video_path.name} | {orig_w}x{orig_h} | {fps:.1f} FPS | {duration_sec:.1f}s | Target: {target_frames} keyframes spanning 100% orbit")

        # First pass: collect best frame per window
        current_window = 0
        best_in_window: Optional[Tuple[float, np.ndarray, int]] = None
        selected_raw_frames: List[Tuple[float, np.ndarray, int]] = []

        pbar = tqdm(total=total_frames, desc="Analyzing Keyframes")
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if self.config.resize_max_dim and max(h, w) > self.config.resize_max_dim:
                scale = self.config.resize_max_dim / float(max(h, w))
                new_w, new_h = int(round(w * scale)), int(round(h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            sharpness = self.compute_sharpness(frame)

            win = int(f_idx / window_size)
            if win > current_window:
                if best_in_window is not None:
                    selected_raw_frames.append(best_in_window)
                current_window = win
                best_in_window = (sharpness, frame.copy(), f_idx)
            else:
                if best_in_window is None or sharpness > best_in_window[0]:
                    best_in_window = (sharpness, frame.copy(), f_idx)

            f_idx += 1
            pbar.update(1)

        if best_in_window is not None:
            selected_raw_frames.append(best_in_window)

        pbar.close()
        cap.release()

        # Save selected sharp keyframes and run marker detection
        extracted_frames: List[Dict] = []
        for saved_idx, (sharpness, frame, original_f_idx) in enumerate(selected_raw_frames):
            filename = f"frame_{saved_idx:04d}.jpg"
            file_path = output_frames_dir / filename
            cv2.imwrite(str(file_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # Detect markers
            detections = self.detector.detect(frame)
            has_any_marker = len(detections) > 0

            if detections:
                annotated = self.detector.draw_detections(frame, detections)
                cv2.imwrite(str(debug_dir / filename), annotated)

            det_dicts = [
                {
                    "marker_id": d.marker_id,
                    "corners_2d": d.corners.tolist(),
                    "center_2d": d.center.tolist()
                }
                for d in detections
            ]

            extracted_frames.append({
                "frame_index": original_f_idx,
                "saved_index": saved_idx,
                "file_name": filename,
                "file_path": str(file_path),
                "timestamp_sec": round(float(original_f_idx / fps), 3),
                "sharpness_var": round(sharpness, 2),
                "marker_detected": has_any_marker,
                "detections": det_dicts
            })

        print(f"[VideoIngest] Extracted {len(extracted_frames)} sharp keyframes uniformly across {duration_sec:.1f}s to: {output_frames_dir}")

        manifest = {
            "video_file": str(video_path),
            "video_metadata": {
                "width": orig_w,
                "height": orig_h,
                "fps": fps,
                "total_frames": total_frames,
                "duration_sec": duration_sec
            },
            "ingest_settings": {
                "target_fps": self.config.target_fps,
                "min_laplacian_var": self.config.min_laplacian_var,
                "max_frames": self.config.max_frames
            },
            "total_extracted": len(extracted_frames),
            "frames": extracted_frames
        }

        # Save manifest in parent directory so frames folder only contains valid images for COLMAP
        manifest_path = output_frames_dir.parent / "frames_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
