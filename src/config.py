"""
Configuration and settings dataclasses for the Video-to-3D Metric Model Pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class MarkerConfig:
    """Settings for fiducial markers (AprilTag / ArUco)."""
    family: str = "tag36h11"          # AprilTag family: tag36h11, tag25h9, tag16h5
    marker_id: int = 0                # Primary target marker ID
    marker_size_mm: float = 50.0      # Physical size of the outer black square in mm
    use_aruco_fallback: bool = True   # Fallback to cv2.aruco if pupil-apriltags is unavailable
    dictionary_name: str = "DICT_APRILTAG_36h11"


@dataclass
class VideoIngestConfig:
    """Settings for frame extraction and quality filtering."""
    target_fps: float = 4.0           # Frames per second to extract from video
    min_laplacian_var: float = 30.0   # Minimum Laplacian variance (blur detection threshold)
    max_frames: int = 60              # Maximum keyframes to retain (optimal for SfM performance)
    require_marker: bool = False      # Whether frames without marker are discarded
    resize_max_dim: Optional[int] = 1920 # Resize longest edge to speed up SfM if > 1920


@dataclass
class SfMConfig:
    """Settings for Structure-from-Motion (COLMAP)."""
    camera_model: str = "SIMPLE_RADIAL"  # SIMPLE_RADIAL (recommended for smartphones) or RADIAL
    matcher_type: str = "sequential"     # sequential (fast for video) or exhaustive
    max_features: int = 8192             # SIFT max features
    colmap_binary_path: Optional[str] = None # Path to colmap.exe, auto-detected if None
    use_gpu: bool = False                # Set to True if NVIDIA CUDA build is installed


@dataclass
class MeshConfig:
    """Settings for Poisson surface reconstruction and mesh cleanup."""
    poisson_depth: int = 9            # Poisson tree depth (8-10 recommended)
    density_trim_percentile: float = 5.0 # Trim lower 5% Poisson density vertices (noise)
    filter_statistical_outliers: bool = True
    nb_neighbors: int = 30
    std_ratio: float = 2.0
    smooth_iterations: int = 5
    crop_to_roi: bool = True          # Crop mesh around marker base
    roi_radius_mm: float = 120.0      # Radius around marker center (mm)
    roi_height_mm: float = 350.0      # Max height above base (mm)


@dataclass
class SliceConfig:
    """Settings for geometric cross-section analysis and slicing."""
    step_height_mm: float = 10.0      # Slicing interval in mm
    min_height_mm: float = 5.0        # Start height above base plane
    max_height_mm: Optional[float] = None # End height (auto-calculated if None)


@dataclass
class NeuralMatcherConfig:
    """Settings for neural feature extraction and matching (LightGlue + DISK)."""
    enabled: bool = False             # Set to True to use DISK + LightGlue instead of SIFT
    device: str = "cpu"               # "cpu" (ideal without NVIDIA GPU), "cuda", or "auto"
    filter_threshold: float = 0.1     # LightGlue confidence pruning threshold
    min_inliers: int = 15             # Minimum inlier matches to register an image pair
    max_frames_exhaustive: int = 150  # Up to 150 frames uses exhaustive all-to-all matching


@dataclass
class TextureConfig:
    """Settings for UV parametrization and real multi-view texture baking."""
    enabled: bool = False             # Set to True to perform UV unwrapping and texture baking
    atlas_resolution: int = 4096      # Dimension of square texture atlas image (e.g. 2048 or 4096 px)
    seam_blending: bool = True        # Smooth transition between adjacent UV charts
    export_glb: bool = True           # Export interactive GLB format for Three.js viewer


@dataclass
class PipelineConfig:
    """Master configuration holding all sub-configs and paths."""
    video_path: Path = field(default_factory=lambda: Path("data/input_videos/test_bottle.mp4"))
    output_dir: Path = field(default_factory=lambda: Path("output/scan_001"))
    
    marker: MarkerConfig = field(default_factory=MarkerConfig)
    ingest: VideoIngestConfig = field(default_factory=VideoIngestConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    slice: SliceConfig = field(default_factory=SliceConfig)
    neural: NeuralMatcherConfig = field(default_factory=NeuralMatcherConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)

    def ensure_dirs(self) -> None:
        """Create necessary output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "frames").mkdir(exist_ok=True)
        (self.output_dir / "sfm").mkdir(exist_ok=True)
        (self.output_dir / "mesh").mkdir(exist_ok=True)
        (self.output_dir / "texture").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
