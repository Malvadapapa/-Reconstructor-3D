"""
Unified Matcher Interface for Structure-from-Motion.
Defines BaseMatcher and MatchingResult contracts to allow swapping
SIFT and DISK + LightGlue without altering SfM mapper or camera configuration.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np


@dataclass
class MatchingResult:
    """Standardized output produced by any Matcher for the SfM reconstruction stage."""
    engine_name: str                      # "SIFT" or "DISK + LightGlue"
    num_images: int                       # Total images processed
    total_pairs_processed: int            # Total candidate pairs evaluated
    valid_pairs: int                      # Pairs with sufficient geometric inliers
    inliers_mean: float                   # Mean inliers per valid pair
    inliers_median: float                 # Median inliers per valid pair
    inliers_p10: float                    # 10th percentile inliers
    inliers_p90: float                    # 90th percentile inliers
    database_path: Path                   # Path to prepared COLMAP SQLite database
    metadata: Dict                        # Additional engine-specific stats


class BaseMatcher(ABC):
    """Abstract Base Class for feature extraction and two-view matching engines."""

    @abstractmethod
    def extract_and_match(
        self,
        image_dir: Path,
        database_path: Path,
        camera_config,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> MatchingResult:
        """
        Extract features from images in image_dir, perform pairwise matching and geometric
        verification, writing cameras, images, keypoints, and two_view_geometries to database_path.
        """
        pass
