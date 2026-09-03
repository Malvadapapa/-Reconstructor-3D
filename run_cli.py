"""
CLI entry point for running the Video to 3D Metric Model pipeline from terminal.
Example usage:
  python run_cli.py --video data/input_videos/bottle_test.mp4 --marker-size 50.0 --output output/bottle_01
"""
import argparse
import sys
from pathlib import Path

from src.config import PipelineConfig, MarkerConfig, VideoIngestConfig, SfMConfig, MeshConfig, SliceConfig
from src.pipeline import VideoTo3DPipeline
from src.marker_generator import create_marker_pdf, create_marker_png


def parse_args():
    parser = argparse.ArgumentParser(description="Video to 3D Metric Model Pipeline (COLMAP + Open3D + AprilTag)")
    parser.add_argument("--video", type=str, required=False, help="Path to input video (.mp4, .mov)")
    parser.add_argument("--output", type=str, default="output/scan_result", help="Output directory for 3D models and reports")
    parser.add_argument("--marker-size", type=float, default=50.0, help="Physical marker side length in mm (default: 50.0)")
    parser.add_argument("--marker-id", type=int, default=0, help="AprilTag ID (default: 0)")
    parser.add_argument("--fps", type=float, default=3.0, help="Target frame extraction FPS (default: 3.0)")
    parser.add_argument("--min-laplacian", type=float, default=80.0, help="Laplacian variance blur threshold (default: 80.0)")
    parser.add_argument("--poisson-depth", type=int, default=9, help="Poisson tree depth (default: 9)")
    parser.add_argument("--slice-step", type=float, default=10.0, help="Slicing interval in mm (default: 10.0)")
    parser.add_argument("--generate-markers", action="store_true", help="Generate printable AprilTag calibration targets (PDF & PNG)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.generate_markers:
        print("[CLI] Generating printable calibration targets...")
        target_dir = Path("data/printable_markers")
        pdf = create_marker_pdf(target_dir / f"target_apriltag_id{args.marker_id}_{args.marker_size:.0f}mm.pdf", args.marker_id, args.marker_size)
        png = create_marker_png(target_dir / f"target_apriltag_id{args.marker_id}_{args.marker_size:.0f}mm.png", args.marker_id, args.marker_size)
        print(f"[OK] Generated PDF: {pdf}")
        print(f"[OK] Generated PNG: {png}")
        if not args.video:
            return

    if not args.video:
        print("Error: --video argument is required (or use --generate-markers to create printable target).")
        sys.exit(1)

    video_p = Path(args.video)
    if not video_p.exists():
        print(f"Error: Video file not found: {video_p}")
        sys.exit(1)

    config = PipelineConfig(
        video_path=video_p,
        output_dir=Path(args.output),
        marker=MarkerConfig(marker_id=args.marker_id, marker_size_mm=args.marker_size),
        ingest=VideoIngestConfig(target_fps=args.fps, min_laplacian_var=args.min_laplacian),
        sfm=SfMConfig(),
        mesh=MeshConfig(poisson_depth=args.poisson_depth),
        slice=SliceConfig(step_height_mm=args.slice_step)
    )

    pipeline = VideoTo3DPipeline(config)
    result = pipeline.run()

    print(f"\n[OK] STL exported to: {result.stl_model_path}")
    print(f"[OK] Measurements saved to: {result.measurements_json_path}")


if __name__ == "__main__":
    main()
