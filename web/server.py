"""
FastAPI Backend Server for 3D Video Scanning MVP.
Handles video uploads, runs pipeline with real-time SSE progress,
serves 3D models, point clouds, calibration targets, and scan history.
"""
import asyncio
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from src.config import PipelineConfig, MarkerConfig, VideoIngestConfig, SfMConfig, MeshConfig, SliceConfig
from src.pipeline import VideoTo3DPipeline
from src.marker_generator import create_marker_pdf, create_marker_png

app = FastAPI(title="Video to 3D Metric Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "web" / "templates"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory storage for active processing sessions
active_sessions: Dict[str, Dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve main interactive dashboard."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html template not found.")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/download-marker-pdf")
async def get_marker_pdf(marker_id: int = 0, size_mm: float = 50.0):
    """Generate and download millimeter-scaled printable PDF target."""
    target_dir = DATA_DIR / "printable_markers"
    pdf_path = create_marker_pdf(target_dir / f"target_apriltag_id{marker_id}_{size_mm:.0f}mm.pdf", marker_id, size_mm)
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)


@app.get("/api/download-marker-png")
async def get_marker_png(marker_id: int = 0, size_mm: float = 50.0):
    """Generate and download printable PNG target."""
    target_dir = DATA_DIR / "printable_markers"
    png_path = create_marker_png(target_dir / f"target_apriltag_id{marker_id}_{size_mm:.0f}mm.png", marker_id, size_mm)
    return FileResponse(str(png_path), media_type="image/png", filename=png_path.name)



@app.get("/api/download-doc-tecnica-pdf")
async def get_doc_tecnica_pdf():
    """Download technical documentation & post-mortem engineering report PDF."""
    pdf_path = BASE_DIR / "DOCUMENTACION_TECNICA_Y_BITACORA.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Documentación técnica PDF no encontrada.")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_path.name)


def send_progress(session_id: str, stage: int, percent: int, message: str, data: dict = None):
    """Push a progress event to the session's queue."""
    if session_id in active_sessions:
        event = {
            "stage": stage,
            "percent": percent,
            "message": message,
        }
        if data:
            event["data"] = data
        active_sessions[session_id]["queue"].put(event)


def run_pipeline_with_progress(session_id: str, config: PipelineConfig):
    """Run the pipeline in a background thread, pushing progress events."""
    try:
        start_time = time.time()
        pipeline = VideoTo3DPipeline(config)

        # ── STAGE 1: Video Ingest ──
        send_progress(session_id, 1, 5, "Abriendo video y analizando metadata...")

        frames_dir = config.output_dir / "frames"
        manifest = pipeline.ingestor.process_video(config.video_path, frames_dir)
        total_extracted = manifest["total_extracted"]

        if total_extracted < 5:
            raise ValueError(f"Muy pocos frames extraídos ({total_extracted}). Grabe un video más largo o con mejor iluminación.")

        send_progress(session_id, 1, 100, f"✅ {total_extracted} frames extraídos",
                      {"frames": total_extracted})

        # ── STAGE 2: Structure from Motion ──
        send_progress(session_id, 2, 10, "Extrayendo features SIFT de cada frame...")

        sfm_dir = config.output_dir / "sfm"

        def sfm_progress(pct, msg):
            send_progress(session_id, 2, pct, msg)

        sfm_result = pipeline.sfm.run_reconstruction(frames_dir, sfm_dir, progress_callback=sfm_progress)

        if sfm_result["num_registered_images"] < 3:
            raise RuntimeError(f"COLMAP solo registró {sfm_result['num_registered_images']} cámaras. Reconstrucción insuficiente.")

        send_progress(session_id, 2, 100,
                      f"✅ {sfm_result['num_registered_images']} cámaras, {sfm_result['num_points3d']} puntos 3D",
                      {"cameras": sfm_result["num_registered_images"], "points": sfm_result["num_points3d"]})

        # ── STAGE 3: Metric Calibration ──
        send_progress(session_id, 3, 20, "Triangulando marcadores para escala métrica...")

        from src.metric_scaler import MetricScaler, MetricCalibrationResult
        calibration: MetricCalibrationResult = pipeline.scaler.calibrate_and_align(sfm_result, manifest)

        raw_xyz = sfm_result["xyz"]
        metric_xyz = MetricScaler.apply_transform_to_points(raw_xyz, calibration.transform_matrix)
        rgb_colors = sfm_result["rgb"]

        send_progress(session_id, 3, 100,
                      f"✅ Escala: {calibration.scale_factor:.4f} mm/u",
                      {"scale": calibration.scale_factor})

        # ── STAGE 4: Mesh Generation ──
        send_progress(session_id, 4, 10, "Filtrando outliers y estimando normales...")

        mesh_dir = config.output_dir / "mesh"
        send_progress(session_id, 4, 30, "Ejecutando reconstrucción Poisson Surface...")

        from src.mesh_generator import MeshGenerationResult
        mesh_result: MeshGenerationResult = pipeline.mesher.generate_mesh(
            points_xyz_metric=metric_xyz,
            colors_rgb=rgb_colors,
            output_dir=mesh_dir,
            base_name=config.video_path.stem
        )

        send_progress(session_id, 4, 100,
                      f"✅ Mesh: {mesh_result.num_vertices} vértices, {mesh_result.num_triangles} triángulos",
                      {"vertices": mesh_result.num_vertices, "triangles": mesh_result.num_triangles})

        # ── STAGE 5: Slicing & Measurements ──
        send_progress(session_id, 5, 20, "Cortando secciones transversales...")

        reports_dir = config.output_dir / "reports"
        report_json_path = reports_dir / "measurements.json"

        from src.mesh_analysis import MeshAnalyzer, MeshAnalysisReport
        analysis_report: MeshAnalysisReport = pipeline.analyzer.analyze_mesh(
            mesh_path=mesh_result.stl_path,
            output_json_path=report_json_path
        )

        send_progress(session_id, 5, 100,
                      f"✅ {analysis_report.num_slices} secciones medidas")

        total_duration = time.time() - start_time

        # Build summary
        summary = {
            "pipeline_status": "SUCCESS",
            "duration_seconds": round(total_duration, 2),
            "video_frames": total_extracted,
            "registered_cameras": sfm_result["num_registered_images"],
            "sparse_points_3d": sfm_result["num_points3d"],
            "scale_factor_applied": calibration.scale_factor,
            "calibration_error_mm": calibration.scale_error_mm,
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
            "paths": {
                "stl_model": str(mesh_result.stl_path),
                "obj_model": str(mesh_result.obj_path),
                "ply_model": str(mesh_result.ply_path),
                "measurements_json": str(report_json_path)
            }
        }

        # Save summary report to JSON
        summary_path = reports_dir / "pipeline_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Load measurements
        measurements_dict = {}
        if report_json_path.exists():
            with open(report_json_path, "r", encoding="utf-8") as f:
                measurements_dict = json.load(f)

        final_data = {
            "success": True,
            "scan_id": session_id,
            "total_time_sec": round(total_duration, 1),
            "video_frames_extracted": total_extracted,
            "num_registered_cameras": sfm_result["num_registered_images"],
            "num_sparse_points": sfm_result["num_points3d"],
            "scale_factor_mm": calibration.scale_factor,
            "stl_model_path": str(mesh_result.stl_path),
            "obj_model_path": str(mesh_result.obj_path),
            "ply_model_path": str(mesh_result.ply_path),
            "measurements": measurements_dict,
            "summary_report": summary
        }

        active_sessions[session_id]["result"] = final_data
        active_sessions[session_id]["status"] = "done"
        send_progress(session_id, 0, 100,
                      f"🎉 Pipeline completado en {total_duration:.1f}s",
                      final_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_data = {"success": False, "error": str(e)}
        active_sessions[session_id]["result"] = error_data
        active_sessions[session_id]["status"] = "error"
        send_progress(session_id, -1, 0, f"❌ Error: {str(e)}", error_data)


@app.post("/api/upload-video")
async def upload_video_endpoint(
    video: UploadFile = File(...),
    marker_size_mm: float = Form(50.0),
    target_fps: float = Form(4.0),
    min_laplacian_var: float = Form(30.0),
    poisson_depth: int = Form(9)
):
    """Upload video and start processing in background thread."""
    try:
        raw_name = Path(video.filename).stem if video.filename else 'session'
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)[:35].strip('_') or 'session'
        scan_id = f"scan_{int(time.time())}_{clean_name}_{os.urandom(3).hex()}"
        session_out_dir = OUTPUT_DIR / scan_id
        session_out_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(video.filename).suffix if (video.filename and Path(video.filename).suffix) else ".mp4"
        saved_video_path = session_out_dir / f"{clean_name}{ext}"
        with open(saved_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        config = PipelineConfig(
            video_path=saved_video_path,
            output_dir=session_out_dir,
            marker=MarkerConfig(marker_size_mm=marker_size_mm),
            ingest=VideoIngestConfig(target_fps=target_fps, min_laplacian_var=min_laplacian_var),
            sfm=SfMConfig(),
            mesh=MeshConfig(poisson_depth=poisson_depth),
            slice=SliceConfig(step_height_mm=10.0)
        )

        session_id = scan_id
        active_sessions[session_id] = {
            "queue": Queue(),
            "status": "running",
            "result": None
        }

        thread = threading.Thread(target=run_pipeline_with_progress, args=(session_id, config), daemon=True)
        thread.start()

        return JSONResponse(content={
            "session_id": session_id,
            "message": "Video subido. Conectando al stream de progreso..."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/progress/{session_id}")
async def progress_stream(session_id: str):
    """SSE stream for real-time progress events."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    async def event_generator():
        session = active_sessions[session_id]
        while True:
            try:
                event = session["queue"].get(timeout=0.5)
                yield {
                    "event": "progress",
                    "data": json.dumps(event, ensure_ascii=False)
                }
                if event.get("stage") in (0, -1):
                    break
            except Empty:
                yield {
                    "event": "heartbeat",
                    "data": json.dumps({"status": session["status"]})
                }
                if session["status"] in ("done", "error"):
                    break
            await asyncio.sleep(0.05)

        await asyncio.sleep(5)
        active_sessions.pop(session_id, None)

    return EventSourceResponse(event_generator())


@app.get("/api/projects")
async def list_projects():
    """
    Scans the output directory and returns a list of all historical scan projects.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    projects = []

    for item in OUTPUT_DIR.iterdir():
        if not item.is_dir():
            continue

        summary_file = item / "reports" / "pipeline_summary.json"
        measurements_file = item / "reports" / "measurements.json"
        stls = list(item.glob("**/*.stl"))

        created_ts = item.stat().st_mtime
        created_str = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(created_ts))

        # Check for summary
        summary_data = {}
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
            except Exception:
                pass

        stl_path = str(stls[0]) if stls else ""
        stl_filename = Path(stl_path).name if stl_path else ""
        stl_size_kb = round(Path(stl_path).stat().st_size / 1024, 1) if stl_path and Path(stl_path).exists() else 0

        project_info = {
            "scan_id": item.name,
            "created_at": created_str,
            "timestamp": created_ts,
            "has_stl": bool(stls),
            "stl_path": stl_path,
            "stl_filename": stl_filename,
            "stl_size_kb": stl_size_kb,
            "has_summary": bool(summary_data),
            "duration_sec": summary_data.get("duration_seconds", 0),
            "frames": summary_data.get("video_frames", 0),
            "cameras": summary_data.get("registered_cameras", 0),
            "points": summary_data.get("sparse_points_3d", 0),
            "triangles": summary_data.get("mesh_stats", {}).get("num_triangles", 0),
            "is_watertight": summary_data.get("mesh_stats", {}).get("is_watertight", False),
            "volume_cm3": summary_data.get("mesh_stats", {}).get("volume_cm3", 0.0),
            "dimensions_mm": summary_data.get("mesh_dimensions_mm", {}),
            "slices_count": summary_data.get("slices_extracted", 0)
        }
        projects.append(project_info)

    # Sort newest first
    projects.sort(key=lambda p: p["timestamp"], reverse=True)
    return JSONResponse(content={"success": True, "projects": projects})


@app.get("/api/project/{scan_id}")
async def get_project_details(scan_id: str):
    """
    Returns full metadata, measurements, and model paths for a specific past scan.
    """
    proj_dir = OUTPUT_DIR / scan_id
    if not proj_dir.exists():
        raise HTTPException(status_code=404, detail="Proyecto no encontrado.")

    summary_file = proj_dir / "reports" / "pipeline_summary.json"
    measurements_file = proj_dir / "reports" / "measurements.json"
    stls = list(proj_dir.glob("**/*.stl"))

    summary_data = {}
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

    measurements_data = {}
    if measurements_file.exists():
        with open(measurements_file, "r", encoding="utf-8") as f:
            measurements_data = json.load(f)

    stl_path = str(stls[0]) if stls else ""

    return JSONResponse(content={
        "success": True,
        "scan_id": scan_id,
        "stl_model_path": stl_path,
        "summary_report": summary_data,
        "measurements": measurements_data
    })


@app.delete("/api/project/{scan_id}")
async def delete_project(scan_id: str):
    """Deletes a historical scan project."""
    proj_dir = OUTPUT_DIR / scan_id
    if not proj_dir.exists():
        raise HTTPException(status_code=404, detail="Proyecto no encontrado.")

    try:
        shutil.rmtree(proj_dir)
        return JSONResponse(content={"success": True, "message": f"Proyecto {scan_id} eliminado."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/download-stl")
async def download_stl(path: str):
    """Serve reconstructed STL model by file path."""
    file_p = Path(path)
    # Also support relative paths inside output/
    if not file_p.is_absolute():
        file_p = BASE_DIR / path

    if not file_p.exists():
        raise HTTPException(status_code=404, detail=f"Modelo STL no encontrado en: {path}")
    return FileResponse(str(file_p), media_type="model/stl", filename=file_p.name)
