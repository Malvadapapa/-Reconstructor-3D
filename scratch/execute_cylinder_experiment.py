"""
Execution script: Cylinder vs Cup Comparison Experiment
======================================================
Compare controlled cylinder reference geometry against the real cup
using the identical DISK + LightGlue + COLMAP pipeline.
"""
import sys
sys.path.insert(0, ".")
from pathlib import Path
import struct
import json
import sqlite3
import subprocess
import tempfile
import time
import pickle
import shutil
import cv2
import numpy as np

from src.sfm_reconstruction import ColmapModelParser, SfMReconstructor
from src.neural_matcher import NeuralMatcher
from src.metric_scaler import MetricScaler
from src.marker_detector import MarkerDetector
from src.config import NeuralMatcherConfig, SfMConfig, MarkerConfig

# Paths
exp_base = Path("scratch/cylinder_vs_cup_experiment")
exp_base.mkdir(parents=True, exist_ok=True)
cyl_dir = exp_base / "cylinder"
cyl_frames_dir = cyl_dir / "frames"
cyl_frames_dir.mkdir(parents=True, exist_ok=True)
cyl_sfm_dir = cyl_dir / "sfm"
cyl_sparse_dir = cyl_sfm_dir / "sparse"
cyl_sfm_dir.mkdir(parents=True, exist_ok=True)
cyl_sparse_dir.mkdir(parents=True, exist_ok=True)

# Ground truth constants
GROUND_TRUTH = {
    "cylinder": {
        "description": "Cilindro de referencia geométrico perfecto",
        "real_diameter_mm": 70.00,
        "real_radius_mm": 35.00,
        "real_height_mm": 120.00,
        "nominal_slope_b": 0.0000,
        "nominal_cyl_ratio": 1.0000,
    },
    "cup": {
        "description": "Taza real de cerámica con calibración en tablero",
        "real_mouth_diameter_mm": 82.50,
        "approx_height_mm": 105.00,
    }
}

print("="*75)
print("  EXPERIMENTO: COMPARATIVA CILINDRO DE REFERENCIA VS TAZA")
print("  Pipeline: DISK + LightGlue + COLMAP (SIMPLE_RADIAL)")
print("="*75)
print("\n[GROUND TRUTH DISPONIBLE]")
print(f"  Objeto A - Cilindro de referencia:")
print(f"    * Diámetro real: {GROUND_TRUTH['cylinder']['real_diameter_mm']:.2f} mm (Radio = {GROUND_TRUTH['cylinder']['real_radius_mm']:.2f} mm)")
print(f"    * Altura real:   {GROUND_TRUTH['cylinder']['real_height_mm']:.2f} mm")
print(f"    * Pendiente nominal b: {GROUND_TRUTH['cylinder']['nominal_slope_b']:.4f} (paredes perfectamente verticales)")
print(f"    * CylRatio nominal:    {GROUND_TRUTH['cylinder']['nominal_cyl_ratio']:.4f} (sin conicidad)")
print(f"\n  Objeto B - Taza:")
print(f"    * Diámetro real de boca: {GROUND_TRUTH['cup']['real_mouth_diameter_mm']:.2f} mm")
print(f"    * Altura aproximada:     {GROUND_TRUTH['cup']['approx_height_mm']:.2f} mm")
print("="*75)

# Camera parameters matching the cup video
IMG_W, IMG_H = 474, 850
FOCAL_PX = 1020.0  # 1.2 * max(W, H)
CX, CY = IMG_W / 2.0, IMG_H / 2.0
NUM_VIEWS = 60

# ══════════════════════════════════════════════════════════════════════
#  FASE 1: RENDERIZADO DEL CILINDRO DE REFERENCIA CON TABLERO APRILTAG
# ══════════════════════════════════════════════════════════════════════
print("\n[FASE 1] Renderizado de 60 frames del cilindro con textura y marcadores...")

# Pre-generate AprilTag markers (tag36h11)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
TAG_SIZE_PX = 160
tag_images = {}
for tid in range(5):
    raw_tag = cv2.aruco.generateImageMarker(aruco_dict, tid, TAG_SIZE_PX)
    pad = cv2.copyMakeBorder(raw_tag, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    tag_images[tid] = cv2.cvtColor(pad, cv2.COLOR_GRAY2BGR)

# Rich procedural texture for the cylinder (high-frequency detail for DISK)
W_tex, H_tex = 1024, 512
rng = np.random.RandomState(42)
base_noise = rng.randint(40, 210, (H_tex // 8, W_tex // 8, 3), dtype=np.uint8)
tex = cv2.resize(base_noise, (W_tex, H_tex), interpolation=cv2.INTER_CUBIC)
for y in range(0, H_tex, 25):
    cv2.line(tex, (0, y), (W_tex, y), (20, 20, 20), 1)
for x in range(0, W_tex, 32):
    cv2.putText(tex, f"TAG-{x}", (x, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
    cv2.putText(tex, f"CYL-{x}", (x, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)

R_cyl = 35.0
H_cyl = 120.0
TAG_SIZE_MM = 25.0                                  # Active AprilTag code dimension (black boundary)
BORDER_PX = 20                                      # Padding border in pixels
BORDER_MM = (BORDER_PX / TAG_SIZE_PX) * TAG_SIZE_MM # Physical padding width: (20 / 160) * 25 = 3.125 mm
TOTAL_PATCH_MM = TAG_SIZE_MM + 2.0 * BORDER_MM      # Total footprint with margin: 31.25 mm
HALF_PATCH = TOTAL_PATCH_MM / 2.0                   # 15.625 mm

TAG_CENTERS = {
    0: np.array([0.0, 0.0]),
    1: np.array([0.0, 72.0]),
    2: np.array([72.0, 0.0]),
    3: np.array([0.0, -72.0]),
    4: np.array([-72.0, 0.0])
}

def render_cylinder_frame(angle_deg):
    rad = np.radians(angle_deg)
    cam_pos = np.array([250.0 * np.cos(rad), 250.0 * np.sin(rad), 85.0])
    target = np.array([0.0, 0.0, 55.0])
    fwd = (target - cam_pos) / np.linalg.norm(target - cam_pos)
    up_w = np.array([0.0, 0.0, 1.0])
    rt = np.cross(fwd, up_w) / np.linalg.norm(np.cross(fwd, up_w))
    up = np.cross(rt, fwd)
    R = np.array([rt, -up, fwd])
    
    grid_x, grid_y = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
    ray_cam = np.stack([(grid_x - CX) / FOCAL_PX, (grid_y - CY) / FOCAL_PX, np.ones_like(grid_x)], axis=-1)
    ray_w = ray_cam @ R
    D = ray_w / np.linalg.norm(ray_w, axis=-1, keepdims=True)
    C = cam_pos
    
    # Cylinder intersection: (Cx + s*Dx)^2 + (Cy + s*Dy)^2 = R^2
    A_quad = D[..., 0]**2 + D[..., 1]**2
    B_quad = C[0] * D[..., 0] + C[1] * D[..., 1]
    C_quad = C[0]**2 + C[1]**2 - R_cyl**2
    disc = B_quad**2 - A_quad * C_quad
    
    hit_mask = disc > 0
    s_cyl = np.full_like(disc, 1e9)
    s_cyl[hit_mask] = (-B_quad[hit_mask] - np.sqrt(disc[hit_mask])) / A_quad[hit_mask]
    
    P_z = C[2] + s_cyl * D[..., 2]
    valid_cyl = hit_mask & (s_cyl > 0) & (P_z >= 0) & (P_z <= H_cyl)
    
    # Ground plane intersection (Z = 0)
    Dz = D[..., 2]
    s_plane = np.full_like(Dz, 1e9)
    valid_plane_dz = Dz < -1e-4
    s_plane[valid_plane_dz] = -C[2] / Dz[valid_plane_dz]
    
    X_plane = C[0] + s_plane * D[..., 0]
    Y_plane = C[1] + s_plane * D[..., 1]
    
    # Board is [-105, 105] x [-105, 105] mm
    on_board = (
        valid_plane_dz &
        (s_plane > 0) &
        (np.abs(X_plane) <= 105.0) &
        (np.abs(Y_plane) <= 105.0) &
        (~valid_cyl | (s_plane < s_cyl))
    )
    
    img = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * 220
    img[on_board] = [245, 245, 245]
    
    # Draw AprilTags on the board (active marker is exactly 25.0 mm surrounded by white padding)
    for tid, center in TAG_CENTERS.items():
        in_tag = (
            on_board &
            (np.abs(X_plane - center[0]) <= HALF_PATCH) &
            (np.abs(Y_plane - center[1]) <= HALF_PATCH)
        )
        if np.any(in_tag):
            pad_size = TAG_SIZE_PX + 2 * BORDER_PX
            u_tag = (((X_plane[in_tag] - (center[0] - HALF_PATCH)) / TOTAL_PATCH_MM) * pad_size).astype(int)
            v_tag = ((((center[1] + HALF_PATCH) - Y_plane[in_tag]) / TOTAL_PATCH_MM) * pad_size).astype(int)
            u_tag = np.clip(u_tag, 0, pad_size - 1)
            v_tag = np.clip(v_tag, 0, pad_size - 1)
            img[in_tag] = tag_images[tid][v_tag, u_tag]
            
    # Draw cylinder
    P_x = C[0] + s_cyl * D[..., 0]
    P_y = C[1] + s_cyl * D[..., 1]
    
    theta = np.arctan2(P_y[valid_cyl], P_x[valid_cyl]) % (2 * np.pi)
    u = ((theta / (2 * np.pi)) * W_tex).astype(int) % W_tex
    v = ((P_z[valid_cyl] / H_cyl) * H_tex).astype(int) % H_tex
    
    norm_x = P_x[valid_cyl] / R_cyl
    norm_y = P_y[valid_cyl] / R_cyl
    light_dir = -fwd
    shade = np.clip(norm_x * light_dir[0] + norm_y * light_dir[1], 0.35, 1.0)
    
    img[valid_cyl] = (tex[v, u].astype(float) * shade[:, None]).astype(np.uint8)
    
    # Add subtle realistic sensor noise
    noise = np.random.normal(0, 1.5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

detector = MarkerDetector(family="tag36h11")
cyl_frames_info = []

for i in range(NUM_VIEWS):
    ang = 360.0 * i / NUM_VIEWS
    frame_name = f"frame_{i:04d}.jpg"
    frame_path = cyl_frames_dir / frame_name
    
    if not frame_path.exists():
        im = render_cylinder_frame(ang)
        cv2.imwrite(str(frame_path), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        im = cv2.imread(str(frame_path))
    
    sharpness = float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    dets = detector.detect(im)
    det_dicts = [
        {"marker_id": d.marker_id, "corners_2d": d.corners.tolist(), "center_2d": d.center.tolist()}
        for d in dets
    ]
    
    cyl_frames_info.append({
        "frame_index": i,
        "saved_index": i,
        "file_name": frame_name,
        "file_path": str(frame_path),
        "timestamp_sec": round(float(i / 30.0), 3),
        "sharpness_var": round(sharpness, 2),
        "marker_detected": len(dets) > 0,
        "detections": det_dicts
    })
    if (i + 1) % 15 == 0:
        print(f"  Rendered [{i+1}/{NUM_VIEWS}] frames...")

cyl_manifest = {
    "video_file": "synthetic_cylinder_reference.mp4",
    "video_metadata": {"width": IMG_W, "height": IMG_H, "fps": 30.0, "total_frames": NUM_VIEWS, "duration_sec": 2.0},
    "ingest_settings": {"method": "controlled_reference_render", "target_frames": NUM_VIEWS},
    "total_extracted": len(cyl_frames_info),
    "frames": cyl_frames_info
}
with open(cyl_dir / "frames_manifest.json", "w") as f:
    json.dump(cyl_manifest, f, indent=2)

tag_counts = {}
for fr in cyl_frames_info:
    for d in fr.get("detections", []):
        tag_counts[d["marker_id"]] = tag_counts.get(d["marker_id"], 0) + 1
print(f"  Frames listos: {len(cyl_frames_info)} frames.")
print(f"  Detecciones por Tag AprilTag: {tag_counts}")

# ══════════════════════════════════════════════════════════════════════
#  FASE 2: DISK + LIGHTGLUE EN EL CILINDRO
# ══════════════════════════════════════════════════════════════════════
print("\n[FASE 2] Extracción DISK y Matching LightGlue para el Cilindro...")
sfm_cfg = SfMConfig()
sfm_recon = SfMReconstructor(sfm_cfg)
colmap_bin = sfm_recon.colmap_bin

neural_cfg = NeuralMatcherConfig(enabled=True, device="cpu", filter_threshold=0.1, min_inliers=15)
matcher = NeuralMatcher(neural_cfg, colmap_bin=colmap_bin)

image_files = sorted([cyl_frames_dir / f"frame_{i:04d}.jpg" for i in range(NUM_VIEWS)])
feat_cache_path = cyl_sfm_dir / "disk_features.pkl"

if feat_cache_path.exists():
    with open(feat_cache_path, "rb") as f:
        cyl_features = pickle.load(f)
    print(f"  Cargados {len(cyl_features)} DISK features en caché.")
else:
    print("  Extrayendo DISK features de 60 frames...")
    cyl_features = {}
    for idx, imp in enumerate(image_files):
        cyl_features[imp.name] = matcher.extract_frame_features(imp)
        if (idx + 1) % 15 == 0:
            print(f"    DISK [{idx + 1}/{NUM_VIEWS}]...")
    with open(feat_cache_path, "wb") as f:
        pickle.dump(cyl_features, f)
    print("  DISK completado y guardado en caché.")

pairs = matcher.generate_image_pairs([f.name for f in image_files])
print(f"  Generados {len(pairs)} pares para matching (sequential + loop closure).")

match_cache_path = cyl_sfm_dir / "lightglue_matches.pkl"
if match_cache_path.exists():
    with open(match_cache_path, "rb") as f:
        cyl_matches = pickle.load(f)
    print(f"  Cargados {len(cyl_matches)} matches LightGlue en caché.")
else:
    print(f"  Ejecutando LightGlue en {len(pairs)} pares...")
    cyl_matches = {}
    feats_list = [cyl_features[f.name] for f in image_files]
    t_match0 = time.time()
    for p_idx, (i, j) in enumerate(pairs):
        name_i = image_files[i].name
        name_j = image_files[j].name
        m = matcher.match_pair(feats_list[i], feats_list[j])
        cyl_matches[(name_i, name_j)] = m
        if (p_idx + 1) % 50 == 0:
            print(f"    LightGlue [{p_idx + 1}/{len(pairs)}] pares...")
    with open(match_cache_path, "wb") as f:
        pickle.dump(cyl_matches, f)
    print(f"  Matching LightGlue completado en {time.time()-t_match0:.1f}s.")

# Matching statistics
feat_counts = [cyl_features[f.name]["keypoints"].shape[0] for f in image_files]
m_counts = [cyl_matches[k].shape[0] for k in cyl_matches]
valid_m = [c for c in m_counts if c >= 15]

cyl_matching_stats = {
    "features_per_image_mean": float(np.mean(feat_counts)),
    "features_per_image_median": float(np.median(feat_counts)),
    "total_pairs": len(pairs),
    "valid_pairs": len(valid_m),
    "inliers_mean": float(np.mean(valid_m)) if valid_m else 0,
    "inliers_median": float(np.median(valid_m)) if valid_m else 0,
    "inliers_p10": float(np.percentile(valid_m, 10)) if valid_m else 0,
    "inliers_p90": float(np.percentile(valid_m, 90)) if valid_m else 0,
}
print(f"  Estadísticas matching cilindro: Pares válidos={len(valid_m)}/{len(pairs)}, Inliers/par mean={cyl_matching_stats['inliers_mean']:.1f}")

# ══════════════════════════════════════════════════════════════════════
#  FASE 3: COLMAP MAPPER PARA EL CILINDRO
# ══════════════════════════════════════════════════════════════════════
print("\n[FASE 3] Construcción de base de datos y reconstrucción SfM (COLMAP)...")
db_path = cyl_sfm_dir / "database.db"
if db_path.exists():
    db_path.unlink()

conn = sqlite3.connect(str(db_path))
cur = conn.cursor()
cur.executescript("""
    CREATE TABLE IF NOT EXISTS cameras (
        camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
        params BLOB, prior_focal_length INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
        prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
        prior_tx REAL, prior_ty REAL, prior_tz REAL,
        CONSTRAINT image_id_check CHECK(image_id >= 0)
    );
    CREATE TABLE IF NOT EXISTS keypoints (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
    );
    CREATE TABLE IF NOT EXISTS descriptors (
        image_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
    );
    CREATE TABLE IF NOT EXISTS matches (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
    );
    CREATE TABLE IF NOT EXISTS two_view_geometries (
        pair_id INTEGER PRIMARY KEY NOT NULL,
        rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB,
        config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB
    );
""")

camera_params = np.array([FOCAL_PX, CX, CY, 0.0], dtype=np.float64)
cur.execute(
    "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
    (1, 2, IMG_W, IMG_H, camera_params.tobytes(), 1)
)

for idx, img_p in enumerate(image_files):
    img_id = idx + 1
    cur.execute("INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)", (img_id, img_p.name, 1))
    kpts = cyl_features[img_p.name]["keypoints"]
    cur.execute("INSERT INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (img_id, kpts.shape[0], 2, kpts.tobytes()))

conn.commit()
conn.close()

# Import matches
valid_items = [
    (image_files[i].name, image_files[j].name, cyl_matches[(image_files[i].name, image_files[j].name)])
    for i, j in pairs
    if (image_files[i].name, image_files[j].name) in cyl_matches
    and cyl_matches[(image_files[i].name, image_files[j].name)].shape[0] >= 15
]

batch_size = 30
for b_idx in range(0, len(valid_items), batch_size):
    batch = valid_items[b_idx:b_idx + batch_size]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f_m:
        t_path = Path(f_m.name)
        for name_i, name_j, m in batch:
            f_m.write(f"{name_i} {name_j}\n")
            for match in m:
                f_m.write(f"{match[0]} {match[1]}\n")
            f_m.write("\n")
    
    cmd_import = [
        colmap_bin, "matches_importer",
        "--database_path", str(db_path),
        "--match_list_path", str(t_path),
        "--match_type", "raw",
        "--SiftMatching.use_gpu", "0"
    ]
    res = subprocess.run(cmd_import, capture_output=True, text=True)
    t_path.unlink()
    if res.returncode != 0:
        raise RuntimeError(f"matches_importer error: {res.stderr}")

for d in cyl_sparse_dir.iterdir():
    if d.is_dir():
        shutil.rmtree(d)

cmd_map = [
    colmap_bin, "mapper",
    "--database_path", str(db_path),
    "--image_path", str(cyl_frames_dir),
    "--output_path", str(cyl_sparse_dir),
    "--Mapper.min_num_matches", "10",
    "--Mapper.init_min_num_inliers", "15",
    "--Mapper.abs_pose_min_num_inliers", "15",
    "--Mapper.init_min_tri_angle", "4.0",
    "--Mapper.ba_refine_extra_params", "0"
]
print("  Ejecutando COLMAP mapper...")
t0_map = time.time()
res_map = subprocess.run(cmd_map, capture_output=True, text=True)
print(f"  COLMAP mapper finalizado en {time.time()-t0_map:.1f}s.")

# Check reconstruction subdirs
subdirs = [d for d in cyl_sparse_dir.iterdir() if d.is_dir()]
if not subdirs:
    raise RuntimeError("COLMAP mapper no generó ninguna reconstrucción para el cilindro!")
primary_cand = max(subdirs, key=lambda d: len(ColmapModelParser.read_images_binary(d / "images.bin")) if (d / "images.bin").exists() else 0)

cyl_cameras = ColmapModelParser.read_cameras_binary(primary_cand / "cameras.bin")
cyl_images = ColmapModelParser.read_images_binary(primary_cand / "images.bin")

def read_points3D_full(path):
    points = {}
    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", fid.read(8))[0]
            xyz = struct.unpack("<3d", fid.read(24))
            rgb = struct.unpack("<3B", fid.read(3))
            error = struct.unpack("<d", fid.read(8))[0]
            track_length = struct.unpack("<Q", fid.read(8))[0]
            track_data = struct.unpack(f"<{2*track_length}i", fid.read(8 * track_length))
            image_ids = track_data[0::2]
            point2D_idxs = track_data[1::2]
            points[point3D_id] = {
                "id": point3D_id, "xyz": np.array(xyz), "rgb": np.array(rgb),
                "error": error, "track_length": track_length,
                "image_ids": list(image_ids), "point2D_idxs": list(point2D_idxs)
            }
    return points

cyl_points3d = read_points3D_full(primary_cand / "points3D.bin")
print(f"  Reconstrucción cilindro: {len(cyl_images)}/60 cámaras registradas, {len(cyl_points3d)} puntos 3D.")

# ══════════════════════════════════════════════════════════════════════
#  FASE 4: CALIBRACIÓN MÉTRICA Y ANÁLISIS GEOMÉTRICO
# ══════════════════════════════════════════════════════════════════════
print("\n[FASE 4] Calibración métrica (MetricScaler) y análisis geométrico...")

scaler = MetricScaler(MarkerConfig(marker_size_mm=50.0))
cyl_sfm_dict = {
    "xyz": np.array([p["xyz"] for p in cyl_points3d.values()]),
    "rgb": np.array([p["rgb"] for p in cyl_points3d.values()]),
    "cameras": cyl_cameras,
    "images": cyl_images,
    "points3d": cyl_points3d
}
cyl_calib = scaler.calibrate_and_align(cyl_sfm_dict, cyl_manifest)

T_cyl = cyl_calib.transform_matrix
xyz_raw_cyl = np.array([p["xyz"] for p in cyl_points3d.values()])
xyz_homo_cyl = np.column_stack([xyz_raw_cyl, np.ones(len(xyz_raw_cyl))])
xyz_metric_cyl = (xyz_homo_cyl @ T_cyl.T)[:, :3]

# Camera centers
cyl_cam_centers_metric = {}
for img_id, img_data in cyl_images.items():
    R = ColmapModelParser.qvec2rotmat(img_data["qvec"])
    t = img_data["tvec"]
    C_sfm = -R.T @ t
    C_m = (np.append(C_sfm, 1.0) @ T_cyl.T)[:3]
    cyl_cam_centers_metric[img_id] = C_m

# Fit axis on cylinder body points (Z in [20, 100])
r_temp = np.sqrt(xyz_metric_cyl[:, 0]**2 + xyz_metric_cyl[:, 1]**2)
mask_mid_cyl = (xyz_metric_cyl[:, 2] >= 20.0) & (xyz_metric_cyl[:, 2] <= 100.0) & (r_temp < 50.0)
pts_mid_cyl = xyz_metric_cyl[mask_mid_cyl]

if len(pts_mid_cyl) >= 10:
    A_fit = np.column_stack([pts_mid_cyl[:, 0], pts_mid_cyl[:, 1], np.ones(len(pts_mid_cyl))])
    b_fit = pts_mid_cyl[:, 0]**2 + pts_mid_cyl[:, 1]**2
    c_res, _, _, _ = np.linalg.lstsq(A_fit, b_fit, rcond=None)
    Xc_cyl = float(c_res[0] / 2.0)
    Yc_cyl = float(c_res[1] / 2.0)
else:
    Xc_cyl, Yc_cyl = 0.0, 0.0

r_metric_cyl = np.sqrt((xyz_metric_cyl[:, 0] - Xc_cyl)**2 + (xyz_metric_cyl[:, 1] - Yc_cyl)**2)
pids_cyl = np.array(list(cyl_points3d.keys()))

# Filter body points (Z in [10, 110], R <= 50)
mask_body_cyl = (xyz_metric_cyl[:, 2] >= 10.0) & (xyz_metric_cyl[:, 2] <= 110.0) & (r_metric_cyl <= 50.0)
body_idx_cyl = np.where(mask_body_cyl)[0]

cyl_body_pts = []
for idx in body_idx_cyl:
    pid = pids_cyl[idx]
    pt = cyl_points3d[pid]
    pt_m = xyz_metric_cyl[idx]
    
    ray_vecs = []
    for iid in pt["image_ids"]:
        if iid in cyl_images:
            C_m = cyl_cam_centers_metric[iid]
            v = C_m - pt_m
            ray_vecs.append(v / np.linalg.norm(v))
    
    if len(ray_vecs) >= 2:
        rmat = np.array(ray_vecs)
        cos_mat = np.clip(rmat @ rmat.T, -1.0, 1.0)
        triu = np.triu_indices(len(ray_vecs), k=1)
        angs = np.degrees(np.arccos(cos_mat[triu]))
        max_ang = float(np.max(angs))
        mean_ang = float(np.mean(angs))
    else:
        max_ang, mean_ang = 0.0, 0.0
    
    cyl_body_pts.append({
        "pid": pid, "x": float(pt_m[0]), "y": float(pt_m[1]), "z": float(pt_m[2]),
        "r": float(r_metric_cyl[idx]), "error": float(pt["error"]),
        "track_length": int(pt["track_length"]),
        "max_angle": max_ang, "mean_angle": mean_ang
    })

# Cylinder bands (10 to 110 mm)
cyl_bands_def = [
    ("Z 10-30 mm", 10.0, 30.0),
    ("Z 30-50 mm", 30.0, 50.0),
    ("Z 50-70 mm", 50.0, 70.0),
    ("Z 70-90 mm", 70.0, 90.0),
    ("Z 90-110 mm", 90.0, 110.0),
]
cyl_band_metrics = {}
for bname, z0, z1 in cyl_bands_def:
    b_pts = [p for p in cyl_body_pts if z0 <= p["z"] < z1]
    if b_pts:
        r_b = [p["r"] for p in b_pts]
        cyl_band_metrics[bname] = {
            "N": len(b_pts),
            "r_mean": float(np.mean(r_b)), "r_median": float(np.median(r_b)),
            "diam_mean": float(2 * np.mean(r_b)), "diam_median": float(2 * np.median(r_b))
        }
    else:
        cyl_band_metrics[bname] = {"N": 0, "r_mean": 0, "r_median": 0, "diam_mean": 0, "diam_median": 0}

# Cylinder regressions
z_cyl = np.array([p["z"] for p in cyl_body_pts])
r_cyl = np.array([p["r"] for p in cyl_body_pts])

if len(z_cyl) >= 10:
    b_cyl, a_cyl = np.polyfit(z_cyl, r_cyl, 1)
    pred_r = b_cyl * z_cyl + a_cyl
    res_cyl = r_cyl - pred_r
    r2_cyl = float(1.0 - (np.sum(res_cyl**2) / np.sum((r_cyl - np.mean(r_cyl))**2)))
    rmse_cyl = float(np.sqrt(np.mean(res_cyl**2)))
else:
    b_cyl, a_cyl, r2_cyl, rmse_cyl = 0, 0, 0, 0

# Cylinder HQ
hq_cyl = [p for p in cyl_body_pts if p["error"] < 1.0 and p["track_length"] >= 5 and p["max_angle"] >= 2.0]
z_hq_cyl = np.array([p["z"] for p in hq_cyl])
r_hq_cyl = np.array([p["r"] for p in hq_cyl])

if len(z_hq_cyl) >= 10:
    b_hq_cyl, a_hq_cyl = np.polyfit(z_hq_cyl, r_hq_cyl, 1)
    pred_hq = b_hq_cyl * z_hq_cyl + a_hq_cyl
    res_hq = r_hq_cyl - pred_hq
    r2_hq_cyl = float(1.0 - (np.sum(res_hq**2) / np.sum((r_hq_cyl - np.mean(r_hq_cyl))**2)))
    rmse_hq_cyl = float(np.sqrt(np.mean(res_hq**2)))
else:
    b_hq_cyl, a_hq_cyl, r2_hq_cyl, rmse_hq_cyl = 0, 0, 0, 0

# Diameters at Z=20, Z=60, Z=100
d20_cyl = float(2 * (a_hq_cyl + b_hq_cyl * 20.0))
d60_cyl = float(2 * (a_hq_cyl + b_hq_cyl * 60.0))
d100_cyl = float(2 * (a_hq_cyl + b_hq_cyl * 100.0))
d_mid_cyl = d60_cyl
error_dim_cyl = abs(d_mid_cyl - 70.00)
error_pct_cyl = (error_dim_cyl / 70.00) * 100.0
cyl_ratio_cyl = d100_cyl / d20_cyl if d20_cyl > 0 else 1.0

# ══════════════════════════════════════════════════════════════════════
#  FASE 5: CARGA DE RESULTADOS BASELINE DE LA TAZA
# ══════════════════════════════════════════════════════════════════════
with open("scratch/frame_selection_experiment/frame_selection_results.json") as f:
    fs_results = json.load(f)

cup_data = fs_results["variant_A"]

# Compile complete results dictionary
results = {
    "cylinder": {
        "label": "Cilindro de Referencia (Sintético con Ground Truth)",
        "ground_truth": {
            "real_diameter_mm": 70.00,
            "real_height_mm": 120.00,
            "nominal_b": 0.0000,
            "nominal_cyl_ratio": 1.0000
        },
        "num_frames": NUM_VIEWS,
        "num_reg_cams": len(cyl_images),
        "num_total_pts": len(cyl_points3d),
        "body_pts_count": len(cyl_body_pts),
        "matching_stats": cyl_matching_stats,
        "track_length": {
            "mean": float(np.mean([p["track_length"] for p in cyl_points3d.values()])),
            "median": float(np.median([p["track_length"] for p in cyl_points3d.values()])),
        },
        "reprojection_error": {
            "mean": float(np.mean([p["error"] for p in cyl_points3d.values()])),
            "median": float(np.median([p["error"] for p in cyl_points3d.values()])),
            "p90": float(np.percentile([p["error"] for p in cyl_points3d.values()], 90)),
        },
        "triangulation_angle": {
            "mean": float(np.mean([p["mean_angle"] for p in cyl_body_pts if p["mean_angle"] > 0])),
            "median": float(np.median([p["mean_angle"] for p in cyl_body_pts if p["mean_angle"] > 0])),
        },
        "band_metrics": cyl_band_metrics,
        "regression_all": {
            "a": float(a_cyl), "b": float(b_cyl), "r2": r2_cyl, "rmse": rmse_cyl, "N": len(cyl_body_pts)
        },
        "regression_hq": {
            "a_hq": float(a_hq_cyl), "b_hq": float(b_hq_cyl), "r2_hq": r2_hq_cyl, "rmse_hq": rmse_hq_cyl, "n_hq": len(hq_cyl),
            "d_low": d20_cyl, "d_mid": d60_cyl, "d_high": d100_cyl,
            "error_dimensional_mm": error_dim_cyl, "error_dimensional_pct": error_pct_cyl,
            "cyl_ratio": cyl_ratio_cyl
        },
        "scale_factor_mm": cyl_calib.scale_factor,
        "scale_error_mm": cyl_calib.scale_error_mm,
        "axis_center": {"Xc": Xc_cyl, "Yc": Yc_cyl}
    },
    "cup": {
        "label": "Taza de Cerámica (Real)",
        "ground_truth": {
            "real_mouth_diameter_mm": 82.50,
            "nominal_shape": "Cilíndrica / Leve ensanchamiento cerámico"
        },
        "num_frames": cup_data["num_frames"],
        "num_reg_cams": cup_data["num_reg_cams"],
        "num_total_pts": cup_data["num_total_pts"],
        "body_pts_count": cup_data["body_pts_count"],
        "matching_stats": cup_data["matching_stats"],
        "track_length": cup_data["track_length"],
        "reprojection_error": cup_data["reprojection_error"],
        "triangulation_angle": cup_data["triangulation_angle"],
        "band_metrics": cup_data["band_metrics"],
        "regression_all": cup_data["regression_all"],
        "regression_hq": cup_data["regression_hq"],
        "scale_factor_mm": cup_data["scale_factor_mm"]
    }
}

output_json_path = exp_base / "cylinder_vs_cup_results.json"
with open(output_json_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*75)
print("  EXPERIMENTO COMPLETADO CON ÉXITO")
print(f"  Resultados exportados a: {output_json_path}")
print("="*75)
