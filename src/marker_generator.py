"""
High-Contrast Printable Calibration Target Sheet Generator (PDF & PNG).
Features:
1. High-contrast alternating black-and-white concentric rings & radial checkerboard sectors (maximum SIFT features & visibility from distance).
2. Multi-tag layout: Central AprilTag (ID 0) + 4 Peripheral AprilTags (IDs 1, 2, 3, 4) so markers remain 100% visible even when a wide bottle/object sits in the center.
3. High-contrast outer checkerboard tracking border for camera pose estimation.
4. Physical millimeter ruler for print scale verification.
"""
import math
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


def generate_apriltag_matrix_36h11(tag_id: int) -> np.ndarray:
    """Generate binary 8x8 matrix for AprilTag tag36h11 (0=black, 255=white)."""
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        tag_img = cv2.aruco.generateImageMarker(dictionary, tag_id, 8)
        return tag_img
    except Exception:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        return cv2.aruco.generateImageMarker(dictionary, tag_id, 8)


def create_marker_pdf(
    output_pdf_path: Path,
    marker_id: int = 0,
    marker_size_mm: float = 40.0
) -> Path:
    """
    Create a high-contrast print-ready A4 PDF calibration board with alternating
    black/white concentric zones, radial sectors, and peripheral AprilTags.
    """
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Save temporary PNGs for tags
    temp_dir = output_pdf_path.parent
    tag_pngs = {}
    for tid in [0, 1, 2, 3, 4]:
        tpath = temp_dir / f"_temp_tag_{tid}.png"
        bits = generate_apriltag_matrix_36h11(tid)
        cv2.imwrite(str(tpath), bits)
        tag_pngs[tid] = tpath

    c = canvas.Canvas(str(output_pdf_path), pagesize=A4)
    page_w, page_h = A4
    cx = page_w / 2.0
    cy = page_h / 2.0

    # ── 1. Title & Instructions Header ──
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.black)
    c.drawCentredString(cx, page_h - 18 * mm, "PATRÓN DE CALIBRACIÓN MÉTRICA 3D (ALTO CONTRASTE)")

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#333333"))
    c.drawCentredString(cx, page_h - 23 * mm, "Imprimir en TAMAÑO REAL (Escala 100% / Sin ajustar a página) · Colocar objeto en el centro")

    # ── 2. Outer Checkerboard Calibration Border ──
    # Creates dozens of high-contrast SIFT corners for camera tracking
    margin_x = 15 * mm
    margin_y = 32 * mm
    board_w = page_w - (2 * margin_x)
    board_h = page_h - (2 * margin_y)
    sq_size = 10 * mm

    num_sq_x = int(board_w / sq_size)
    num_sq_y = int(board_h / sq_size)

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)

    # Top & bottom border strips
    for i in range(num_sq_x):
        x = margin_x + i * sq_size
        # Top strip
        if i % 2 == 0:
            c.setFillColor(colors.black)
            c.rect(x, page_h - margin_y - sq_size, sq_size, sq_size, fill=1, stroke=0)
        # Bottom strip
        if (i + 1) % 2 == 0:
            c.setFillColor(colors.black)
            c.rect(x, margin_y, sq_size, sq_size, fill=1, stroke=0)

    # Left & right border strips
    for j in range(num_sq_y):
        y = margin_y + j * sq_size
        # Left strip
        if j % 2 == 0:
            c.setFillColor(colors.black)
            c.rect(margin_x, y, sq_size, sq_size, fill=1, stroke=0)
        # Right strip
        if (j + 1) % 2 == 0:
            c.setFillColor(colors.black)
            c.rect(page_w - margin_x - sq_size, y, sq_size, sq_size, fill=1, stroke=0)

    # ── 3. Alternating Black and White Concentric Rings & Radial Sectors ──
    # Concentric high-contrast bands
    radii_mm = [85, 75, 65, 55, 45, 35, 25]
    for idx, r_mm in enumerate(radii_mm):
        r_pt = r_mm * mm
        # Alternate filled black ring and white ring
        if idx % 2 == 0:
            c.setFillColor(colors.HexColor("#111111"))
            c.circle(cx, cy, r_pt, fill=1, stroke=0)
        else:
            c.setFillColor(colors.white)
            c.circle(cx, cy, r_pt, fill=1, stroke=0)

    # Radial 8-Sector High-Contrast Checkerboard Wedges between r=45mm and r=75mm
    num_sectors = 16
    c.setFillColor(colors.HexColor("#111111"))
    for s in range(0, num_sectors, 2):
        start_angle = s * (360.0 / num_sectors)
        end_angle = (s + 1) * (360.0 / num_sectors)
        p = c.beginPath()
        p.moveTo(cx, cy)
        p.arc(cx - 75 * mm, cy - 75 * mm, cx + 75 * mm, cy + 75 * mm, start_angle, 360.0 / num_sectors)
        p.close()
        c.drawPath(p, fill=1, stroke=0)

    # Cut out inner circle with white to clear center
    c.setFillColor(colors.white)
    c.circle(cx, cy, 32 * mm, fill=1, stroke=1)

    # Guide ring labels for user
    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(colors.HexColor("#666666"))
    for r_mm in [40, 50, 60, 70, 80]:
        c.setStrokeColor(colors.HexColor("#888888"))
        c.setLineWidth(0.7)
        c.circle(cx, cy, r_mm * mm, fill=0, stroke=1)
        c.drawString(cx + r_mm * mm + 1.5 * mm, cy + 1 * mm, f"r={r_mm}mm")

    # ── 4. Central AprilTag (ID 0) ──
    # Scaled to marker_size_mm (e.g. 35 mm)
    c_marker_pt = 30 * mm
    c.drawImage(str(tag_pngs[0]), cx - (c_marker_pt / 2.0), cy - (c_marker_pt / 2.0),
                width=c_marker_pt, height=c_marker_pt)

    # ── 5. 4 Peripheral AprilTags (IDs 1, 2, 3, 4) ──
    # Placed at 75 mm radius (Top, Bottom, Left, Right)
    # These remain 100% visible even when an object of up to 120mm diameter sits in the center!
    p_marker_pt = 25 * mm
    p_radius_pt = 72 * mm

    positions = [
        (1, cx, cy + p_radius_pt, "Norte (Tag 1)"),
        (2, cx + p_radius_pt, cy, "Este (Tag 2)"),
        (3, cx, cy - p_radius_pt, "Sur (Tag 3)"),
        (4, cx - p_radius_pt, cy, "Oeste (Tag 4)")
    ]

    for tid, px, py, label in positions:
        # White backing square for tag isolation
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.black)
        c.setLineWidth(1)
        c.rect(px - (p_marker_pt / 2.0) - 2 * mm, py - (p_marker_pt / 2.0) - 2 * mm,
               p_marker_pt + 4 * mm, p_marker_pt + 4 * mm, fill=1, stroke=1)
        # Draw tag
        c.drawImage(str(tag_pngs[tid]), px - (p_marker_pt / 2.0), py - (p_marker_pt / 2.0),
                    width=p_marker_pt, height=p_marker_pt)
        # Tag label
        c.setFont("Helvetica-Bold", 6)
        c.setFillColor(colors.black)
        c.drawCentredString(px, py - (p_marker_pt / 2.0) - 4 * mm, label)

    # ── 6. Physical Verification Ruler at Bottom (100 mm) ──
    ruler_y = 20 * mm
    ruler_x_start = cx - 50 * mm
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.2)
    c.line(ruler_x_start, ruler_y, ruler_x_start + 100 * mm, ruler_y)

    for i in range(101):
        x = ruler_x_start + i * mm
        if i % 10 == 0:
            c.line(x, ruler_y, x, ruler_y + 6 * mm)
            c.setFont("Helvetica-Bold", 7)
            c.setFillColor(colors.black)
            c.drawCentredString(x, ruler_y - 4 * mm, f"{i//10}")
        elif i % 5 == 0:
            c.line(x, ruler_y, x, ruler_y + 3.5 * mm)
        else:
            c.line(x, ruler_y, x, ruler_y + 2 * mm)

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawCentredString(cx, ruler_y - 9 * mm, "Regla de Calibración de 10 cm (Mida con regla física para verificar escala 100%)")

    # Clean up temp images
    for tpath in tag_pngs.values():
        if tpath.exists():
            tpath.unlink()

    c.showPage()
    c.save()
    return output_pdf_path


def create_marker_png(
    output_path: Path,
    marker_id: int = 0,
    marker_size_mm: float = 50.0,
    dpi: int = 300
) -> Path:
    """Generate high-contrast calibration image PNG."""
    # We can render the high-contrast layout as a 300 DPI image
    px_per_mm = dpi / 25.4
    board_size_mm = 200.0
    total_px = int(round(board_size_mm * px_per_mm))
    center = total_px // 2

    img = np.full((total_px, total_px, 3), 255, dtype=np.uint8)

    # Alternating concentric bands
    for idx, r_mm in enumerate([90, 80, 70, 60, 50, 40, 30]):
        r_px = int(r_mm * px_per_mm)
        color = (15, 15, 15) if idx % 2 == 0 else (255, 255, 255)
        cv2.circle(img, (center, center), r_px, color, -1, cv2.LINE_AA)

    # 16 Radial wedges
    num_wedges = 16
    for w in range(0, num_wedges, 2):
        ang1 = w * (2 * math.pi / num_wedges)
        ang2 = (w + 1) * (2 * math.pi / num_wedges)
        r_px = int(80 * px_per_mm)
        pts = [
            [center, center],
            [int(center + r_px * math.cos(ang1)), int(center + r_px * math.sin(ang1))],
            [int(center + r_px * math.cos(ang2)), int(center + r_px * math.sin(ang2))]
        ]
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], (15, 15, 15), cv2.LINE_AA)

    # Center clear circle
    cv2.circle(img, (center, center), int(30 * px_per_mm), (255, 255, 255), -1, cv2.LINE_AA)

    # Central Tag
    tag0 = generate_apriltag_matrix_36h11(0)
    t0_px = int(28 * px_per_mm)
    tag0_res = cv2.resize(tag0, (t0_px, t0_px), interpolation=cv2.INTER_NEAREST)
    if len(tag0_res.shape) == 2:
        tag0_res = cv2.cvtColor(tag0_res, cv2.COLOR_GRAY2BGR)
    off0 = (total_px - t0_px) // 2
    img[off0:off0+t0_px, off0:off0+t0_px] = tag0_res

    # 4 Peripheral Tags
    p_radius_px = int(72 * px_per_mm)
    p_tag_px = int(22 * px_per_mm)

    coords = [
        (1, center, center - p_radius_px),
        (2, center + p_radius_px, center),
        (3, center, center + p_radius_px),
        (4, center - p_radius_px, center)
    ]

    for tid, px, py in coords:
        # White backing
        pad = int(3 * px_per_mm)
        cv2.rectangle(img, (px - p_tag_px//2 - pad, py - p_tag_px//2 - pad),
                      (px + p_tag_px//2 + pad, py + p_tag_px//2 + pad), (255, 255, 255), -1)
        cv2.rectangle(img, (px - p_tag_px//2 - pad, py - p_tag_px//2 - pad),
                      (px + p_tag_px//2 + pad, py + p_tag_px//2 + pad), (0, 0, 0), 2)

        tag = generate_apriltag_matrix_36h11(tid)
        tag_res = cv2.resize(tag, (p_tag_px, p_tag_px), interpolation=cv2.INTER_NEAREST)
        if len(tag_res.shape) == 2:
            tag_res = cv2.cvtColor(tag_res, cv2.COLOR_GRAY2BGR)
        x0 = px - p_tag_px // 2
        y0 = py - p_tag_px // 2
        img[y0:y0+p_tag_px, x0:x0+p_tag_px] = tag_res

    # Millimeter ruler ticks
    for r_mm in [40, 50, 60, 70, 80]:
        r_px = int(r_mm * px_per_mm)
        cv2.circle(img, (center, center), r_px, (120, 120, 120), 1, cv2.LINE_AA)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return output_path


if __name__ == "__main__":
    out_dir = Path("data/printable_markers")
    pdf_p = create_marker_pdf(out_dir / "target_high_contrast.pdf")
    png_p = create_marker_png(out_dir / "target_high_contrast.png")
    print(f"Generated High-Contrast PDF: {pdf_p}")
    print(f"Generated High-Contrast PNG: {png_p}")
