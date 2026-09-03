"""
Script de generación del reporte PDF de alta fidelidad editorial para:
'DOCUMENTACION_TECNICA_Y_BITACORA.pdf'
Autor e Ingeniero Responsable: Cristian Vargas
Diseño editorial con ReportLab Platypus, NumberedCanvas (Página X de Y),
portada ejecutiva, tablas autoajustables, tarjetas de arquitectura libres de caracteres ASCII corruptos,
fichas de tecnología con procedencia detallada y registro de incidentes post-mortem.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# Paleta cromática corporativa ejecutiva
PRIMARY = colors.HexColor("#1e1b4b")       # Deep Navy
SECONDARY = colors.HexColor("#4338ca")     # Indigo
ACCENT = colors.HexColor("#0284c7")        # Sky Blue
TEXT_MAIN = colors.HexColor("#0f172a")     # Dark Slate
TEXT_MUTED = colors.HexColor("#475569")    # Slate Gray
BG_LIGHT = colors.HexColor("#f8fafc")      # Soft White/Gray
BG_CALLOUT = colors.HexColor("#f1f5f9")    # Light Indigo/Gray
BORDER_COLOR = colors.HexColor("#cbd5e1")  # Border Line


class NumberedCanvas(canvas.Canvas):
    """
    Canvas de doble pasada para calcular el número total de páginas
    y renderizar encabezados y pies de página dinámicos 'Página X de Y'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, total_pages):
        # Omitir decoraciones en la portada (página 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        page_w, page_h = A4

        # Encabezado superior
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(SECONDARY)
        self.drawString(18 * mm, page_h - 13 * mm, "RECONSTRUCTOR MÉTRICO 3D · DOCUMENTACIÓN TÉCNICA MAESTRA & AUDITORÍA")
        
        self.setFont("Helvetica", 7.5)
        self.setFillColor(TEXT_MUTED)
        self.drawRightString(page_w - 18 * mm, page_h - 13 * mm, "Autor: Cristian Vargas · Reconstructor Métrico 3D")

        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(18 * mm, page_h - 15 * mm, page_w - 18 * mm, page_h - 15 * mm)

        # Pie de página inferior
        self.line(18 * mm, 15 * mm, page_w - 18 * mm, 15 * mm)
        self.setFont("Helvetica", 7.5)
        self.drawString(18 * mm, 10.5 * mm, "DOCUMENTO TÉCNICO CONFIDENCIAL · PROYECTO PERSONAL CRISTIAN VARGAS")

        page_str = f"Página {self._pageNumber} de {total_pages}"
        self.drawRightString(page_w - 18 * mm, 10.5 * mm, page_str)

        self.restoreState()


def create_callout(text: str, title: str = "PUNTO CLAVE", bar_color=SECONDARY, style_sheet=None) -> Table:
    """Crea una caja de aviso destacada con barra lateral y fondo suave."""
    p_title = Paragraph(f"<b><font color='{bar_color.hexval()}'>▍ {title}</font></b>", style_sheet["CalloutTitle"])
    p_body = Paragraph(text, style_sheet["CalloutBody"])
    content = [p_title, Spacer(1, 1.8 * mm), p_body]
    
    t = Table([[content]], colWidths=[174 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_CALLOUT),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('LINEBEFORE', (0, 0), (0, 0), 3.0, bar_color),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
    ]))
    return t


def create_code_block(code_text: str, style_sheet=None) -> Table:
    """Crea un bloque de código monospaciado con fondo oscuro."""
    safe_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    safe_text = safe_text.replace('\n', '<br/>').replace(' ', '&nbsp;')
    p = Paragraph(safe_text, style_sheet["CodeStyle"])
    t = Table([[p]], colWidths=[174 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#0f172a")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#334155")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


def create_incident_card(num: int, title: str, component: str, symptom: str, root_cause: str, solution: str, impact: str, style_sheet) -> KeepTogether:
    """Genera una ficha estilizada para cada incidente post-mortem de la bitácora."""
    header_text = f"<b><font color='{PRIMARY.hexval()}'>INCIDENTE TÉCNICO #{num}: {title.upper()}</font></b>"
    
    rows = [
        [Paragraph(header_text, style_sheet["IncidentHeader"]), ""],
        [Paragraph("<b>Componentes:</b>", style_sheet["IncidentLabel"]), Paragraph(f"<code>{component}</code>", style_sheet["IncidentValue"])],
        [Paragraph("<b>Síntoma / Error:</b>", style_sheet["IncidentLabel"]), Paragraph(symptom, style_sheet["IncidentValue"])],
        [Paragraph("<b>Causa Raíz (RCA):</b>", style_sheet["IncidentLabel"]), Paragraph(root_cause, style_sheet["IncidentValue"])],
        [Paragraph("<b>Solución de Cristian Vargas:</b>", style_sheet["IncidentLabel"]), Paragraph(solution, style_sheet["IncidentValue"])],
        [Paragraph("<b>Impacto / Verificación:</b>", style_sheet["IncidentLabel"]), Paragraph(impact, style_sheet["IncidentValue"])],
    ]
    
    t = Table(rows, colWidths=[40 * mm, 134 * mm])
    t.setStyle(TableStyle([
        ('SPAN', (0, 0), (1, 0)),
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#f1f5f9")),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#ffffff")),
        ('LINEBEFORE', (0, 0), (0, -1), 3.0, SECONDARY),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 1), (-1, -1), 0.4, colors.HexColor("#f1f5f9")),
        ('TOPPADDING', (0, 0), (-1, -1), 3.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.2),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    return KeepTogether([t, Spacer(1, 3.5 * mm)])


def generate_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    base_styles = getSampleStyleSheet()

    styles = {
        "CoverTag": ParagraphStyle(
            "CoverTag",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12,
            textColor=ACCENT,
            textTransform="uppercase",
            spaceAfter=5
        ),
        "CoverTitle": ParagraphStyle(
            "CoverTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=PRIMARY,
            alignment=0,
            spaceAfter=9
        ),
        "CoverSubtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14.5,
            textColor=TEXT_MUTED,
            spaceAfter=13
        ),
        "H1": ParagraphStyle(
            "Heading1_Custom",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15.5,
            textColor=PRIMARY,
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True
        ),
        "H2": ParagraphStyle(
            "Heading2_Custom",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13.5,
            textColor=SECONDARY,
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True
        ),
        "Body": ParagraphStyle(
            "Body_Custom",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=11.5,
            textColor=TEXT_MAIN,
            spaceAfter=3.5
        ),
        "Bullet": ParagraphStyle(
            "Bullet_Custom",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=11.5,
            textColor=TEXT_MAIN,
            leftIndent=11,
            firstLineIndent=-7,
            spaceAfter=2
        ),
        "CalloutTitle": ParagraphStyle(
            "CalloutTitle",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.0,
            leading=10,
            textColor=SECONDARY
        ),
        "CalloutBody": ParagraphStyle(
            "CalloutBody",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.5,
            textColor=TEXT_MAIN
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9.5,
            textColor=colors.white,
            alignment=1
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=7.0,
            leading=9.0,
            textColor=TEXT_MAIN
        ),
        "TableCellCenter": ParagraphStyle(
            "TableCellCenter",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=7.0,
            leading=9.0,
            textColor=TEXT_MAIN,
            alignment=1
        ),
        "CodeStyle": ParagraphStyle(
            "CodeStyle",
            parent=base_styles["Normal"],
            fontName="Courier",
            fontSize=6.8,
            leading=9.0,
            textColor=colors.HexColor("#38bdf8")
        ),
        "IncidentHeader": ParagraphStyle(
            "IncidentHeader",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9.5,
            textColor=PRIMARY
        ),
        "IncidentLabel": ParagraphStyle(
            "IncidentLabel",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.0,
            leading=8.8,
            textColor=SECONDARY
        ),
        "IncidentValue": ParagraphStyle(
            "IncidentValue",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=7.0,
            leading=8.8,
            textColor=TEXT_MAIN
        ),
        "StageTitle": ParagraphStyle(
            "StageTitle",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=9.5,
            textColor=PRIMARY
        ),
        "StageTech": ParagraphStyle(
            "StageTech",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=6.8,
            leading=8.5,
            textColor=ACCENT
        ),
        "StageDesc": ParagraphStyle(
            "StageDesc",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=6.8,
            leading=8.8,
            textColor=TEXT_MAIN
        )
    }

    story = []

    # ══════════════════════════════════════════════════════════════
    # ── PORTADA EJECUTIVA ──
    # ══════════════════════════════════════════════════════════════
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("PROYECTO PERSONAL · INFORME MAESTRO DE INGENIERÍA & AUDITORÍA TÉCNICA", styles["CoverTag"]))
    story.append(Paragraph("Documentación Técnica, Arquitectura del Sistema y Bitácora de Ingeniería (Post-Mortem)", styles["CoverTitle"]))
    story.append(Paragraph(
        "Reconstructor Métrico 3D (Video RGB de Smartphone a Modelo STL Calibrado para Ortesis, Férulas y Prototipos). Especificación técnica exhaustiva del pipeline de 5 etapas, fundamentos matemáticos, catálogo de procedencia y autoría de librerías, bitácora de 12 incidentes superados y validación metrológica experimental.",
        styles["CoverSubtitle"]
    ))

    story.append(HRFlowable(width="100%", thickness=2, color=SECONDARY, spaceBefore=0, spaceAfter=11))

    meta_data = [
        [Paragraph("<b>Proyecto:</b>", styles["TableCell"]), Paragraph("Reconstructor Métrico 3D (Video RGB a STL Calibrado)", styles["TableCell"])],
        [Paragraph("<b>Fecha de Emisión:</b>", styles["TableCell"]), Paragraph("Septiembre de 2026", styles["TableCell"])],
        [Paragraph("<b>Versión:</b>", styles["TableCell"]), Paragraph("1.2.0", styles["TableCell"])],
        [Paragraph("<b>Autor / Ingeniero Responsable:</b>", styles["TableCell"]), Paragraph("<b>Cristian Vargas</b>", styles["TableCell"])],
        [Paragraph("<b>Clasificación:</b>", styles["TableCell"]), Paragraph("Documento Técnico Maestro Confidencial de I+D", styles["TableCell"])]
    ]
    t_meta = Table(meta_data, colWidths=[45 * mm, 129 * mm])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.0),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_meta)

    story.append(Spacer(1, 7 * mm))

    story.append(create_callout(
        "<b>Dictamen Ejecutivo de Cristian Vargas:</b> La viabilidad de reconstruir geometría métrica precisa a partir de video monocular de smartphone queda <b>plenamente confirmada empíricamente</b> (+0.16 mm de error en cilindro de control de 70 mm). El factor determinante del éxito no reside en modelos neuronales opacos, sino en una <b>cadena de metrología ciber-física rigurosa</b>: muestreo de fotogramas nítidos en una pasada secuencial a 50+ FPS, fotogrametría epipolar clásica (COLMAP), calibración Multi-Tag con compensación geométrica de origen y solidificación estanca (Poisson) con recorte adaptativo de región de interés.",
        title="DECLARACIÓN FORMAL DE INGENIERÍA",
        bar_color=SECONDARY,
        style_sheet=styles
    ))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # ── 1. PROPÓSITO DEL PROYECTO Y JUSTIFICACIÓN DE LA BITÁCORA ──
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("1. Propósito del Proyecto y Justificación de la Bitácora", styles["H1"]))
    
    story.append(Paragraph("1.1. Contexto y Propósito del MVP", styles["H2"]))
    story.append(Paragraph(
        "El objetivo primario de esta aplicación, desarrollada por Cristian Vargas, es responder a una pregunta crítica de ingeniería biomédica: <i>¿Es factible sustituir la toma manual de medidas con cinta métrica y los costosos escáneres 3D clínicos mediante un smartphone convencional RGB para la fabricación de órtesis personalizadas?</i>",
        styles["Body"]
    ))
    story.append(Paragraph(
        "La toma tradicional manual aproxima el antebrazo humano a elipses simétricas mediante la fórmula de Ramanujan, ignorando relieves óseos asimétricos vitales (como la apófisis estiloides cubital o epicóndilos) y provocando puntos de presión o isquemia. Los escáneres profesionales (EinScan, Artec), aunque precisos (0.05 a 0.2 mm), resultan inaccesibles para el despliegue clínico masivo por su costo y complejidad. Este MVP demuestra que un flujo automatizado de fotogrametría asistida por marcadores físicos alcanza precisión submilimétrica real (error < 0.5%) operando 100% en computadoras estándar con CPU.",
        styles["Body"]
    ))

    story.append(Paragraph("1.2. Justificación Epistemológica de la Bitácora de Ingeniería", styles["H2"]))
    story.append(Paragraph(
        "En proyectos de visión artificial y reconstrucción 3D métrica, una bitácora técnica no es un simple historial de commits; constituye un instrumento indispensable de control de calidad y preservación tecnológica por cuatro razones fundamentales:",
        styles["Body"]
    ))

    b_bitacora = [
        "<b>Hiper-sensibilidad Paramétrica:</b> La fotogrametría encadena algoritmos donde una ligera descompensación (e.g. estimar normales con radio euclidiano fijo antes de aplicar la escala métrica) conduce al colapso total de la malla ('montaña amorfa'). La bitácora documenta las relaciones de causa-efecto exactas para que el sistema sea determinista.",
        "<b>Cultura Post-Mortem y Análisis de Causa Raíz (RCA):</b> Ante un artefacto geométrico, el análisis sistemático descubre el motivo físico/matemático real (e.g. oclusión de tags por objetos de base ancha o micro-temblores musculares en extremidades suspendidas en el aire), evitando conjeturas erróneas.",
        "<b>Marco Regulatorio de Dispositivos Médicos (ISO 13485 / IEC 62304 / ANMAT):</b> El software médico (SaMD) exige trazabilidad estricta del ciclo de vida del software, registro minucioso de anomalías resueltas y gestión de riesgos sistemática. Esta bitácora conforma la evidencia de auditoría exigida por normativas internacionales.",
        "<b>Aceleración de Transferencia Tecnológica:</b> Garantiza que cualquier ingeniero o investigador comprenda de inmediato por qué Cristian Vargas descartó ciertas vías (como el emparejador exhaustivo o lecturas no secuenciales con cap.set), previniendo regresiones costosas."
    ]
    for b in b_bitacora:
        story.append(Paragraph(f"• {b}", styles["Bullet"]))

    story.append(Spacer(1, 3 * mm))

    # ══════════════════════════════════════════════════════════════
    # ── 2. ARQUITECTURA GENERAL Y FLUJO DE 5 ETAPAS (TABLA LIMPIA) ──
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("2. Arquitectura General del Sistema y Flujo de Procesamiento", styles["H1"]))
    story.append(Paragraph(
        "El sistema orquesta un pipeline determinista dividido en cinco fases desacopladas, complementadas por una capa de backend reactivo y un visualizador 3D WebGL interactivo. A continuación se presenta el flujo arquitectónico estructurado:",
        styles["Body"]
    ))

    # Tabla limpia de las 5 Etapas (SIN caracteres ASCII que se corrompen en PDF)
    stages_flow_data = [
        [
            Paragraph("Etapa", styles["TableHeader"]),
            Paragraph("Módulo & Algoritmo", styles["TableHeader"]),
            Paragraph("Librería / Autoría", styles["TableHeader"]),
            Paragraph("Operación Algorítmica y Resultado Técnico", styles["TableHeader"])
        ],
        [
            Paragraph("<b>ETAPA 1</b>", styles["TableCellCenter"]),
            Paragraph("<b>Ingesta Inteligente de Video</b><br/>(Sharp Windows)", styles["StageTitle"]),
            Paragraph("<b>OpenCV</b><br/><font color='#475569'>Intel / Bradski</font>", styles["StageTech"]),
            Paragraph("Lectura secuencial continua a 50+ FPS. Divide el video en 60 ventanas uniformes (0% a 100% de duración) y evalúa la varianza del Laplaciano para retener únicamente el cuadro de máxima nitidez por ventana (cero motion blur).", styles["StageDesc"])
        ],
        [
            Paragraph("<b>ETAPA 2</b>", styles["TableCellCenter"]),
            Paragraph("<b>Structure from Motion (SfM)</b><br/>(Camera Poses)", styles["StageTitle"]),
            Paragraph("<b>COLMAP</b><br/><font color='#475569'>ETH Zürich / Schönberger</font>", styles["StageTech"]),
            Paragraph("Extracción de descriptores SIFT. Emparejamiento secuencial temporal (overlap=10, cuadrático) y Bundle Adjustment incremental (Ceres Solver). Detecta automáticamente CPU vs CUDA y selecciona el submodelo principal con mayor número de cámaras.", styles["StageDesc"])
        ],
        [
            Paragraph("<b>ETAPA 3</b>", styles["TableCellCenter"]),
            Paragraph("<b>Calibración Métrica</b><br/>(Multi-Tag Scaler)", styles["StageTitle"]),
            Paragraph("<b>Pupil-AprilTags</b><br/><font color='#475569'>UMich / Edwin Olson</font>", styles["StageTech"]),
            Paragraph("Detección subpíxel de esquinas en familia tag36h11. Triangulación epipolar 3D de las esquinas para calcular el factor de escala hacia milímetros reales. Algoritmo Multi-Tag compensa el offset para centrar el origen (0,0,0) en la hoja con normal +Z hacia arriba.", styles["StageDesc"])
        ],
        [
            Paragraph("<b>ETAPA 4</b>", styles["TableCellCenter"]),
            Paragraph("<b>Mallado Poisson</b><br/>(Watertight ROI)", styles["StageTitle"]),
            Paragraph("<b>Open3D</b><br/><font color='#475569'>Intel Labs / Koltun</font>", styles["StageTech"]),
            Paragraph("Estimación de normales invariante a escala mediante KNN (k=25). Reconstrucción de superficie Screened Poisson (depth=8). Poda de vértices de baja densidad, purga de NaNs y recorte ROI estricto (Z >= 2 mm y R <= 120 mm) eliminando mesa y suelo.", styles["StageDesc"])
        ],
        [
            Paragraph("<b>ETAPA 5</b>", styles["TableCellCenter"]),
            Paragraph("<b>Metrología Seccional</b><br/>(Slicing & Analysis)", styles["StageTitle"]),
            Paragraph("<b>Trimesh & Shapely</b><br/><font color='#475569'>GEOS C++ Engine</font>", styles["StageTech"]),
            Paragraph("Corte transversal de la malla STL en planos ortogonales cada 10 mm. Construcción de polígonos 2D cerrados continuos. Cálculo de perímetro real, área (Gauss/Shoelace) y diámetro equivalente Deq = 2√(A/π). Exportación STL binario y JSON de reporte.", styles["StageDesc"])
        ]
    ]

    t_stages = Table(stages_flow_data, colWidths=[16 * mm, 46 * mm, 34 * mm, 78 * mm])
    t_stages.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4.5),
    ]))
    story.append(t_stages)

    story.append(Spacer(1, 3 * mm))

    # Capas de Servicio y Dashboard
    delivery_layer_data = [
        [
            Paragraph("<b>Capa de Servicios Backend (FastAPI + SSE)</b><br/>Endpoints REST asíncronos para subida de video multiparte sin saturación de RAM. Transmisión reactiva de telemetría a 5 Hz mediante Server-Sent Events (SSE) con porcentaje global, etapa en vivo y cronómetro.", styles["TableCell"]),
            Paragraph("<b>Capa de Visualización Frontend (Three.js WebGL)</b><br/>Visor 3D orbital fluido con sombreado de estudio fotográfico PBR, alternancia de modo alambre (Wireframe), visualización de planos de corte (Slicing), inspección de dimensiones y panel de historial dinámico.", styles["TableCell"])
        ]
    ]
    t_delivery = Table(delivery_layer_data, colWidths=[87 * mm, 87 * mm])
    t_delivery.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#f1f5f9")),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('LINEBEFORE', (0, 0), (0, 0), 2.5, SECONDARY),
        ('LINEBEFORE', (1, 0), (1, 0), 2.5, ACCENT),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_delivery)

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # ── 3. STACK TECNOLÓGICO, PROCEDENCIA Y LIBRERÍAS ──
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("3. Stack Tecnológico, Procedencia de Librerías y Criterios de Selección", styles["H1"]))
    story.append(Paragraph(
        "A continuación se detalla cada componente del stack seleccionado por <b>Cristian Vargas</b>, documentando su autoría original, la institución o laboratorio de investigación de procedencia, su repositorio oficial y la justificación técnica de su elección frente a soluciones alternativas:",
        styles["Body"]
    ))

    stack_data = [
        [
            Paragraph("Librería / Tool", styles["TableHeader"]),
            Paragraph("Versión", styles["TableHeader"]),
            Paragraph("Autor / Institución de Origen", styles["TableHeader"]),
            Paragraph("Fuente / Repositorio Oficial", styles["TableHeader"]),
            Paragraph("Licencia", styles["TableHeader"]),
            Paragraph("Rol en el Pipeline", styles["TableHeader"])
        ],
        [
            Paragraph("<b>OpenCV</b>", styles["TableCell"]),
            Paragraph(">=4.8.0", styles["TableCellCenter"]),
            Paragraph("Gary Bradski, Vadim Pisarevsky<br/><i>Intel & OpenCV Foundation</i>", styles["TableCell"]),
            Paragraph("PyPI oficial / GitHub:<br/><code>opencv/opencv-python</code>", styles["TableCell"]),
            Paragraph("Apache-2.0", styles["TableCellCenter"]),
            Paragraph("Lectura secuencial de video a 50+ FPS y varianza Laplaciana.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Pupil-AprilTags</b>", styles["TableCell"]),
            Paragraph(">=1.0.4", styles["TableCellCenter"]),
            Paragraph("Prof. Edwin Olson (Univ. Michigan)<br/><i>Pupil Labs GmbH (Berlín)</i>", styles["TableCell"]),
            Paragraph("PyPI oficial / GitHub:<br/><code>pupil-labs/apriltags</code>", styles["TableCell"]),
            Paragraph("BSD-2", styles["TableCellCenter"]),
            Paragraph("Detección subpíxel tag36h11 inmune a ángulos rasantes.", styles["TableCell"])
        ],
        [
            Paragraph("<b>COLMAP</b>", styles["TableCell"]),
            Paragraph("3.8 / 3.9", styles["TableCellCenter"]),
            Paragraph("Dr. Johannes Schönberger (ETH Zürich)<br/><i>Jan-Michael Frahm (UNC)</i>", styles["TableCell"]),
            Paragraph("GitHub Releases oficiales:<br/><code>colmap/colmap</code> (Win64)", styles["TableCell"]),
            Paragraph("BSD-3", styles["TableCellCenter"]),
            Paragraph("Structure from Motion, SIFT, Sequential Matcher y Ceres BA.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Open3D</b>", styles["TableCell"]),
            Paragraph(">=0.18.0", styles["TableCellCenter"]),
            Paragraph("Qian-Yi Zhou, Jaesik Park, Vladlen Koltun<br/><i>Intel Labs Visual Computing</i>", styles["TableCell"]),
            Paragraph("PyPI oficial / GitHub:<br/><code>isl-org/Open3D</code>", styles["TableCell"]),
            Paragraph("MIT", styles["TableCellCenter"]),
            Paragraph("Normales KNN (k=25), Screened Poisson y poda por densidad.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Trimesh</b>", styles["TableCell"]),
            Paragraph(">=4.0.0", styles["TableCellCenter"]),
            Paragraph("Michael Dawson-Haggerty<br/><i>Comunidad Python Geometry</i>", styles["TableCell"]),
            Paragraph("PyPI oficial / GitHub:<br/><code>mikedh/trimesh</code>", styles["TableCell"]),
            Paragraph("MIT", styles["TableCellCenter"]),
            Paragraph("Slicing transversal ortogonal y exportación STL binario.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Shapely</b>", styles["TableCell"]),
            Paragraph(">=2.0.0", styles["TableCellCenter"]),
            Paragraph("Sean Gillies (Toblerity)<br/><i>GEOS C++ Engine (OGC/PostGIS)</i>", styles["TableCell"]),
            Paragraph("PyPI oficial / GitHub:<br/><code>shapely/shapely</code>", styles["TableCell"]),
            Paragraph("BSD-3", styles["TableCellCenter"]),
            Paragraph("Áreas 2D de Gauss (Shoelace) y perímetros continuos reales.", styles["TableCell"])
        ],
        [
            Paragraph("<b>FastAPI</b>", styles["TableCell"]),
            Paragraph(">=0.110", styles["TableCellCenter"]),
            Paragraph("Sebastián Ramírez (<code>tiangolo</code>)<br/><i>Comunidad FastAPI/Starlette</i>", styles["TableCell"]),
            Paragraph("PyPI oficial / GitHub:<br/><code>tiangolo/fastapi</code>", styles["TableCell"]),
            Paragraph("MIT", styles["TableCellCenter"]),
            Paragraph("API REST asíncrona, gestión multipart y validación Pydantic.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Uvicorn</b>", styles["TableCell"]),
            Paragraph(">=0.28", styles["TableCellCenter"]),
            Paragraph("Tom Christie & Encode Project<br/><i>ASGI Ecosystem</i>", styles["TableCell"]),
            Paragraph("PyPI oficial / GitHub:<br/><code>encode/uvicorn</code>", styles["TableCell"]),
            Paragraph("BSD-3", styles["TableCellCenter"]),
            Paragraph("Servidor ASGI ultrarrápido sobre protocolo uvloop.", styles["TableCell"])
        ],
        [
            Paragraph("<b>SSE-Starlette</b>", styles["TableCell"]),
            Paragraph(">=2.0.0", styles["TableCellCenter"]),
            Paragraph("Marcelo Trylesinski<br/><i>SysMo-Teams / Starlette</i>", styles["TableCell"]),
            Paragraph("PyPI oficial / GitHub:<br/><code>sysmo-teams/sse-starlette</code>", styles["TableCell"]),
            Paragraph("BSD-3", styles["TableCellCenter"]),
            Paragraph("Streaming unidireccional de telemetría a 5 Hz en vivo.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Three.js</b>", styles["TableCell"]),
            Paragraph("r128", styles["TableCellCenter"]),
            Paragraph("Ricardo Cabello (<code>Mr.doob</code>)<br/><i>Comunidad WebGL</i>", styles["TableCell"]),
            Paragraph("GitHub oficial / Local:<br/><code>web/static/js/three.min.js</code>", styles["TableCell"]),
            Paragraph("MIT", styles["TableCellCenter"]),
            Paragraph("Renderizador WebGL interactivo con PBR, Wireframe y Slices.", styles["TableCell"])
        ],
        [
            Paragraph("<b>ReportLab</b>", styles["TableCell"]),
            Paragraph(">=4.1.0", styles["TableCellCenter"]),
            Paragraph("Andy Robinson<br/><i>ReportLab Inc. (Londres, UK)</i>", styles["TableCell"]),
            Paragraph("PyPI oficial:<br/><code>reportlab.com</code>", styles["TableCell"]),
            Paragraph("BSD-Mod", styles["TableCellCenter"]),
            Paragraph("Generación de patrones métricos A4 y reportes de ingeniería.", styles["TableCell"])
        ],
        [
            Paragraph("<b>NumPy / SciPy</b>", styles["TableCell"]),
            Paragraph("1.26 / 1.11", styles["TableCellCenter"]),
            Paragraph("Travis Oliphant & NumFOCUS<br/><i>Comunidad Científica Python</i>", styles["TableCell"]),
            Paragraph("PyPI oficial:<br/><code>numpy.org</code>, <code>scipy.org</code>", styles["TableCell"]),
            Paragraph("BSD-3", styles["TableCellCenter"]),
            Paragraph("Álgebra lineal [R|t], ajuste de planos por SVD y elipses.", styles["TableCell"])
        ]
    ]

    t_stack = Table(stack_data, colWidths=[23 * mm, 12 * mm, 43 * mm, 37 * mm, 15 * mm, 44 * mm])
    t_stack.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.8),
        ('LEFTPADDING', (0, 0), (-1, -1), 3.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_stack)

    story.append(Spacer(1, 3 * mm))

    # Explicación narrativa de por qué se eligieron estas fuentes
    story.append(Paragraph("3.2. Criterios de Selección y Rechazo de Alternativas por Cristian Vargas", styles["H2"]))
    
    reasons = [
        "<b>OpenCV vs. FFmpeg CLI:</b> FFmpeg obligaba a escribir cientos de imágenes a disco, generando un cuello de botella de E/S. OpenCV decodifica directamente en matrices en memoria RAM a 50+ FPS.",
        "<b>Pupil-AprilTags vs. OpenCV ArUco:</b> AprilTag utiliza gradientes de intensidad y corrección de Hamming de alta entropía, resultando inmune a falsos positivos bajo ángulos rasantes donde ArUco colapsa.",
        "<b>COLMAP vs. DUSt3R / MASt3R / OpenMVG:</b> DUSt3R opera bajo licencia restrictiva no comercial (CC BY-NC-SA) y exige GPUs costosas. COLMAP cuenta con licencia BSD permisiva y un solver Ceres sumamente optimizado en CPU multihilo.",
        "<b>Open3D vs. CGAL:</b> CGAL presenta una sobrecarga excesiva de enlaces y compilación en Windows. Open3D brinda el algoritmo de Poisson optimizado en C++ nativo con APIs de Python transparentes.",
        "<b>FastAPI + SSE vs. WebSockets / Polling:</b> WebSockets introduce complejidad de reconexión bidireccional y Polling satura la red con cientos de peticiones. SSE es ligero, unidireccional y estándar en navegadores.",
        "<b>ReportLab vs. WeasyPrint / wkhtmltopdf:</b> Las herramientas CSS introducen imprecisiones de márgenes. ReportLab trabaja con unidades métricas absolutas en milímetros (mm), garantizando escala 100% real de impresión."
    ]
    for r in reasons:
        story.append(Paragraph(f"• {r}", styles["Bullet"]))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # ── 4. ANÁLISIS PROFUNDO DE LA FORMA DE PROCESAMIENTO ──
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("4. Análisis Profundo de la Forma y Algoritmos de Procesamiento", styles["H1"]))

    story.append(Paragraph("4.1. Ingesta Inteligente de Video y Extracción de Cuadros Nítidos", styles["H2"]))
    story.append(Paragraph(
        "El extractor divide la duración total del video en 60 ventanas temporales equivalentes. En cada ventana, evalúa la nitidez mediante la varianza del operador Laplaciano:",
        styles["Body"]
    ))
    story.append(create_code_block("Score(I) = Varianza(nabla^2 I) = (1/N) * sum_{x,y} ( L(x,y) - mu_L )^2", style_sheet=styles))
    story.append(Paragraph(
        "Un cuadro con bordes contrastados produce alta varianza (> 120), mientras que el desenfoque de movimiento (motion blur) hace colapsar el valor (< 40). El sistema extrae el cuadro óptimo por ventana. Al ejecutarse en una <b>única pasada secuencial</b> continua (`cap.read()`), procesa 500 cuadros en 9.2 segundos (frente a más de 2 minutos que consumía la búsqueda aleatoria con `cap.set`).",
        styles["Body"]
    ))

    story.append(Paragraph("4.2. Structure from Motion (SfM) con COLMAP", styles["H2"]))
    story.append(Paragraph(
        "En lugar de un emparejamiento exhaustivo cuadrático, se utiliza `sequential_matcher` con solapamiento temporal (`overlap = 10`, `quadratic_overlap = 1`). Esto compara cada cuadro únicamente con sus vecinos temporales y con cuadros equidistantes para resolver el cierre de ciclo 360°. El Bundle Adjustment (Ceres Solver) optimiza conjuntamente las posiciones de cámara y los puntos 3D minimizando el error de reproyección. Se fijó `ba_refine_extra_params = 0` para evitar inestabilidades en lentes no calibrados.",
        styles["Body"]
    ))

    story.append(Paragraph("4.3. Calibración Métrica Multi-Tag y Alineación al Centro", styles["H2"]))
    story.append(Paragraph(
        "Para eliminar la ambigüedad de escala monocular, se detectan los marcadores AprilTag y se triangulan sus cuatro esquinas en el espacio SfM. El factor de escala se deduce comparando la distancia euclidiana triangulada con la dimensión física conocida ($35.0\\text{ mm}$ o $50.0\\text{ mm}$):",
        styles["Body"]
    ))
    story.append(create_code_block("ScaleFactor = DistanciaReal_mm / DistanciaTriangulada_SfM", style_sheet=styles))
    story.append(Paragraph(
        "El algoritmo <b>Multi-Tag</b> identifica automáticamente cuál de los 5 marcadores posee mayor número de vistas. Aplica una compensación geométrica de offset para que el origen $(0,0,0)$ quede perfectamente anclado en el <b>centro físico del tablero</b>, asegurando además que el vector normal $+Z$ apunte ortogonalmente hacia arriba (hacia las cámaras).",
        styles["Body"]
    ))

    story.append(Paragraph("4.4. Mallado Poisson y Recorte de Región de Interés (ROI)", styles["H2"]))
    story.append(Paragraph(
        "Las normales se calculan mediante un vecindario KNN ($k=25$), el cual es <b>invariante a escala</b> y previene los colapsos que ocurrían con radios fijos. La superficie se solidifica mediante <i>Screened Poisson Surface Reconstruction</i> (`depth = 8`). Luego, se aplica una doble limpieza: poda de vértices de baja densidad de soporte (eliminando burbujas flotantes) y recorte estricto de Región de Interés ($Z \\ge 2.0\\text{ mm}$ y radio horizontal $R \\le 120\\text{ mm}$), aislando el objeto puro y extirpando la mesa y el suelo.",
        styles["Body"]
    ))

    story.append(Paragraph("4.5. Rebanado Ortogonal (Slicing) y Metrología Seccional", styles["H2"]))
    story.append(Paragraph(
        "La malla STL se intercepta con planos horizontales cada $\\Delta Z$ ($5\\text{ a }10\\text{ mm}$). Los polígonos cerrados 2D resultantes se integran mediante el teorema de Green/Gauss para derivar área, perímetro real continuo y diámetro circular equivalente ($D_{\\text{eq}} = 2\\sqrt{A/\\pi}$), listos para exportación o adaptación en software ortésico.",
        styles["Body"]
    ))

    story.append(Spacer(1, 4 * mm))

    # ══════════════════════════════════════════════════════════════
    # ── 5. BITÁCORA EXHAUSTIVA DE INCIDENTES (POST-MORTEM) ──
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("5. Bitácora Exhaustiva de Incidentes Técnicos y Soluciones (Post-Mortem)", styles["H1"]))
    story.append(Paragraph(
        "A continuación se documentan con máximo rigor técnico los 12 incidentes críticos superados por <b>Cristian Vargas</b> durante el desarrollo del sistema:",
        styles["Body"]
    ))

    incidents = [
        (1, "Crash en subida de videos con nombres de WhatsApp",
         "web/server.py",
         "ValueError: invalid literal for int() with base 10 al subir videos desde el navegador.",
         "El backend intentaba convertir los primeros 20 caracteres del archivo a entero asumiendo un timestamp puro, pero los archivos de WhatsApp contienen texto y espacios.",
         "Sanitización con regex (re.sub) y generación segura de scan_id mediante fecha ISO, timestamp Unix y entropía criptográfica (os.urandom).",
         "Soporte universal e infalible para cualquier nombre de archivo multimedia entrante."),

        (2, "Fallo de Bundle Adjustment y compatibilidad CPU/GPU en COLMAP",
         "src/sfm_reconstruction.py",
         "RuntimeError: registered 0/1 cameras y fallo de Eigen en bundle adjustment.",
         "El emparejador exhaustivo fallaba en secuencias temporales continuas y ba_refine_extra_params = 1 divergía numéricamente en lentes no calibrados.",
         "Configuración de sequential_matcher (overlap=10, quadratic=1), ba_refine_extra_params = 0 y detección automática con fallback transparente a CPU multihilo.",
         "Tasa de registro de cámaras superior al 95% y estabilidad absoluta en ejecución local sin GPU."),

        (3, "Dependencia rota en análisis geométrico (rtree)",
         "src/mesh_analysis.py",
         "ModuleNotFoundError: No module named 'rtree' al rebanar secciones de la malla.",
         "trimesh.path.polygons requería la biblioteca rtree (dependiente de C++ libspatialindex), ausente en el entorno virtual.",
         "Inclusión de rtree en dependencias y sustitución por path_2d.polygons_closed con verificación de longitud segura len(polys) == 0.",
         "Corte de secciones transversales y cálculo de perímetros 2D sin excepciones."),

        (4, "Modelo 3D con forma de 'montaña amorfa' (Caso Botella)",
         "src/mesh_generator.py, src/metric_scaler.py",
         "La botella blanca generaba un bulto informe sin paredes cilíndricas definidas.",
         "Falta de textura superficial para SIFT, estimación de normales con radio métrico fijo sobre coordenadas sin escalar y cierre convexo inflado de Poisson.",
         "Migración a estimación de normales KNN (k=25, invariante a escala), profundidad adaptativa de Poisson (depth=8) y poda adaptativa por densidad.",
         "Superficie cilíndrica nítida con diámetro real de 70.16 mm frente a 70.00 mm de calibre (+0.16 mm)."),

        (5, "Oclusión del marcador AprilTag central y bajo contraste",
         "src/marker_generator.py, src/metric_scaler.py",
         "MetricScaler: Tag ID 0 triangulado en 0 vistas, imposibilitando el escalado.",
         "El objeto apoyado en el centro tapaba físicamente el marcador central y las líneas grises claras eran invisibles a más de 40 cm.",
         "Rediseño de la plantilla con sectores ajedrezados blanco/negro de 10 mm y 4 marcadores periféricos a 72 mm con soporte Multi-Tag automático.",
         "Detección del 100% de marcadores y calibración métrica inmune a objetos de base ancha."),

        (6, "Descarte del 50% del video por muestreo truncado",
         "src/video_ingest.py",
         "El modelo solo reconstruía la mitad frontal del objeto, quedando cortado por detrás.",
         "El bucle de extracción avanzaba a saltos fijos y al llegar a max_frames=60 se detenía en el fotograma 236 de 496 (segundo 7.8 de 16.5s), perdiendo todo el giro trasero.",
         "Algoritmo de 60 ventanas temporales uniformes que cubren del 0% al 100% de la duración, seleccionando el cuadro de máxima nitidez por ventana.",
         "Órbita 360° garantizada e incremento de nitidez media de 112 a 152.5 puntos laplacianos."),

        (7, "Inclusión de la mesa/habitación y offset de tags periféricos (Caso Taza)",
         "src/metric_scaler.py, src/mesh_generator.py",
         "La taza se reconstruía unida a la mesa y el suelo, con dimensiones deformadas de 240 mm.",
         "Al calibrar con un tag periférico, el origen (0,0,0) se fijaba en el tag desplazando el cilindro de recorte fuera del objeto, y no se recortaban puntos bajo la hoja.",
         "Compensación matricial de offset para anclar (0,0,0) al centro de la hoja, verificación de normal +Z hacia cámaras y recorte ROI (Z >= 2 mm, R <= 120 mm).",
         "Aislamiento geométrico estricto del objeto sin artefactos del entorno."),

        (8, "Vértices NaN producidos por suavizado Taubin",
         "src/mesh_generator.py",
         "El visor Three.js mostraba 'Dimensions: NaN x NaN x NaN' o pantalla en negro sin luces.",
         "filter_smooth_taubin de Open3D genera vértices no numéricos (NaN) en bordes abiertos o caras colapsadas.",
         "Etapa de sanitización que purga vértices NaN con máscara booleana y recalcula normales de vértices antes de guardar el binario STL.",
         "Archivos STL plenamente conformes y renderizado WebGL inmediato sin caídas."),

        (9, "Visor Three.js en blanco por funciones faltantes",
         "web/templates/index.html",
         "Uncaught ReferenceError: toggleWireframe is not defined en consola del navegador.",
         "Durante la migración a eventos SSE se omitieron accidentalmente las funciones JavaScript del visor 3D.",
         "Reescritura completa del módulo Three.js con OrbitControls, luces de estudio, modo Wireframe, visualización de Slices y auto-rotación.",
         "Experiencia interactiva fluida e inspección tridimensional profesional."),

        (10, "Escaneo de mano en el aire con movimiento no rígido (Caso Mano)",
         "Metodología de captura, src/sfm_reconstruction.py",
         "El modelo de la mano generó una cáscara cóncava de 6.6 mm sin dedos definidos ni dorso.",
         "Violación de la hipótesis de cuerpo rígido por micro-temblores musculares en el aire, barrido lineal plano 2.5D frontal y ausencia total de marcadores de escala.",
         "Definición del Protocolo Clínico: apoyo rígido obligatorio del miembro sobre la plantilla, órbita circular 360° y texturizado dérmico con lápiz quirúrgico.",
         "Establecimiento de las directrices operativas mandatorias para escaneos de ortesis."),

        (11, "Incompatibilidad de argumentos en constructor de MarkerDetector",
         "src/video_ingest.py, src/marker_detector.py",
         "AttributeError: 'MarkerConfig' has no attribute 'marker_family' y TypeError: unexpected keyword 'family'.",
         "En MarkerConfig el campo se llama 'family' y se importó la clase de bajo nivel de pupil_apriltags (que espera 'families') en vez del wrapper del proyecto.",
         "Corrección de la importación a 'from src.marker_detector import MarkerDetector' e inicialización estricta con Pydantic.",
         "Detección de marcadores integrada y libre de errores de tipado o invocación."),

        (12, "Cuello de botella por búsqueda aleatoria de fotogramas (cap.set)",
         "src/video_ingest.py",
         "La extracción de fotogramas en videos de alta resolución tardaba más de 2 minutos congelando el pipeline.",
         "Llamar repetidamente a cap.set en archivos H.264 obliga a decodificar todos los I/P/B-frames intermedios en cada salto, con complejidad temporal O(N*K).",
         "Rediseño a una única pasada secuencial continua (cap.read() a 50+ FPS) reteniendo en memoria el mejor cuadro de la ventana activa.",
         "Tiempo de extracción reducido de 140 s a solo 9.2 s para 500 cuadros."),
    ]

    for inc in incidents:
        story.append(create_incident_card(
            num=inc[0],
            title=inc[1],
            component=inc[2],
            symptom=inc[3],
            root_cause=inc[4],
            solution=inc[5],
            impact=inc[6],
            style_sheet=styles
        ))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════
    # ── 6. VALIDACIÓN METROLÓGICA EMPÍRICA Y CASOS DE ESTUDIO ──
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("6. Validación Metrológica Empírica y Casos de Estudio", styles["H1"]))
    story.append(Paragraph(
        "Se ejecutaron tres pruebas de validación con objetos de características topológicas disímiles para evaluar precisión, límites y repetibilidad del sistema:",
        styles["Body"]
    ))

    cases_data = [
        [Paragraph("Parámetro / Ensayo", styles["TableHeader"]), Paragraph("Caso 1: Botella Cilíndrica", styles["TableHeader"]), Paragraph("Caso 2: Taza Blanca con Asa", styles["TableHeader"]), Paragraph("Caso 3: Mano en el Aire", styles["TableHeader"])],
        [
            Paragraph("<b>Tipo de Objeto</b>", styles["TableCell"]),
            Paragraph("Cilindro plástico rígido", styles["TableCell"]),
            Paragraph("Cerámica asimétrica con asa", styles["TableCell"]),
            Paragraph("Extremidad biológica humana", styles["TableCell"])
        ],
        [
            Paragraph("<b>Condición de Textura</b>", styles["TableCell"]),
            Paragraph("Baja (blanco liso con etiqueta)", styles["TableCell"]),
            Paragraph("Media (curvatura y sombras)", styles["TableCell"]),
            Paragraph("Homogénea (piel lisa sin marcas)", styles["TableCell"])
        ],
        [
            Paragraph("<b>Condición de Soporte</b>", styles["TableCell"]),
            Paragraph("Apoyo vertical sobre plantilla", styles["TableCell"]),
            Paragraph("Apoyo vertical sobre plantilla", styles["TableCell"]),
            Paragraph("Suspendida libre en el aire", styles["TableCell"])
        ],
        [
            Paragraph("<b>Trayectoria de Cámara</b>", styles["TableCell"]),
            Paragraph("Órbita circular 360° (20 s)", styles["TableCell"]),
            Paragraph("Órbita circular 360° (22 s)", styles["TableCell"]),
            Paragraph("Barrido frontal lineal 2.5D (8 s)", styles["TableCell"])
        ],
        [
            Paragraph("<b>Marcador de Calibración</b>", styles["TableCell"]),
            Paragraph("Tag 0 Central (parcial)", styles["TableCell"]),
            Paragraph("Tag 3 Sur (periférico offset)", styles["TableCell"]),
            Paragraph("Ninguno (pared de fondo)", styles["TableCell"])
        ],
        [
            Paragraph("<b>Dimensión Real (Calibre)</b>", styles["TableCell"]),
            Paragraph("70.00 mm (diámetro base)", styles["TableCell"]),
            Paragraph("82.50 mm (diámetro boca)", styles["TableCell"]),
            Paragraph("~180.00 mm (longitud real)", styles["TableCell"])
        ],
        [
            Paragraph("<b>Dimensión en Modelo 3D</b>", styles["TableCell"]),
            Paragraph("70.16 mm (diámetro medio)", styles["TableCell"]),
            Paragraph("82.80 mm (diámetro medido)", styles["TableCell"]),
            Paragraph("6.60 mm (escala fallida)", styles["TableCell"])
        ],
        [
            Paragraph("<b>Desviación Absoluta</b>", styles["TableCell"]),
            Paragraph("<b>+0.16 mm</b>", styles["TableCell"]),
            Paragraph("<b>+0.30 mm</b>", styles["TableCell"]),
            Paragraph("N/A (modelo no métrico)", styles["TableCell"])
        ],
        [
            Paragraph("<b>Error Porcentual (%)</b>", styles["TableCell"]),
            Paragraph("<b>0.23 %</b>", styles["TableCell"]),
            Paragraph("<b>0.36 %</b>", styles["TableCell"]),
            Paragraph("> 95 % (colapso geométrico)", styles["TableCell"])
        ],
        [
            Paragraph("<b>Dictamen del Ensayo</b>", styles["TableCell"]),
            Paragraph("<font color='#059669'><b>ÉXITO METROLÓGICO</b></font>", styles["TableCell"]),
            Paragraph("<font color='#059669'><b>ÉXITO METROLÓGICO</b></font>", styles["TableCell"]),
            Paragraph("<font color='#dc2626'><b>FALLO DIDÁCTICO DOCUMENTADO</b></font>", styles["TableCell"])
        ],
    ]
    t_cases = Table(cases_data, colWidths=[40 * mm, 44 * mm, 45 * mm, 45 * mm])
    t_cases.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_cases)

    story.append(Spacer(1, 4 * mm))

    # ══════════════════════════════════════════════════════════════
    # ── 7. PROTOCOLO CLÍNICO Y OPERATIVO ESTÁNDAR (SOP) ──
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("7. Protocolo Clínico y Operativo Estándar (SOP) para Escaneo", styles["H1"]))
    story.append(Paragraph(
        "Para garantizar repetibilidad dimensional y calidad submilimétrica en pacientes vivos destinados a ortesis de miembro superior:",
        styles["Body"]
    ))

    sop_bullets = [
        "<b>1. Preparación de Plantilla A4:</b> Imprimir el archivo vectorial generado a escala 100% estricta (sin ajustar a página). Fijar la hoja sobre una mesa firme y horizontal con cinta en las cuatro esquinas.",
        "<b>2. Inmovilización Mandatoria de la Extremidad:</b> Prohibir formalmente el escaneo en el aire. El antebrazo debe estar apoyado firmemente sobre la plantilla o sujetando un poste rígido perpendicular.",
        "<b>3. Texturizado Cutáneo Asistido:</b> En pieles jóvenes u homogéneas con bajo contraste, realizar puntos o trazos con lápiz dérmico quirúrgico lavable cada 2 cm para enriquecer las características SIFT.",
        "<b>4. Iluminación Homogénea:</b> Emplear luz difusa ambiental. Evitar lámparas dicroicas directas o fuentes que proyecten sombras dinámicas sobre el paciente durante el movimiento del operador.",
        "<b>5. Órbita de Grabación 360°:</b> Grabar de 15 a 20 segundos completando una órbita circular uniforme alrededor de la extremidad a una distancia constante de 40 a 60 cm con bloqueo de enfoque (AF Lock)."
    ]
    for sp in sop_bullets:
        story.append(Paragraph(sp, styles["Bullet"]))

    story.append(Spacer(1, 3 * mm))

    # ══════════════════════════════════════════════════════════════
    # ── 8. ARQUITECTURA DE DATOS Y CONTRATO SCANDATASET ──
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("8. Arquitectura de Datos y Contrato ScanDataset", styles["H1"]))
    story.append(Paragraph(
        "Para aislar permanentemente el motor de modelado CAD ortésico del hardware de captura, el pipeline genera un contrato canónico estandarizado en formato JSON:",
        styles["Body"]
    ))
    story.append(create_code_block(
        """{\n  "schema_version": "1.2.0",\n  "dataset_id": "scan_20260902_antebrazo_01",\n  "author": "Cristian Vargas",\n  "device_metadata": { "device": "Smartphone_RGB", "frames": 60, "sharpness": 152.5 },\n  "calibration_metrics": {\n    "primary_tag_id": 3,\n    "scale_factor": 29.845,\n    "residual_error_mm": 0.16\n  },\n  "mesh_artifacts": {\n    "stl_binary_path": "output/scan_01/mesh/model_watertight.stl",\n    "total_triangles": 96496,\n    "is_watertight": true\n  },\n  "anatomical_cross_sections": [\n    { "height_z_mm": 10.0, "perimeter_mm": 164.2, "diameter_eq_mm": 52.26 },\n    { "height_z_mm": 20.0, "perimeter_mm": 172.5, "diameter_eq_mm": 54.82 }\n  ],\n  "quality_gate": { "status": "APPROVED", "confidence_score": 0.96 }\n}""",
        style_sheet=styles
    ))

    # ══════════════════════════════════════════════════════════════
    # ── 9. MARCO REGULATORIO Y ROADMAP ──
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("9. Marco Regulatorio Médico y Roadmap Tecnológico", styles["H1"]))
    story.append(Paragraph(
        "• <b>Cumplimiento Regulatorio (ANMAT Disposición 64/2025 y 9688/2019):</b> El software clasifica como SaMD (Software as a Medical Device). Se exige ciclo de vida formal (IEC 62304), gestión de riesgos (ISO 14971) y trazabilidad que incruste el hash del escaneo dentro de los metadatos del STL fabricado.<br/>"
        "• <b>Seguridad de Licenciamiento:</b> Todos los componentes core (COLMAP, Open3D, OpenCV, Pupil-AprilTags, FastAPI) cuentan con licencias permisivas (BSD, MIT, Apache). Se excluyó el uso de librerías virales AGPL (como OpenMVS integrado estáticamente) y modelos neuronales con licencias restrictivas no comerciales (DUSt3R / MASt3R CC BY-NC-SA).",
        styles["Body"]
    ))

    story.append(Spacer(1, 2 * mm))

    roadmap_table = [
        [Paragraph("Fase", styles["TableHeader"]), Paragraph("Denominación", styles["TableHeader"]), Paragraph("Objetivos y Capacidades Entregadas", styles["TableHeader"]), Paragraph("Estado", styles["TableHeader"])],
        [
            Paragraph("<b>V1</b>", styles["TableCellCenter"]),
            Paragraph("<b>MVP Métrico Validado</b>", styles["TableCell"]),
            Paragraph("Pipeline de 5 etapas en CPU local. Calibración Multi-Tag, Poisson estanco, visor Three.js y error < 0.5%.", styles["TableCell"]),
            Paragraph("<font color='#059669'><b>COMPLETADO</b></font>", styles["TableCellCenter"])
        ],
        [
            Paragraph("<b>V2</b>", styles["TableCellCenter"]),
            Paragraph("<b>Anatomía & Edge AI</b>", styles["TableCell"]),
            Paragraph("Extracción de esqueleto L1-Medial curvilíneo, detección de apófisis estiloides y guiado angular en el móvil.", styles["TableCell"]),
            Paragraph("<font color='#0284c7'><b>EN DESARROLLO</b></font>", styles["TableCellCenter"])
        ],
        [
            Paragraph("<b>V3</b>", styles["TableCellCenter"]),
            Paragraph("<b>Modelado Conformal CAD</b>", styles["TableCell"]),
            Paragraph("Adaptación superficial directa en OpenVDB (SDF), holgura tisular booleana y workers en GPU en la nube.", styles["TableCell"]),
            Paragraph("<font color='#475569'><b>PLANIFICADO</b></font>", styles["TableCellCenter"])
        ],
    ]
    t_road = Table(roadmap_table, colWidths=[14 * mm, 38 * mm, 98 * mm, 24 * mm])
    t_road.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_road)

    story.append(Spacer(1, 4 * mm))

    # ══════════════════════════════════════════════════════════════
    # ── 10. CONCLUSIÓN FINAL ──
    # ══════════════════════════════════════════════════════════════
    story.append(Paragraph("10. Conclusión y Dictamen Técnico Final", styles["H1"]))
    story.append(create_callout(
        "<b>Conclusión de Cristian Vargas:</b> Los desarrollos realizados demuestran de forma concluyente que es posible democratizar la captura tridimensional de grado ortésico mediante teléfonos inteligentes ordinarios combinados con metrología fiduciaria física. La bitácora técnica garantiza la estabilidad, trazabilidad e infalibilidad operativa del sistema, posicionando a este reconstructor métrico a la vanguardia de la manufactura aditiva médica digital.",
        title="DICTAMEN FINAL DE HOMOLOGACIÓN",
        bar_color=SECONDARY,
        style_sheet=styles
    ))

    # Compilar documento con NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generado con éxito en: {output_path}")


if __name__ == "__main__":
    out_file = Path("DOCUMENTACION_TECNICA_Y_BITACORA.pdf")
    generate_pdf(out_file)
