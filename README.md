# 🦿 Reconstructor Métrico 3D (RGB a STL Calibrado)

> **Digitalización 3D y metrología submilimétrica a partir de secuencias de video monocular de smartphone con marcadores AprilTag para ortesis biomédicas y prototipos.**

**Autor / Desarrollador:** **Cristian Vargas**  
**Proyecto:** Prototipo Personal de Investigación y Desarrollo  
**Versión:** 1.2.0  
**Licencia:** MIT / BSD Permisiva  

---

## 📌 Descripción General

El **Reconstructor Métrico 3D** es una solución de ingeniería de software, visión por computadora y metrología biomédica concebida para superar las limitaciones de las mediciones manuales con cinta métrica y la alta barrera de costos de los escáneres 3D clínicos industriales. 

Permite transformar un video continuo capturado con cualquier smartphone convencional (RGB monocular, sin requerir sensores LiDAR ni cámaras estéreo activas) en un **modelo tridimensional sólido estanco (*watertight*) en formato `.STL` y `.OBJ` con calibración métrica real en milímetros ($mm$)**, listo para fabricación aditiva (impresión 3D) o software CAD paramétrico.

---

## 🌟 Características Principales

* ⚡ **Ingesta Inteligente de Video a 50+ FPS:** Muestreo temporal uniforme en 60 ventanas (0% a 100% de la duración) en una sola pasada secuencial continua (`cap.read()`). Evalúa cuantitativamente el desenfoque de movimiento mediante la **Varianza del Operador Laplaciano** ($\text{Var} > 80$) para extraer únicamente los fotogramas más nítidos (*cero motion blur*).
* 🌐 **Structure from Motion (SfM) Robusto:** Motor fotogramétrico basado en **COLMAP** con emparejador secuencial temporal (`sequential_matcher`, $k=10$), triangulación epipolar y optimización global no lineal *Bundle Adjustment* con el solver Ceres (Google). Detección automática de aceleración GPU (CUDA) con fallback transparente a CPU multihilo.
* 🎯 **Calibración Métrica Multi-Tag de Alta Precisión:** Detección subpíxel de esquinas con **Pupil-AprilTags** (familia `tag36h11`). Triangula las esquinas en el espacio 3D para resolver la ambigüedad de escala monocular, aplicando traslación rígida $[R \vert t]$ con compensación de offset para centrar el origen $(0,0,0)$ en el tablero físico con vector normal $+Z$ hacia las cámaras.
* 🧊 **Mallado Poisson y Aislamiento de ROI:** Estimación de normales mediante $k$-vecinos más cercanos (**KNN**, $k=25$, invariante a escala previa). Reconstrucción estanca (*Screened Poisson Surface*, `depth = 8`), filtrado adaptativo de densidad, purga de coordenadas no numéricas (`NaN`) y recorte estricto de Región de Interés ($Z \ge 2.0\text{ mm}$ y radio horizontal $R \le 120\text{ mm}$) que extirpa la mesa y el suelo.
* 📐 **Metrología Seccional Automatizada (Slicing):** Corte ortogonal de la malla STL cada $10\text{ mm}$ mediante **Trimesh** y **Shapely** (motor C++ GEOS). Integración de contornos cerrados 2D para derivar áreas seccionales (teorema de Green/Gauss), perímetros reales continuos y diámetros circulares equivalentes ($D_{\text{eq}} = 2\sqrt{A/\pi}$).
* 🖥️ **Dashboard Web Interactivo con Telemetría SSE:** Servidor asíncrono **FastAPI** que emite eventos de progreso en tiempo real a 5 Hz vía *Server-Sent Events* (SSE). Frontend con visor 3D WebGL (**Three.js** + OrbitControls) con sombreado de estudio PBR, modo alambre (*Wireframe*), planos de corte dinámicos e historial de escaneos.
* 💻 **CLI Integrada:** Interfaz de línea de comandos (`run_cli.py`) para procesamiento por lotes (*batch*), pruebas automatizadas y generación de patrones de calibración.

---

## 📊 Validación Metrológica Empírica

El sistema fue validado experimentalmente frente a mediciones de control físico con **calibre pie de rey digital certificado**:

| Caso de Estudio | Tipo de Objeto | Dimensión Real (Calibre) | Dimensión en Modelo 3D | Desviación Absoluta | Error Relativo | Dictamen Técnico |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Caso 1: Botella** | Cilindro plástico rígido | $70.00\text{ mm}$ | $70.16\text{ mm}$ | **$+0.16\text{ mm}$** | **$0.23\%$** | **Éxito Metrológico** |
| **Caso 2: Taza Blanca** | Cerámica asimétrica con asa | $82.50\text{ mm}$ | $82.80\text{ mm}$ | **$+0.30\text{ mm}$** | **$0.36\%$** | **Éxito Metrológico** |
| **Caso 3: Mano en el Aire** | Miembro biológico en suspensión | $\sim 180.00\text{ mm}$ | $6.60\text{ mm}$ | N/A | $> 95\%$ | **Fallo Didáctico** *(base del protocolo clínico)* |

> [!NOTE]
> El Caso 3 demostró la violación de la hipótesis de cuerpo rígido debido a micro-temblores musculares y la ausencia de órbita 360°, estableciendo el protocolo mandatorio de inmovilización sobre la plantilla de calibración física.

---

## 🏗️ Arquitectura del Sistema (5 Etapas)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FLUJO DEL PIPELINE DETERMINISTA                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

   [ Etapa 1: Ingesta de Video ]
   ├── Lectura secuencial continua a 50+ FPS en memoria RAM (OpenCV).
   ├── Segmentación en 60 ventanas temporales equivalentes (0% a 100% de duración).
   └── Selección del fotograma con máxima varianza Laplaciana (Var > 80, cero blur).
            │
            ▼
   [ Etapa 2: COLMAP SfM ]
   ├── Detección y descripción de puntos clave SIFT invariantes.
   ├── Sequential Matcher con solapamiento temporal (overlap=10, cuadrático).
   └── Bundle Adjustment incremental con Ceres Solver. Fallback transparente a CPU.
            │
            ▼
   [ Etapa 3: Calibración Métrica Multi-Tag ]
   ├── Detección subpíxel de esquinas AprilTag tag36h11 (Pupil-AprilTags).
   ├── Triangulación epipolar 3D y cálculo del factor de escala (mm reales).
   └── Matriz rígida [R|t] que traslada el origen (0,0,0) al centro de la hoja con normal +Z.
            │
            ▼
   [ Etapa 4: Reconstrucción Poisson & ROI ]
   ├── Estimación de normales invariante mediante KNN (k=25).
   ├── Screened Poisson Surface Reconstruction (depth=8).
   └── Poda de densidad, sanitización de NaNs y recorte ROI (Z >= 2 mm, R <= 120 mm).
            │
            ▼
   [ Etapa 5: Rebanado y Metrología 2D ]
   ├── Slicing planar ortogonal cada 10 mm (Trimesh / Shapely GEOS).
   ├── Cálculo de polígonos cerrados, perímetros continuos y áreas de Gauss.
   └── Exportación de STL binario watertight, OBJ y archivo de reporte JSON.
            │
            ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  • Backend FastAPI: API REST asíncrona + Streaming SSE a 5 Hz (sse-starlette).                   │
│  • Frontend Three.js: Visor 3D orbital WebGL, Wireframe, Slicing e Historial de Escaneos.        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Stack Tecnológico y Procedencia de Librerías

Todas las librerías fueron seleccionadas bajo criterios de estabilidad computacional, precisión métrica y compatibilidad con licencias comerciales permisivas:

| Librería | Versión | Autor / Institución de Origen | Procedencia Oficial | Licencia |
| :--- | :---: | :--- | :--- | :---: |
| **OpenCV** | `>=4.8.0` | Gary Bradski, Vadim Pisarevsky / *Intel & OpenCV Foundation* | PyPI / `opencv/opencv-python` | Apache-2.0 |
| **Pupil-AprilTags** | `>=1.0.4` | Prof. Edwin Olson (*UMich*) / *Pupil Labs GmbH* | PyPI / `pupil-labs/apriltags` | BSD-2-Clause |
| **COLMAP** | `3.8 / 3.9` | Dr. Johannes Schönberger (*ETH Zürich*) & Jan-Michael Frahm (*UNC*) | GitHub / `colmap/colmap` (Win64) | BSD-3-Clause |
| **Open3D** | `>=0.18.0` | Qian-Yi Zhou, Jaesik Park, Vladlen Koltun / *Intel Labs* | PyPI / `isl-org/Open3D` | MIT |
| **Trimesh** | `>=4.0.0` | Michael Dawson-Haggerty & *Comunidad Python* | PyPI / `mikedh/trimesh` | MIT |
| **Shapely** | `>=2.0.0` | Sean Gillies (*Toblerity*) / *Motor C++ GEOS (PostGIS/OGC)* | PyPI / `shapely/shapely` | BSD-3-Clause |
| **FastAPI + Uvicorn** | `>=0.110 / >=0.28` | Sebastián Ramírez (`tiangolo`) / Tom Christie (*Encode*) | PyPI / `fastapi`, `uvicorn` | MIT / BSD-3 |
| **SSE-Starlette** | `>=2.0.0` | Marcelo Trylesinski / *SysMo-Teams* | PyPI / `sse-starlette` | BSD-3-Clause |
| **Three.js** | `r128` | Ricardo Cabello (`Mr.doob`) & *Comunidad WebGL* | GitHub / `mrdoob/three.js` | MIT |
| **ReportLab** | `>=4.1.0` | Andy Robinson / *ReportLab Inc. (Londres, UK)* | PyPI / `reportlab.com` | BSD-Mod |
| **NumPy & SciPy** | `>=1.26 / >=1.11` | Travis Oliphant & *NumFOCUS* | PyPI / `numpy.org`, `scipy.org` | BSD-3-Clause |

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/reconstructor-3d-rgb.git
cd reconstructor-3d-rgb
```

### 2. Configurar Entorno Virtual (Python 3.10 o 3.11 recomendado)
En Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurar COLMAP (Structure from Motion)
El sistema incluye detección automática de COLMAP en el sistema o en la carpeta `tools/colmap/`.
* Si COLMAP no está instalado en el PATH global, `src/sfm_reconstruction.py` descargará y extraerá automáticamente la versión precompilada oficial para Windows en la primera ejecución.
* También puedes descargar manualmente la versión oficial desde [COLMAP Releases](https://github.com/colmap/colmap/releases) y ubicarla en `tools/colmap/` o añadirla a tu variable de entorno `PATH`.

---

## 📄 Generación del Patrón de Calibración A4

Antes de realizar cualquier escaneo, es imprescindible generar e imprimir la plantilla física de calibración:

```powershell
python run_cli.py --generate-markers --marker-size 50.0
```

Los archivos se generarán en `data/printable_markers/`:
* `target_apriltag_id0_50mm.pdf`: **Imprimir en hoja A4 a escala 100% real (sin ajustar a página ni distorsionar márgenes)**.
* `target_apriltag_id0_50mm.png`: Versión en imagen de alta resolución.

---

## 📹 Protocolo de Captura para Escaneo Óptimo

Para asegurar reconstrucciones submilimétricas fiables:

1. **Fijación de la Plantilla:** Colocar la hoja A4 sobre una superficie horizontal plana y sujetar las esquinas con cinta adhesiva.
2. **Apoyo Rígido Mandatorio:** Apoyar firmemente el objeto o la extremidad sobre la plantilla. **Queda prohibido escanear objetos o miembros suspendidos en el aire.**
3. **Textura Visual Asistida:** En superficies lisas o reflectantes, asegurar marcas visuales o contraste para facilitar el emparejamiento SIFT.
4. **Iluminación Difusa:** Iluminación ambiental uniforme. Evitar fuentes de luz directa que arrojen sombras dinámicas o brillos especulares durante el movimiento.
5. **Órbita Circular de 360°:** Grabar durante **15 a 25 segundos** realizando una vuelta circular completa alrededor del objeto a una distancia constante de **40 a 60 cm**, manteniendo bloqueados el enfoque (**AF Lock**) y la exposición (**AE Lock**).

---

## 💻 Modos de Uso

### Opción A: Interfaz Web Interactiva (Recomendado)
Inicie el servidor web local con Three.js y telemetría en tiempo real:

```powershell
python run_web.py
```

Abra su navegador en: **`http://127.0.0.1:8000`**
1. Arrastre el video grabado a la zona de carga.
2. Ajuste los parámetros deseados (tamaño de marcador en mm, profundidad de Poisson, FPS).
3. Presione **🚀 Iniciar Reconstrucción 3D**.
4. Observe el progreso en tiempo real (0 a 100%) y el cronómetro de procesamiento.
5. Inspeccione interactivamente la malla 3D con órbita, sombreado PBR, modo *Wireframe* y planos de corte *Slices*.
6. Descargue el archivo **`.STL`** binario estanco y explore el **Historial de Proyectos**.

### Opción B: Línea de Comandos (CLI)
Para flujos automáticos o integración en servidores:

```powershell
python run_cli.py --video data/input_videos/mi_escaneo.mp4 --marker-size 50.0 --output output/prueba_01
```

Parámetros opcionales disponibles:
* `--marker-size`: Dimensión del lado del marcador en mm (por defecto: `50.0`).
* `--fps`: Tasa de extracción de cuadros por segundo (por defecto: `3.0`).
* `--min-laplacian`: Umbral de varianza del filtro de nitidez (por defecto: `80.0`).
* `--poisson-depth`: Profundidad del árbol octree de Poisson (por defecto: `9`).
* `--slice-step`: Intervalo de corte vertical en mm (por defecto: `10.0`).

### Opción C: Compilación de Documentación Técnica PDF
Para compilar la documentación técnica editorial del proyecto personal:

```powershell
python scripts/generate_tech_doc_pdf.py
```

---

## 📁 Estructura del Repositorio

```
reconstructor-3d-rgb/
├── .gitignore                                # Reglas de exclusión para Git
├── requirements.txt                          # Dependencias oficiales de Python
├── README.md                                 # Documentación general del repositorio
├── run_web.py                                # Punto de entrada para el servidor web
├── run_cli.py                                # Punto de entrada por línea de comandos
│
├── src/                                      # Núcleo del pipeline de procesamiento
│   ├── config.py                             # Clases de configuración Pydantic
│   ├── pipeline.py                           # Orquestador secuencial de 5 etapas
│   ├── video_ingest.py                       # Extractor de cuadros nítidos y Laplaciano
│   ├── marker_detector.py                    # Wrapper para Pupil-AprilTags
│   ├── marker_generator.py                   # Generador vectorial de patrones A4
│   ├── sfm_reconstruction.py                 # Wrapper de COLMAP SfM y Ceres BA
│   ├── metric_scaler.py                      # Triangulación 3D y calibrador Multi-Tag
│   ├── mesh_generator.py                     # Normales KNN, Poisson y recorte ROI
│   └── mesh_analysis.py                      # Slicing ortogonal y cálculo de perímetros
│
├── web/                                      # Aplicación Web FastAPI y Frontend
│   ├── server.py                             # API REST asíncrona y streaming SSE
│   ├── templates/
│   │   └── index.html                        # Dashboard HTML5 / Three.js WebGL
│   └── static/
│       ├── css/style.css                     # Estilos visuales corporativos
│       └── js/three.min.js                   # Biblioteca Three.js local
│
├── scripts/                                  # Scripts de generación de documentación
│   └── generate_tech_doc_pdf.py              # Compilador PDF de documentación técnica
│
├── tests/                                    # Batería de pruebas automatizadas
│   ├── test_blur_filter.py                   # Pruebas del filtro de nitidez Laplaciana
│   ├── test_marker_detector.py               # Pruebas de detección de esquinas AprilTag
│   ├── test_mesh_slicing.py                  # Pruebas de rebanado de mallas y perímetros
│   └── test_end_to_end_synthetic.py          # Prueba de integración extremo a extremo
│
├── data/                                     # Almacenamiento de patrones e inputs
│   ├── printable_markers/                    # Patrones PDF y PNG imprimibles
│   └── input_videos/                         # Carpeta para videos de entrada (.gitkeep)
│
├── tools/                                    # Herramientas binarias externas
│   └── colmap/                               # Binarios de COLMAP para Windows (.gitkeep)
│
└── output/                                   # Modelos STL y reportes generados (.gitkeep)
```

---

## 📜 Licencias y Consideraciones

* **Código del Proyecto:** Distribuido bajo licencia permisiva de código abierto (MIT / BSD).
* **Compatibilidad de Dependencias:** Todos los componentes integrados en el núcleo (OpenCV, Open3D, COLMAP, Pupil-AprilTags, FastAPI, Three.js) operan bajo licencias comerciales permisivas (BSD, MIT, Apache 2.0). Se prohíbe el enlace estático de dependencias con licencia restrictiva viral (como AGPL) o modelos con licencias no comerciales (CC BY-NC-SA).

---

**Desarrollado como proyecto personal por Cristian Vargas · 2026**
