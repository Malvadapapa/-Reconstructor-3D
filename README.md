# 🦿 Reconstructor Métrico 3D (RGB a STL Calibrado)

> **Una herramienta abierta de digitalización 3D y metrología submilimétrica que transforma videos comunes de smartphone en modelos tridimensionales calibrados en milímetros reales, pensada para órtesis biomédicas, férulas y prototipado físico.**

**Estado del proyecto:** Versión `0.1.0` (Fase Pre-Beta / Investigación Aplicada)  
**Licencia:** Código abierto bajo términos permisivos (MIT / BSD / Apache-2.0)  

---

## 💡 ¿De qué se trata este proyecto?

Medir partes del cuerpo o piezas físicas para fabricar órtesis personalizadas suele plantear un dilema: las cintas métricas manuales introducen errores humanos y no capturan la curvatura anatómica real, mientras que los escáneres 3D clínicos e industriales son sumamente costosos y requieren hardware propietario.

El **Reconstructor Métrico 3D** nació para explorar una alternativa accesible: **utilizar la cámara que ya llevamos en el bolsillo**. 

Con solo grabar un video corto en órbita alrededor del objeto apoyado sobre una plantilla impresa de calibración (con marcadores AprilTag), el sistema reconstruye la geometría completa, calcula la escala milimétrica exacta y produce una **malla sólida estanca (*watertight*) en formato `.STL` y `.OBJ`**, lista para enviar a la impresora 3D o importar en cualquier software CAD.

---

## ✨ Puntos Clave del Pipeline

* 📱 **Monocular y accesible:** Funciona con cualquier teléfono convencional. No depende de sensores LiDAR, proyectores infrarrojos ni cámaras estéreo especiales.
* 💻 **Amigable con hardware estándar:** Diseñado para ejecutarse eficientemente tanto en computadoras portátiles con CPU convencional como en estaciones con aceleración gráfica opcional.
* 🎯 **Escala real en milímetros (mm):** Gracias a la triangulación de marcadores ópticos AprilTag en el espacio 3D, el modelo resultante no tiene dimensiones arbitrarias, sino medidas físicas 1:1 listas para fabricación.
* 🔍 **Filtro inteligente de fotogramas:** Analiza el desenfoque de movimiento mediante la varianza del operador Laplaciano, descartando cuadros borrosos para conservar únicamente tomas nítidas de 360°.
* 📐 **Metrología transversal automática:** Realiza rebanado (*slicing*) geométrico cada 10 mm a lo largo del eje Z, extrayendo perímetros reales, áreas de sección y diámetros equivalentes para cotejar con calibres físicos.
* 🌐 **Dashboard visual interactivo:** Incluye un servidor web local con telemetría en tiempo real (SSE) y visor 3D en el navegador con Three.js, visualización de planos de corte y exportación directa.

---

## 🏗️ Cómo Funciona por Dentro (Paso a Paso)

El flujo de reconstrucción se organiza en una cadena modular y transparente:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PIPELINE DE RECONSTRUCCIÓN                               │
└────────────────────────────────────────────────────────────────────────────────────────┘

 1. Ingesta de Video y Selección de Cuadros (OpenCV)
    └── Divide el video en ventanas de tiempo uniformes y elige el cuadro más nítido de cada
        segmento, eliminando el movimiento brusco y asegurando cobertura circular completa.
            │
            ▼
 2. Reconstrucción de Poses de Cámara y Puntos 3D (COLMAP)
    └── Encuentra puntos característicos entre tomas y resuelve la posición espacial de cada
        cámara junto con una nube de puntos dispersa inicial (Bundle Adjustment).
            │
            ▼
 3. Calibración Métrica y Alineación de Coordenadas (Pupil-AprilTags)
    └── Localiza las esquinas de los marcadores fiduciarios en las imágenes, triangula su
        posición 3D real y halla el factor de conversión exacto a milímetros, orientando el
        suelo en Z = 0.
            │
            ▼
 4. Generación de Malla y Cierre Estanco (Open3D)
    └── Estima normales de superficie y aplica reconstrucción de Poisson, recortando la región
        de interés (ROI) para aislar la pieza del plano de apoyo y generar un sólido imprimible.
            │
            ▼
 5. Análisis Geométrico Seccional (Trimesh & Shapely)
    └── Corta la pieza a intervalos milimétricos regulares para extraer medidas clínicas,
        perímetros de ajuste y volumen total en cm³.
```

---

## 📊 Ensayos Experimentales Iniciales

Durante las pruebas de calibración en laboratorio, se evaluó la precisión del reconstructor contrastando las piezas impresas y modelos 3D contra mediciones con **calibre pie de rey digital certificado**:

| Prueba | Objeto Evaluado | Dimensión Real (Calibre) | Dimensión Reconstruida | Desviación | Error Relativo | Observaciones |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Cilindro / Botella** | Plástico rígido simétrico | $70.00\text{ mm}$ | $70.16\text{ mm}$ | **$+0.16\text{ mm}$** | **$0.23\%$** | Reconstrucción métrica excelente. |
| **Taza con Asa** | Cerámica asimétrica | $82.50\text{ mm}$ | $82.80\text{ mm}$ | **$+0.30\text{ mm}$** | **$0.36\%$** | Conserva proporciones del asa y cuerpo. |
| **Mano en suspensión** | Miembro biológico sin apoyo | $\sim 180.00\text{ mm}$ | $6.60\text{ mm}$ | N/A | Invalidador | Los micro-movimientos involuntarios deforman la nube. **Obliga a usar apoyo rígido.** |

> [!TIP]
> **Aprendizaje clave:** La fotogrametría asume que el objeto permanece inmóvil respecto al entorno durante la toma. Por eso, cualquier miembro anatómico (brazo, pierna) debe estar firmemente apoyado sobre la plantilla de calibración durante los segundos de grabación.

---

## 🚀 Guía Rápida de Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/reconstructor-3d-rgb.git
cd reconstructor-3d-rgb
```

### 2. Crear y activar el entorno virtual (Python 3.10 o 3.11)
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

### 3. Instalar las dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Motor COLMAP
El sistema detecta automáticamente COLMAP si está instalado en tu sistema (`PATH`) o en la carpeta local `tools/colmap/`. Si no lo encuentra en la primera ejecución en Windows, intentará descargarlo y configurarlo de forma asistida.

---

## 📄 Preparación: Imprimir la Hoja de Calibración

Antes del primer escaneo, necesitas la hoja guía con los marcadores que le dan escala métrica al sistema:

```powershell
python run_cli.py --generate-markers --marker-size 50.0
```

Esto creará en `data/printable_markers/`:
* `target_apriltag_id0_50mm.pdf`: **Imprimir en hoja A4 a escala 100% real (sin "ajustar a página" en la impresora)**.
* `target_apriltag_id0_50mm.png`: Imagen de respaldo en alta resolución.

---

## 📹 Consejos para una Buena Captura de Video

Para obtener la mejor calidad dimensional en tu modelo:

1. **Plantilla firme:** Pega la hoja A4 a una mesa lisa con cinta en las esquinas.
2. **Objeto apoyado:** Apoya el objeto o la extremidad con firmeza sobre la hoja.
3. **Iluminación pareja:** Trabaja con luz difusa y homogénea; evita sombras duras o luces puntuales muy brillantes.
4. **Enfoque bloqueado:** En la app de cámara de tu celular, mantén presionado sobre el objeto para activar el bloqueo de enfoque y exposición (**AE/AF Lock**).
5. **Vuelta circular:** Camina o rota lentamente alrededor de la pieza completando 360° en unos **15 a 25 segundos**, a una distancia estable de unos 40 a 60 cm.

---

## 🖥️ Cómo Utilizar el Sistema

### Modo Visual e Interactivo (Recomendado)
Levanta la aplicación web local:

```powershell
python run_web.py
```

Luego abre tu navegador en **`http://127.0.0.1:8000`**:
* Arrastra el video que grabaste.
* Presiona **Iniciar Reconstrucción 3D** y sigue la barra de progreso en vivo.
* Explora la pieza en 3D (rotación, zoom, corte transversal, alambre) y descarga el archivo `.STL`.

### Modo Terminal / Línea de Comandos (CLI)
Ideal para procesar en lote o automatizar pruebas:

```powershell
python run_cli.py --video data/input_videos/mi_escaneo.mp4 --marker-size 50.0 --output output/escaneo_01
```

Opciones principales:
* `--marker-size`: Tamaño del cuadrado exterior del marcador en mm (por defecto `50.0`).
* `--fps`: Tasa de muestreo de cuadros (por defecto `3.0`).
* `--poisson-depth`: Nivel de detalle de la malla (por defecto `9`).
* `--slice-step`: Distancia entre cortes de análisis en mm (por defecto `10.0`).
* `--neural`: Activa matching denso neuronal (DISK + LightGlue) para superficies biológicas o de bajo contraste.
* `--texture`: Activa el pipeline de despliegue UV (`xatlas`) y horneado de texturas fotorrealistas (`.obj`, `.glb`).
* `--atlas-res`: Resolución del atlas de texturas (`1024`, `2048`, `4096`).

---

## 🧩 Filosofía de Dependencias y Licencias

Este proyecto respeta una regla estricta de licencias de software: **todos los componentes utilizados poseen licencias comerciales abiertas y permisivas** (MIT, Apache-2.0 o BSD). No se utilizan librerías con restricciones AGPL ni modelos con cláusulas de uso no comercial, facilitando su adopción y evolución en entornos médicos, académicos y profesionales.

---

*Proyecto desarrollado con fines de investigación aplicada y democratización de tecnologías de digitalización física en 3D.*
