# ==============================================================================
# Reconstructor 3D Metric Model - Dockerfile (Optimized for Shared VPS / CPU)
# Base: Debian 12 (Bookworm) Slim with Python 3.11
# ==============================================================================
FROM python:3.11-slim-bookworm

# Metadata
LABEL maintainer="Cristian Vargas"
LABEL description="RGB Video to Calibrated 3D STL/GLB Reconstructor"

# Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=2 \
    MALLOC_ARENA_MAX=2 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

# 1. System Dependencies (COLMAP, OpenCV, Open3D, FFmpeg & build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    colmap \
    ffmpeg \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch CPU directly (avoids downloading 2.5 GB CUDA wheel on VPS)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Install Project Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Application Source Code
COPY src/ /app/src/
COPY web/ /app/web/
COPY scripts/ /app/scripts/
COPY run_web.py run_cli.py /app/

# 5. Create Persistent Data Directories
RUN mkdir -p /app/data/input_videos /app/data/printable_markers /app/output

# 6. Expose Web Port
EXPOSE 8000

# 7. Container Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 8. Start Application with Uvicorn
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]

