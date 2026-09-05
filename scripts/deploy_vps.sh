#!/usr/bin/env bash
# ==============================================================================
# Script de Despliegue Rápido en VPS Compartido (Linux / Ubuntu / Debian)
# ==============================================================================
set -e

echo "======================================================"
echo "  Despliegue de Reconstructor 3D en VPS Compartido"
echo "======================================================"

# 1. Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "[!] Docker no está instalado. Instalando Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# 2. Verificar Docker Compose
if ! docker compose version &> /dev/null; then
    echo "[!] Docker Compose no encontrado. Asegurando plugin..."
    apt-get update && apt-get install -y docker-compose-plugin
fi

# 3. Crear directorios persistentes en el host
mkdir -p data/input_videos data/printable_markers output
chmod -R 777 output data

# 4. Construir y Levantar Contenedor
echo "[*] Construyendo y levantando contenedor en segundo plano..."
docker compose up -d --build

# 5. Esperar comprobación de salud
echo "[*] Esperando a que el servicio esté listo..."
sleep 5
docker compose ps

echo "======================================================"
echo "  ✅ Despliegue completado con éxito."
echo "  Aplicación accesible en: http://<TU_IP_VPS>:8000"
echo "======================================================"
