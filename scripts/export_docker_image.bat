@echo off
REM ==============================================================================
REM Construir y Exportar Imagen Docker a Archivo Comprimido (.tar.gz)
REM Para transferir a VPS compartido sin necesidad de compilar en el servidor
REM ==============================================================================
echo ======================================================
echo   Construyendo imagen Docker de Reconstructor 3D...
echo ======================================================

docker build -t reconstructor-3d:latest .
if %ERRORLEVEL% NEQ 0 (
    echo [!] Error al construir la imagen Docker. Asegurate de que Docker Desktop este abierto.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================
echo   Guardando y comprimiendo imagen a reconstructor-3d-image.tar...
echo ======================================================

docker save reconstructor-3d:latest -o reconstructor-3d-image.tar
if %ERRORLEVEL% NEQ 0 (
    echo [!] Error al exportar la imagen.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================
echo   Listo! Archivo generado: reconstructor-3d-image.tar
echo.
echo   Para desplegar en tu VPS:
echo   1. Sube reconstructor-3d-image.tar y docker-compose.yml al VPS:
echo      scp reconstructor-3d-image.tar docker-compose.yml root@TU_VPS_IP:/root/xkelet/
echo   2. En tu VPS, carga la imagen:
echo      docker load -i reconstructor-3d-image.tar
echo   3. Inicia el servicio:
echo      docker compose up -d
echo ======================================================
pause
