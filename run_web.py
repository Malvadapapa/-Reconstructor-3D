"""
Entry point to run the FastAPI web server with Uvicorn.
Run with: python run_web.py
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  Iniciando Servidor Web del Reconstructor 3D")
    print("  Abrir en navegador: http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=True)

