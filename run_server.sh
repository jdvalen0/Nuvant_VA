#!/usr/bin/env bash
# Nuvant VA: lanzar API + UI (FastAPI sirve backend/api/static/index.html)
# Ejecutar desde la raíz del proyecto: ./run_server.sh
cd "$(dirname "$0")"
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
