from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.db.database import init_db
from backend.api.routers import references, inference

app = FastAPI(title="Nuvant VA System", version="1.0.0")

from fastapi.staticfiles import StaticFiles
from pathlib import Path

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files - use relative path for portability (H2 fix)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/")
def read_root():
    return {"status": "ok", "system": "Nuvant Vision System"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)

app.include_router(references.router, prefix="/api/references", tags=["references"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
