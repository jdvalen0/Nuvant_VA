import os
from pathlib import Path

# Calculate project root relative to this file
# This file is in backend/config.py, so root is two levels up
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage configuration
STORAGE_DIR = os.getenv("STORAGE_DIR", BASE_DIR / "local_storage")
IMAGES_DIR = os.getenv("IMAGES_DIR", BASE_DIR.parent / "images") # Assuming images is in project root parent for dev, or configured

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/db/nuvant.db")

# Ensure directories exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "db", exist_ok=True)

def get_storage_path(ref_id: int) -> Path:
    return Path(STORAGE_DIR) / str(ref_id)
