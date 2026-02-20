# Nuvant Vision System - Production Dockerfile V32.5++
# Base Image: Python 3.10 Slim (Debian) for stability and small size
FROM python:3.10-slim as builder

# Set environment to prevent python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
# Ensure we have pip updated
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Final Stage: Runtime Image
FROM python:3.10-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime libs for OpenCV (GL/Glib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Working Directory
WORKDIR /app

# Copy Application Code
COPY backend /app/backend
# Copy Data Directory structure (empty, for volume mount)
RUN mkdir -p /app/backend/data /app/backend/db

# Expose API Port
EXPOSE 8000

# Copy Entrypoint Script or use default command
# Default: Run Uvicorn Server with production settings
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
