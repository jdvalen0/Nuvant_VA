# Dockerfile para Nuvant Vision System V32
# Optimizado para producción con PatchCore (PyTorch + OpenCV)

FROM python:3.11-slim as base

# Evitar prompts durante instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para OpenCV y PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar solo requerimientos primero para cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
# Nota: Usamos --no-cache-dir para reducir tamaño de imagen
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Crear directorios para almacenamiento persistente y base de datos
RUN mkdir -p local_storage db logs

# Asegurar permisos (opcional si se usa root, pero buena práctica)
RUN chmod -R 777 local_storage db logs

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando para iniciar la aplicación (Producción)
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
