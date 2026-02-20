# Nuvant Vision System V32.5 Gold 🚀🏭

Sistema de inspección de telas de alta precisión basado en el algoritmo **PatchCore (arXiv:2106.08265)** y arquitectura híbrida.

## 🌟 Características Principales
- **Precisión Científica**: Implementa Neighborhood Aggregation y Density Reweighting.
- **Localización de Fallos**: Generación de Heatmaps en tiempo real mediante WebSockets.
- **Arquitectura Híbrida**: Combina Deep Learning (V32) con motores estadísticos Mahalanobis (V31).
- **Industrial Ready**: Dockerizado para despliegue inmediato en Ubuntu/Debian.

---

## 🚀 Guía de Instalación (Nuevo Hardware Ubuntu)

La mejor forma de mover el sistema a un hardware nuevo es mediante **GitHub + Docker**. Esto garantiza que todas las dependencias (PyTorch, OpenCV, etc.) se instalen correctamente sin conflictos.

### 1. Requisitos Previos
- Ubuntu 22.04 LTS o superior.
- Docker & Docker Compose instalados.
- Git instalado.

### 2. Clonación y Despliegue
En la nueva terminal del hardware Ubuntu, ejecute:

```bash
# 1. Clonar el repositorio
git clone <URL_DE_TU_REPOSITORIO_AQUÍ>
cd Nuvant_VA

# 2. Iniciar el sistema con Docker (Modo Producción)
docker-compose up -d --build
```

El sistema estará disponible automáticamente en `http://localhost:8000/static/index.html`.

---

## 🛠️ Estructura del Proyecto
- `backend/`: Núcleo de IA y API FastAPI.
- `docker/`: Configuración de contenedores e infraestructura.
- `docs/`: Manuales técnicos y protocolos de prueba.
- `scripts/`: Herramientas de auditoría y diagnóstico.

---

## 🛡️ Notas de Persistencia
Los modelos entrenados y los registros de defectos se guardan automáticamente en los directorios locales `local_storage/` y `db/`, los cuales están vinculados al contenedor. **Asegúrese de realizar copias de seguridad de estos directorios periódicamente.**

---
**Nuvant VA: Tecnología de Vanguardia para la Industria Textil.**
