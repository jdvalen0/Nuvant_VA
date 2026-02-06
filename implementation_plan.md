# Plan de Implementación: Sistema de Inspección Textil (Deep Learning Híbrido)

Este documento detalla la arquitectura actual (actualizada a Deep Learning) y los pasos futuros para la gestión avanzada de defectos.

## 1. Arquitectura del Sistema (ACTUALIZADA)

Hemos migrado de un enfoque puramente estadístico (LBP) a un enfoque de **Embeddings de Deep Learning** para mayor robustez industrial.

### Componentes Principales
1.  **Core de Visión (Backend - Python)**:
    *   **Feature Extractor**: **MobileNetV2** (pre-entrenada en ImageNet). Extrae "embeddings" (vectores numéricos) que representan la textura de la tela.
    *   **Detector de Anomalías**: **Distancia de Mahalanobis (Robust Covariance)**. Modela la distribución estadística de los embeddings "normales".
    *   **Inferencia**: Tiempo real vía WebSockets.
2.  **Base de Datos (Local - SQLite)**:
    *   Gestión de Referencias y Modelos serializados (`.pkl`).
    *   **[NUEVO]** Catálogo de Tipos de Defecto (ej: Mancha, Hueco, Hilo Roto).
    *   **[NUEVO]** Historial de Defectos con etiquetado manual.
3.  **Interfaz de Operador (Frontend - Prototipo mejorado)**:
    *   **Gestión**: Crear/Borrar Referencias.
    *   **Producción**: Semáforo OK/NOK en tiempo real.
    *   **[NUEVO]** Clasificación: Dropdown para etiquetar defectos detectados.
    *   **[NUEVO]** Historial: Ver y borrar alertas pasadas.

### Flujo de Datos Actualizado
```mermaid
graph TD
    Cam[Cámara / Archivo] -->|Resize 224x224| MobileNet[MobileNetV2 (No-Train)]
    MobileNet -->|Embedding (1280d)| PCA[PCA Reducción]
    PCA -->|Vector| Mahalanobis{Distancia Mahalanobis}
    
    Mahalanobis -->|Score > Umbral| Alerta[ALERTA DEFECTO]
    Mahalanobis -->|Score < Umbral| OK[TELA OK]
    
    Alerta -->|Socket| UI[Interfaz Operario]
    UI -->|Operario Etiqueta| DB[(Base de Datos)]
```

## 2. Estrategia Algorítmica (Few-Shot Learning)

*   **Entrenamiento**: Requiere ~10-20 imágenes de "Tela Buena".
*   **Aumentación**: No necesaria gracias a la robustez de la red neuronal.
*   **Explicabilidad**: El score de anomalía indica "cuán lejos" está la textura actual de lo aprendido.

## 3. Nuevos Requerimientos de UI/UX

### Gestión de Referencias
*   **Eliminar Referencia**: Borrar modelo y datos asociados de telas que ya no se producen.
*   **Limpieza**: Botón para purgar historial de pruebas.

### Gestión de Defectos (Labeling)
*   Cuando aparece un defecto, el operario debe poder seleccionar de una lista:
    *   *Mancha de Aceite*
    *   *Rotura de Trama*
    *   *Destonificado*
    *   *Otro*
*   Esta información se guarda para futuros análisis (Pareto de defectos).

## 4. Stack Tecnológico

*   **Core**: PyTorch, Scikit-Learn, OpenCV.
*   **Backend**: FastAPI, SQLAlchemy.
*   **Frontend**: HTML/JS (Prototipo) -> React (Fase final).

## 5. Plan de Ejecución Restante

1.  **Backend - CRUD**: Endpoints para `DELETE /references/{id}` y gestión de etiquetas de defecto.
2.  **Base de Datos**: Tabla `DefectTypes` y relaciones.
3.  **Frontend**:
    *   Añadir botón "Basura" en lista de referencias.
    *   Añadir Dropdown en el overlay de defecto detectado.
    *   Panel de "Últimos Defectos" con opción de borrar.

