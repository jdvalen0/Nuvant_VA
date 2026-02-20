# Industrial Fabric Inspection System - Task List

## Phase 1: Architecture & Design [Completed]
- [x] Define System Architecture (Software + Data Flow) <!-- id: 1 -->
- [x] Design Database Schema (References, Defect Logs) <!-- id: 2 -->
- [x] Define Statistical Algorithm Pipeline (Feature Extraction -> Anomaly Detection) <!-- id: 3 -->
- [x] Create Implementation Plan <!-- id: 4 -->

## Phase 2: Core Algorithm Implementation (Backend) [Completed]
- [x] Implement Image Preprocessing & ROI Selection <!-- id: 5 -->
- [x] **MIGRATION**: Replace LBP/GLCM with MobileNetV2 Embeddings <!-- id: 6 -->
- [x] Implement PaDiM-like Anomaly Detection (Multivariate Gaussian) <!-- id: 7 -->
- [x] Implement Real-time Inference Engine <!-- id: 8 -->

## Phase 3: Data Management & API [In Progress]
- [x] Setup Local Database (SQLite/PostgreSQL) <!-- id: 9 -->
- [x] Create API for Reference Management (CRUD) <!-- id: 10 -->
- [x] Create API for Inference Results & Video Stream <!-- id: 11 -->
- [x] **[NEW]** API for Reference Deletion (Cleanup) <!-- id: 12 -->
- [x] **[NEW]** API for Defect Labeling/Tagging <!-- id: 13 -->

## Phase 4: User Interface (Frontend) [In Progress]
- [x] **PROTOTYPE**: Single Page HTML/JS for immediate validation <!-- id: 14 -->
- [x] Implement "Delete Reference" Button & Confirmation <!-- id: 15 -->
- [x] Implement Dropdown for Defect Labeling <!-- id: 16 -->
- [x] Implement History View & Clean up <!-- id: 17 -->

## Phase 5: Integration & Validation
- [ ] Integration Testing (Frontend <-> Backend) <!-- id: 15 -->
- [ ] Performance Validation (FPS, Latency) <!-- id: 16 -->
- [x] User Documentation & Deployment Guide <!-- id: 17 -->
- [x] **[NUEVO]** TECHNICAL_MASTER_PLAN.md: Arquitectura, IA y Modelo de Datos (Riguroso) <!-- id: 18 -->
- [x] **[NUEVO]** Profundización Científica y Justificación Industrial (Documentación Pro) <!-- id: 25 -->

## Phase 7: Protocolo de Pruebas y Validación de Precisión [NEW]
- [ ] Generar Script "Test Bench": Selección de muestras (N=50) <!-- id: 24 -->
- [ ] Implementar "Adulteración Sintética": Generación de defectos controlados (pixeles, manchas, ruido) <!-- id: 25 -->
- [ ] Reporte de Matriz de Confusión: Validación de precisión (True Positives vs False Positives) <!-- id: 26 -->
- [x] **FEATURE**: Implementar Clasificador por Memoria (Identificar defectos ya conocidos) <!-- id: 27 -->

## Phase 8: Optimización de Sensibilidad y Estabilidad [Completed]
- [x] Implementar sensibilidad dinámica en `AnomalyDetector` <!-- id: 29 -->
- [x] Persistencia de hiperparámetros en DB (`Reference.params`) <!-- id: 30 -->
- [x] Controles de sensibilidad en la UI (Slider de Rigor) <!-- id: 31 -->
- [ ] Validación cruzada con Test Bench (Detección de 5x5 px) <!-- id: 32 -->
- [ ] Auditoría de bugs en capa de reconocimiento (Memoria) <!-- id: 33 -->
