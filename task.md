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
- [ ] **[NEW]** API for Reference Deletion (Cleanup) <!-- id: 12 -->
- [ ] **[NEW]** API for Defect Labeling/Tagging <!-- id: 13 -->

## Phase 4: User Interface (Frontend) [In Progress]
- [/] **PROTOTYPE**: Single Page HTML/JS for immediate validation <!-- id: 14 -->
- [ ] Implement "Delete Reference" Button & Confirmation <!-- id: 15 -->
- [ ] Implement Dropdown for Defect Labeling <!-- id: 16 -->
- [ ] Implement History View & Clean up <!-- id: 17 -->

## Phase 5: Integration & Validation
- [ ] Integration Testing (Frontend <-> Backend) <!-- id: 15 -->
- [ ] Performance Validation (FPS, Latency) <!-- id: 16 -->
- [ ] User Documentation & Deployment Guide <!-- id: 17 -->
