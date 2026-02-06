from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from backend.db.database import SessionLocal, Reference
from backend.core.features import FeatureExtractor
from backend.core.anomaly import AnomalyDetector
# V32 PatchCore import
try:
    from backend.core.anomaly_patchcore import PatchCoreDetector, AnomalyDetectorV32
    PATCHCORE_AVAILABLE = True
except ImportError:
    PATCHCORE_AVAILABLE = False
    print("[Warning] PatchCore V32 not available, using V31 fallback")
import cv2
import numpy as np
import time
from datetime import datetime
import json
import os
import base64

router = APIRouter()

# Simple in-memory cache for the active model
# In a production app, we might want a proper singleton or dependency injection
ACTIVE_MODEL = {
    "id": None,
    "detector": None,
    "extractor": FeatureExtractor()
}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def load_model(ref_id: int, db: Session):
    global ACTIVE_MODEL
    if ACTIVE_MODEL["id"] == ref_id and ACTIVE_MODEL["detector"] is not None:
        return ACTIVE_MODEL["detector"], ACTIVE_MODEL.get("sensitivity", 0.0), ACTIVE_MODEL.get("version", "V31")
    
    ref = db.query(Reference).filter(Reference.id == ref_id).first()
    if not ref or not ref.model_path:
        return None, 0.0, None
    
    # Try to load as V32 first, fallback to V31
    import joblib
    try:
        model_data = joblib.load(ref.model_path)
        model_version = model_data.get("version", "V31_Mahalanobis")
    except:
        model_version = "V31_Mahalanobis"
    
    if "V32" in model_version and PATCHCORE_AVAILABLE:
        detector = PatchCoreDetector()
        print(f"Loading PatchCore V32 model...")
    else:
        detector = AnomalyDetector()
        model_version = "V31"
        print(f"Loading Mahalanobis V31 model...")
    
    try:
        detector.load(ref.model_path)
        ACTIVE_MODEL["id"] = ref_id
        ACTIVE_MODEL["detector"] = detector
        ACTIVE_MODEL["version"] = model_version
        # Load sensitivity from JSON params
        sens = 0.0
        if ref.params and isinstance(ref.params, dict):
            sens = ref.params.get("sensitivity", 0.0)
        ACTIVE_MODEL["sensitivity"] = sens
        
        print(f"Loaded model for reference {ref.name} (Version: {model_version}, Sensitivity: {sens})")
        return detector, sens, model_version
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, 0.0, None

@router.websocket("/ws/{ref_id}")
async def websocket_endpoint(websocket: WebSocket, ref_id: int):
    await websocket.accept()
    
    db = SessionLocal()
    try:
        detector, sensitivity, model_version = load_model(ref_id, db)
        if not detector:
            print(f"WS Error: Model not found for ref {ref_id}")
            await websocket.send_json({"error": "Model not ready or reference not found"})
            await websocket.close()
            return
        
        is_patchcore = model_version and "V32" in model_version
        print(f"WS Active for ref {ref_id}. Version: {model_version}, Sensitivity: {sensitivity}")

        # Use the global extractor (loaded once)
        extractor = ACTIVE_MODEL["extractor"]
        
        while True:
            # Check for incoming message: could be JSON (command) or Bytes (image)
            message = await websocket.receive()
            
            if "text" in message:
                import json
                try:
                    cmd = json.loads(message["text"])
                    if cmd.get("type") == "set_sensitivity":
                        sensitivity = float(cmd.get("value", 0.0))
                        # Persist to DB so it survives refreshes
                        try:
                            db_ref = db.query(Reference).filter(Reference.id == ref_id).first()
                            if db_ref:
                                params = db_ref.params or {}
                                params["sensitivity"] = sensitivity
                                db_ref.params = params
                                db.commit()
                                print(f"WS Sensitivity persisted to DB: {sensitivity}")
                        except Exception as persistence_err:
                            print(f"Error persisting sensitivity: {persistence_err}")
                    continue
                except Exception as e:
                    print(f"WS Command Error: {e}")
                    continue

            if "bytes" not in message:
                continue

            data = message["bytes"]
            start_time = time.time()
            
            # Decode image
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
            
            # Process with Frame Quality Filter (H7)
            try:
                features = extractor.extract(img)
            except ValueError as quality_error:
                # Frame rejected by quality filter - notify client but don't crash
                await websocket.send_json({
                    "is_defect": False,
                    "score": 0.0,
                    "fps": 0,
                    "quality_warning": str(quality_error)
                })
                continue
            
            # V32 PatchCore: predict directly from image with heatmap
            # V31 Mahalanobis: predict from features
            heatmap_b64 = None
            if is_patchcore:
                # PatchCore V32: predicts from image, returns heatmap
                is_anomaly, score, heatmap = detector.predict(
                    image=img, 
                    sensitivity_offset=sensitivity
                )
                # Encode heatmap as base64 PNG for frontend
                if heatmap is not None:
                    # Convert to colormap (red = anomaly)
                    heatmap_colored = cv2.applyColorMap(
                        (heatmap * 255).astype(np.uint8), 
                        cv2.COLORMAP_JET
                    )
                    _, buffer = cv2.imencode('.png', heatmap_colored)
                    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
            else:
                # V31: predict from features
                is_anomaly, score = detector.predict(features, sensitivity_offset=sensitivity)
            
            proc_time = (time.time() - start_time) * 1000
            
            recognition = None
            if is_anomaly:
                # Memory Search: Find most similar previous defect for this reference
                try:
                    from backend.db.database import DefectLog, DefectType
                    previous_defects = db.query(DefectLog).filter(
                        DefectLog.reference_id == ref_id,
                        DefectLog.embedding != None
                    ).all()
                    
                    best_match = None
                    best_sim = -1.0 # Cosine similarity range [-1, 1]
                    
                    # Flatten all tiles (16, 2560) -> (40960,)
                    feat_vec = features.flatten()
                    norm_feat = np.linalg.norm(feat_vec)
                    
                    for prev in previous_defects:
                        if not prev.embedding: continue
                        
                        # V22: Robust reconstruction and flattening
                        prev_vec = np.array(prev.embedding).flatten()
                        
                        # SHAPE GUARD (V22 flattened compare)
                        if prev_vec.shape != feat_vec.shape:
                            continue
                            
                        # Cosine Similarity
                        sim = np.dot(feat_vec, prev_vec) / (norm_feat * np.linalg.norm(prev_vec) + 1e-9)
                        
                        if sim > best_sim:
                            best_sim = sim
                            best_match = prev
                    
                    # V25: High-Precision Match (Threshold 0.95)
                    # This prevents 'Good' cloth from being mislabeled as 'Suciedad'
                    if best_match and best_sim > 0.95:
                        # H10 fix: Use db.get() instead of deprecated .query().get()
                        dtype = db.get(DefectType, best_match.defect_type_id)
                        recognition = {
                            "label": dtype.name if dtype else "Unknown",
                            "confidence": float(best_sim)
                        }
                        print(f"Recognition HIT: {recognition['label']} ({best_sim:.4f})")
                    else:
                        if best_match:
                            print(f"Recognition MISS: Close but not enough ({best_sim:.4f})")
                except Exception as e:
                    print(f"Recognition Error: {e}")

            response = {
                "is_defect": bool(is_anomaly),
                "score": float(score),
                "fps": 1000.0 / (proc_time + 1e-1),
                "timestamp": time.time(),
                "embedding": features.tolist() if is_anomaly else None,
                "recognition": recognition,
                "heatmap": heatmap_b64,
                "model_version": model_version
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in WS: {e}")
        try:
            await websocket.close()
        except:
            pass
    finally:
        db.close()

def clear_model_cache(ref_id: int):
    """
    Clears the active model if it matches the ref_id.
    This prevents memory leaks and using stale models after deletion.
    """
    global ACTIVE_MODEL
    if ACTIVE_MODEL["id"] == ref_id:
        print(f"Clearing cache for deleted reference {ref_id}")
        ACTIVE_MODEL["id"] = None
        ACTIVE_MODEL["detector"] = None

from pydantic import BaseModel
class DefectLogRequest(BaseModel):
    reference_id: int
    defect_type: str
    score: float
    embedding: list = None # Can be list of lists (tiles)

@router.post("/log_defect")
def log_defect(item: DefectLogRequest, db: Session = Depends(get_db)):
    """Logs a defect identified by the user or system."""
    # Find defect type ID
    from backend.db.database import DefectType, DefectLog
    dtype = db.query(DefectType).filter(DefectType.name == item.defect_type).first()
    type_id = dtype.id if dtype else None
    
    new_log = DefectLog(
        reference_id=item.reference_id,
        anomaly_score=item.score,
        is_defect=1,
        defect_type_id=type_id,
        image_path="", # TODO: Save image blob if sent
        embedding=item.embedding,
        timestamp=datetime.utcnow()
    )
    db.add(new_log)
    db.commit()
    return {"status": "logged", "id": new_log.id}
