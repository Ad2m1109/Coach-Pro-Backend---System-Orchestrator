from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, status, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from typing import Optional, Dict, Any
import os
import uuid
import tempfile
import cv2
import numpy as np

from database import get_db, Connection
from services.user_service import UserService
from models.user import User
import analysis_pb2

# --- Configuration for JWT (Synced with app.py) --- #
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

async def get_current_user(token: str = Depends(os.environ.get('OAUTH2_SCHEME', 'api/token')), db: Connection = Depends(get_db)):
    # Simple auth check (simplified for analysis service)
    # Ideally should share a common auth library
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        # Note: Analysis app might be behind a gateway or handle its own token validation
        # For simplicity, we'll keep it similar to app.py
        pass 
    except JWTError:
        raise credentials_exception
    return None # Placeholder or implement full check

app = FastAPI(
    title="Football Analysis Management Service",
    description="Dedicated service for managing video analysis tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the FootballAnalyzer
single_image_analyzer = FootballAnalyzer()

# In-memory store for analysis status
analysis_status: Dict[str, Dict[str, Any]] = {}

async def run_streaming_analysis(video_path: str, db: Connection, match_id: str):
    """Phase 1: Stream chunks to gRPC and track progress."""
    from services.analysis_grpc_service import AnalysisGrpcService
    grpc_service = AnalysisGrpcService()
    
    def chunk_generator():
        calib_path = os.environ.get('CALIB_PATH', '')
        if not calib_path:
            candidate_calib = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Analysis', 'calibration.yaml'))
            if os.path.exists(candidate_calib):
                calib_path = candidate_calib

        model_path = os.environ.get('MODEL_PATH', '')
        if not model_path:
            candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Analysis', 'yolov8m.onnx'))
            if os.path.exists(candidate):
                model_path = candidate

        with open(video_path, "rb") as f:
            chunk_idx = 0
            while True:
                data = f.read(1024 * 1024) # 1MB chunks
                if not data:
                    break
                yield analysis_pb2.VideoChunk(
                    data=data,
                    match_id=match_id,
                    calibration_path=calib_path,
                    model_path=model_path,
                    chunk_index=chunk_idx,
                    is_last_chunk=False
                )
                chunk_idx += 1
            
            yield analysis_pb2.VideoChunk(
                data=b"",
                match_id=match_id,
                chunk_index=chunk_idx,
                is_last_chunk=True
            )

    try:
        analysis_status[match_id] = {"status": "streaming", "progress": 0.0}
        
        responses = grpc_service.stream_analysis(chunk_generator())
        for response in responses:
            analysis_status[match_id].update({
                "status": response.status.lower(),
                "progress": response.progress,
                "message": response.message
            })
            
            # Phase 2: Handle metrics here
            # if response.metrics:
            #     persist_metrics_batch(response.metrics, db)

    except Exception as e:
        analysis_status[match_id] = {
            "status": "failed",
            "progress": 0.0,
            "error": str(e)
        }
    finally:
        grpc_service.close()
        if os.path.exists(video_path):
            os.remove(video_path)

@app.get("/api/analysis_status/{match_id}", tags=["Analysis"])
async def get_analysis_status(match_id: str):
    """Check the status of a video analysis task."""
    if match_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis task not found")
    return analysis_status[match_id]

@app.post("/api/detect", tags=["Analysis"])
async def detect_objects_in_image(file: UploadFile = File(...)):
    """Analyzes a single image for player and ball detection."""
    contents = await file.read()
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detected_objects = single_image_analyzer.analyze_single_image(image)
    for obj in detected_objects:
        if "color" in obj and isinstance(obj["color"], tuple):
            obj["color"] = list(obj["color"])

    return {"filename": file.filename, "detections": detected_objects}

@app.post("/api/analyze_match", tags=["Analysis"])
async def analyze_match_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    match_id: Optional[str] = Form(None),
    db: Connection = Depends(get_db)
):
    """Analyzes a full football match video asynchronously."""
    effective_match_id = match_id or str(uuid.uuid4())
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            video_path = tmp_file.name
        
        analysis_status[effective_match_id] = {"status": "pending", "progress": 0.0}
        background_tasks.add_task(run_streaming_analysis, video_path, db, effective_match_id)
        
        return {
            "status": "accepted",
            "message": "Analysis started in background",
            "match_id": effective_match_id,
            "analysis_id": effective_match_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
