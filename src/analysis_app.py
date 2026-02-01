from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, status, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional, Dict, Any
import os
import uuid
import tempfile
import cv2
import numpy as np
import shutil

from database import get_db, Connection
from services.user_service import UserService
from models.user import User
from analysis_engine import FootballAnalyzer
import analysis_pb2

# --- Configuration for JWT (RS256 - Public Key Verification Only) --- #
import os

if os.path.exists("certs/public.pem"):
    PUBLIC_KEY_PATH = os.path.join("certs", "public.pem")
elif os.path.exists("../certs/public.pem"):
    PUBLIC_KEY_PATH = os.path.join("../certs", "public.pem")
else:
    PUBLIC_KEY_PATH = "/home/ademyoussfi/Desktop/Projects/football-coach/backend/certs/public.pem"

with open(PUBLIC_KEY_PATH, "r") as f:
    RSA_PUBLIC_KEY = f.read()

ALGORITHM = "RS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

async def get_current_user(token: str = Depends(oauth2_scheme), db: Connection = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(token, RSA_PUBLIC_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        # Optimization: We can also access payload.get("club_id") or payload.get("role") here
        # to enforce authorization without any database lookups in this tier.
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user_service = UserService(db)
    user = user_service.get_user_by_email(email=email)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    return current_user

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


async def update_report_status(match_id_or_report_id, status_str, progress=0.0, message="", live_stats=None):
    """Helper to update analysis_reports in DB. Acquires a fresh connection."""
    import json
    
    # Manually handle the get_db generator since it's not a context manager
    db_gen = get_db()
    db = next(db_gen)
    try:
        with db.cursor() as cursor:
            # We find the latest report for this match or by report id
            report_data = {
                "status": status_str.upper(),
                "progress": progress,
                "message": message,
                "timestamp": "now"
            }
            if live_stats:
                report_data["live_stats"] = live_stats
                
            sql = "UPDATE analysis_reports SET report_data = %s WHERE match_id = %s OR id = %s"
            cursor.execute(sql, (json.dumps(report_data), match_id_or_report_id, match_id_or_report_id))
            db.commit()
    finally:
        # Ensure connection is closed
        try:
            next(db_gen)
        except StopIteration:
            pass

async def run_streaming_analysis_from_disk(file_path: str, db_ignored: Connection, match_id: str, calib_path: str, model_path: str):
    """Phase 2: Relay file chunks from disk to gRPC and track progress."""
    from services.analysis_grpc_service import AnalysisGrpcService
    from services.player_match_statistics_service import PlayerMatchStatisticsService
    from models.player_match_statistics import PlayerMatchStatisticsCreate
    import json

    grpc_service = AnalysisGrpcService()
    # PMS Service isn't strictly needed here as we do raw SQL upserts for performance,
    # but let's initialize it correctly if we were to use it.
    # pms_service = PlayerMatchStatisticsService(db) # REMOVED to fix NameError
    
    async def chunk_generator():
        chunk_idx = 0
        with open(file_path, "rb") as f:
            while True:
                data = f.read(1024 * 1024) 
                if not data:
                    break
                yield analysis_pb2.VideoChunk(
                    data=data, match_id=match_id, calibration_path=calib_path,
                    model_path=model_path, chunk_index=chunk_idx, is_last_chunk=False
                )
                chunk_idx += 1
        yield analysis_pb2.VideoChunk(data=b"", match_id=match_id, chunk_index=chunk_idx, is_last_chunk=True)

    player_state = {} # tracking last position and DB record ID
    frame_counter = 0

    try:
        await update_report_status(match_id, "PROCESSING", 0.0, "Starting gRPC stream...")
        
        responses = grpc_service.stream_analysis(chunk_generator())
        async for response in responses:
            frame_counter += 1
            
            if response.metrics:
                for m in response.metrics:
                    pid = str(m.player_id)
                    new_pos = (m.x, m.y)
                    if pid in player_state:
                        last = player_state[pid]["last_pos"]
                        d = ((new_pos[0]-last[0])**2 + (new_pos[1]-last[1])**2)**0.5
                        player_state[pid]["dist"] += d
                        player_state[pid]["last_pos"] = new_pos
                    else:
                        player_state[pid] = {"dist": 0, "last_pos": new_pos, "db_id": None}

            # Incremental DB update every 100 iterations (approx 500 frames due to throttling in C++)
            if frame_counter % 20 == 0:
                live_summary = {pid: {"distance": round(s["dist"], 2)} for pid, s in player_state.items()}
                await update_report_status(match_id, "PROCESSING", response.progress, response.message, live_summary)
                
                # Batch upsert to player_match_statistics
                for pid, s in player_state.items():
                    stat_create = PlayerMatchStatisticsCreate(
                        match_id=match_id, player_id=pid,
                        distance_covered_km=s["dist"] / 1000.0,
                        notes=json.dumps({"realtime": True, "frame": frame_counter}),
                        minutes_played=0
                    )
                    try:
                        # Simple logic: if db_id is None, create; but actually we should use an upsert
                        # For now, we'll use a raw SQL UPSERT for performance
                        db_gen = get_db()
                        fresh_db = next(db_gen)
                        try:
                            with fresh_db.cursor() as cursor:
                                sql = """INSERT INTO player_match_statistics (id, match_id, player_id, distance_covered_km, notes) 
                                         VALUES (UUID(), %s, %s, %s, %s) 
                                         ON DUPLICATE KEY UPDATE distance_covered_km = %s, notes = %s"""
                                cursor.execute(sql, (match_id, pid, stat_create.distance_covered_km, stat_create.notes, 
                                                     stat_create.distance_covered_km, stat_create.notes))
                                fresh_db.commit()
                        finally:
                            try: next(db_gen)
                            except StopIteration: pass
                    except: pass

        await update_report_status(match_id, "COMPLETED", 1.0, "Analysis complete")

    except Exception as e:
        await update_report_status(match_id, "FAILED", 0.0, f"Error: {str(e)}")
    finally:
        await grpc_service.close()
        if os.path.exists(file_path):
            try: os.remove(file_path)
            except: pass

@app.get("/api/analysis_status/{match_id}", tags=["Analysis"])
async def get_analysis_status(match_id: str, db: Connection = Depends(get_db)):
    """Check the status of a video analysis task from the database."""
    import json
    with db.cursor() as cursor:
        sql = "SELECT report_data FROM analysis_reports WHERE match_id = %s ORDER BY generated_at DESC LIMIT 1"
        cursor.execute(sql, (match_id,))
        report = cursor.fetchone()
        if not report:
            raise HTTPException(status_code=404, detail="Analysis task not found")
        
        data = report['report_data']
        if isinstance(data, str):
            data = json.loads(data)
        return data

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
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Analyzes a full football match video synchronously for upload, then async for analysis."""
    effective_match_id = match_id
    try:
        from services.match_service import MatchService
        from services.analysis_report_service import AnalysisReportService
        from services.team_service import TeamService
        from models.match import MatchCreate, MatchStatusEnum
        from models.analysis_report import AnalysisReportCreate
        from models.team import TeamCreate
        from datetime import datetime

        match_service = MatchService(db)
        report_service = AnalysisReportService(db)
        team_service = TeamService(db)

        # 1. Get user teams to associate the match
        teams = team_service.get_all_teams(current_user.id)
        if not teams:
            # Create a default team if none exists
            team_create = TeamCreate(name="My Team", primary_color="#000000", secondary_color="#FFFFFF")
            default_team = team_service.create_team(team_create, current_user.id)
            home_team_id = default_team.id
        else:
            home_team_id = teams[0].id

        # 2. Create Match if it doesn't exist
        user_team_ids = [t.id for t in teams]
        if not effective_match_id:
            # Create a placeholder match
            match_create = MatchCreate(
                home_team_id=home_team_id,
                away_team_id=home_team_id, # Self match for placeholder
                date_time=datetime.now(),
                venue="Local Pitch",
                status=MatchStatusEnum.live
            )
            created_match = match_service.create_match(match_create, user_team_ids)
            effective_match_id = created_match.id

        # 3. Create Analysis Report immediately (QUEUED state)
        report_create = AnalysisReportCreate(
            match_id=effective_match_id,
            report_type="Video Analysis",
            report_data={"status": "QUEUED", "progress": 0.0, "timestamp": datetime.now().isoformat()},
            generated_by=current_user.id
        )
        report_service.create_analysis_report(report_create, user_team_ids)

        # 4. Save file to disk because UploadFile might be closed after request returns
        upload_dir = "temp_uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{effective_match_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 5. Start async analysis from the saved file
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
        
        background_tasks.add_task(run_streaming_analysis_from_disk, file_path, db, effective_match_id, calib_path, model_path)
        
        return {
            "status": "accepted",
            "message": "Upload complete, analysis record created in history.",
            "match_id": effective_match_id,
            "analysis_id": effective_match_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
