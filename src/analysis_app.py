from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, status, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional, Dict
import os
import uuid
import cv2
import numpy as np
import shutil
from datetime import datetime
import json
from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
from fastapi.responses import StreamingResponse, Response
import subprocess

from database import get_db, Connection
from services.user_service import UserService
from models.user import User
from analysis_engine import FootballAnalyzer
from services.tracking_engine_client import TrackingEngineClient

# --- Configuration for JWT (RS256 - Public Key Verification Only) --- #
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INTERNAL_PIPELINE_ROOT = Path(PROJECT_ROOT).parent / "tracking_engine" / "pipeline"
LEGACY_INTERNAL_DEMO_ROOT = Path(PROJECT_ROOT).parent / "tracking_engine" / "demo"
LEGACY_EXTERNAL_DEMO_ROOT = Path(PROJECT_ROOT).parent / "demo"
ANALYSIS_OUTPUT_ROOT = Path(
    os.getenv("ANALYSIS_OUTPUT_ROOT", str(DEFAULT_INTERNAL_PIPELINE_ROOT))
).resolve()
if not ANALYSIS_OUTPUT_ROOT.exists():
    if LEGACY_INTERNAL_DEMO_ROOT.exists():
        ANALYSIS_OUTPUT_ROOT = LEGACY_INTERNAL_DEMO_ROOT.resolve()
    elif LEGACY_EXTERNAL_DEMO_ROOT.exists():
        ANALYSIS_OUTPUT_ROOT = LEGACY_EXTERNAL_DEMO_ROOT.resolve()
PUBLIC_KEY_PATH = os.path.join(PROJECT_ROOT, "certs", "public.pem")
if not os.path.exists(PUBLIC_KEY_PATH):
    PUBLIC_KEY_PATH = "certs/public.pem" # local fallback

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


async def get_current_user_from_query_token(
    access_token: str,
    db: Connection,
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(access_token, RSA_PUBLIC_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
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
CHUNK_SIZE = 1024 * 1024


def _create_analysis_run(
    run_id: str,
    match_id: str,
    input_video_name: str,
    generated_by: str,
):
    db_gen = get_db()
    db = next(db_gen)
    try:
        with db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO analysis_runs
                    (id, match_id, input_video_name, status, progress, message, generated_by)
                VALUES
                    (%s, %s, %s, 'PENDING', 0.0, %s, %s)
                """,
                (
                    run_id,
                    match_id,
                    input_video_name,
                    "Upload complete. Queued for tracking analysis.",
                    generated_by,
                ),
            )
            db.commit()
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def _update_analysis_run(
    run_id: str,
    status: str,
    progress: float,
    message: str,
    completed: bool = False,
):
    db_gen = get_db()
    db = next(db_gen)
    try:
        with db.cursor() as cursor:
            if completed:
                cursor.execute(
                    """
                    UPDATE analysis_runs
                    SET status = %s,
                        progress = %s,
                        message = %s,
                        completed_at = NOW()
                    WHERE id = %s
                    """,
                    (status, progress, message, run_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE analysis_runs
                    SET status = %s,
                        progress = %s,
                        message = %s
                    WHERE id = %s
                    """,
                    (status, progress, message, run_id),
                )
            db.commit()
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

async def run_tracking_analysis_job(
    job_id: str,
    video_path: str,
    match_id: str,
    frame_limit: int,
    skip_json: bool,
):
    """Run tracking pipeline through Tracking Engine and persist status in DB."""
    client = TrackingEngineClient()
    try:
        _update_analysis_run(job_id, "PROCESSING", 0.05, "Submitting video to tracking engine...")

        async def progress_callback(response):
            _update_analysis_run(
                job_id,
                response.status,
                float(response.progress),
                response.message,
            )

        result = await client.analyze_video(
            video_path=video_path,
            match_id=job_id,
            frame_limit=frame_limit,
            skip_json=skip_json,
            progress_callback=progress_callback,
        )

        outputs = result.get("result", {})

        # Optimize MP4 outputs for streaming and generate preview videos when possible.
        video_output_keys = [
            "tracking_video_path",
            "heatmap_video_path",
            "backline_video_path",
            "animation_video_path",
        ]
        for key in video_output_keys:
            relative_path = outputs.get(key)
            if not relative_path:
                continue
            abs_path = (ANALYSIS_OUTPUT_ROOT / relative_path).resolve()
            if not abs_path.exists():
                continue
            _optimize_mp4_faststart(abs_path)
            preview_relative = _create_preview_video(abs_path)
            if preview_relative:
                outputs[f"{key.replace('_path', '')}_preview_path"] = preview_relative

        _update_analysis_run(
            job_id,
            result.get("status", "COMPLETED"),
            float(result.get("progress", 1.0)),
            result.get("message", "Analysis complete"),
            completed=True,
        )
    except Exception as e:
        _update_analysis_run(
            job_id,
            "FAILED",
            0.0,
            str(e),
            completed=True,
        )
    finally:
        try:
            await client.close()
        except Exception:
            pass
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass


@app.get("/api/analysis_status/{analysis_id}", tags=["Analysis"])
async def get_analysis_status(analysis_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Check the status of a video analysis task from DB."""
    with db.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, match_id, input_video_name, status, progress, message, submitted_at, completed_at
            FROM analysis_runs
            WHERE id = %s AND generated_by = %s
            LIMIT 1
            """,
            (analysis_id, current_user.id),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Analysis task not found")

    return {
        "analysis_id": row["id"],
        "match_id": row.get("match_id"),
        "input_video_name": row.get("input_video_name"),
        "status": row.get("status"),
        "progress": float(row.get("progress") or 0.0),
        "message": row.get("message") or "",
        "outputs": _build_outputs_for_run(row["id"]),
        "submitted_at": row["submitted_at"].isoformat() if row.get("submitted_at") else None,
        "completed_at": row["completed_at"].isoformat() if row.get("completed_at") else None,
    }


@app.get("/api/analysis_history", tags=["Analysis"])
async def get_analysis_history(db: Connection = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Return tracking analysis history for Analyze > History page."""
    with db.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, match_id, input_video_name, status, progress, message, submitted_at, completed_at
            FROM analysis_runs
            WHERE generated_by = %s
            ORDER BY submitted_at DESC
            """,
            (current_user.id,),
        )
        rows = cursor.fetchall()

    history = []
    for row in rows:
        history.append(
            {
                "analysis_id": row["id"],
                "match_id": row.get("match_id"),
                "input_video_name": row.get("input_video_name"),
                "status": row.get("status"),
                "progress": float(row.get("progress") or 0.0),
                "message": row.get("message") or "",
                "outputs": _build_outputs_for_run(row["id"]),
                "submitted_at": row["submitted_at"].isoformat() if row.get("submitted_at") else None,
                "completed_at": row["completed_at"].isoformat() if row.get("completed_at") else None,
            }
        )
    return history


@app.delete("/api/analysis_history/{analysis_id}", tags=["Analysis"])
async def delete_analysis_history_item(
    analysis_id: str,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    with db.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM analysis_runs
            WHERE id = %s AND generated_by = %s
            """,
            (analysis_id, current_user.id),
        )
        db.commit()
        deleted = cursor.rowcount > 0

    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis history item not found")
    return {"message": "Deleted successfully"}


def _resolve_output_path(path: str) -> Path:
    relative = Path(path)
    candidate = (ANALYSIS_OUTPUT_ROOT / relative).resolve()
    output_root_resolved = ANALYSIS_OUTPUT_ROOT.resolve()
    if not str(candidate).startswith(str(output_root_resolved)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate


def _build_outputs_for_run(run_id: str) -> Dict[str, str]:
    """Build output file paths dynamically from run id (no DB path storage)."""
    candidates = {
        "tracking_video_path": f"outputs/{run_id}tracking.mp4",
        "tracking_json_path": f"outputs/{run_id}tracking.json",
        "backline_video_path": f"outputs/analytics/{run_id}backline.mp4",
        "heatmap_video_path": f"outputs/heatmaps/{run_id}heatmap.mp4",
        "possession_analysis_path": f"outputs/analytics/{run_id}possession.json",
        "animation_video_path": f"outputs/analytics/{run_id}animation.mp4",
        "heatmap_image_path": f"outputs/heatmaps/{run_id}heatmap.png",
        "all_players_grid_image_path": f"outputs/heatmaps/{run_id}all_players_grid.png",
        "movement_trail_image_path": f"outputs/analytics/{run_id}movement_trail.png",
        "possession_chart_image_path": f"outputs/analytics/{run_id}possession_analysis.png",
    }

    outputs: Dict[str, str] = {}
    for key, rel_path in candidates.items():
        abs_path = ANALYSIS_OUTPUT_ROOT / rel_path
        if abs_path.exists() and abs_path.is_file():
            outputs[key] = rel_path

    for key in [
        "tracking_video_path",
        "heatmap_video_path",
        "backline_video_path",
        "animation_video_path",
    ]:
        rel_path = outputs.get(key)
        if not rel_path:
            continue
        preview_rel = f"outputs/previews/{Path(rel_path).stem}_preview.mp4"
        preview_abs = ANALYSIS_OUTPUT_ROOT / preview_rel
        if preview_abs.exists() and preview_abs.is_file():
            outputs[f"{key.replace('_path', '')}_preview_path"] = preview_rel

    return outputs


def _optimize_mp4_faststart(file_path: Path) -> None:
    """Move MP4 metadata to the beginning for faster startup."""
    if file_path.suffix.lower() != ".mp4":
        return
    temp_path = file_path.with_name(f"{file_path.stem}.faststart{file_path.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(file_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0 and temp_path.exists():
            temp_path.replace(file_path)
    except Exception:
        # Keep original file if ffmpeg is missing or processing fails.
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def _create_preview_video(file_path: Path) -> Optional[str]:
    """Create low-resolution preview video for faster mobile playback."""
    if file_path.suffix.lower() != ".mp4":
        return None
    previews_dir = ANALYSIS_OUTPUT_ROOT / "outputs" / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    preview_name = f"{file_path.stem}_preview.mp4"
    preview_path = previews_dir / preview_name
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(file_path),
        "-vf",
        "scale=-2:480",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        "-an",
        str(preview_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and preview_path.exists():
            return f"outputs/previews/{preview_name}"
    except Exception:
        pass
    return None


def _stream_file_range(file_path: Path, start: int, end: int):
    with file_path.open("rb") as f:
        f.seek(start)
        bytes_remaining = end - start + 1
        while bytes_remaining > 0:
            read_size = min(CHUNK_SIZE, bytes_remaining)
            chunk = f.read(read_size)
            if not chunk:
                break
            bytes_remaining -= len(chunk)
            yield chunk


@app.get("/api/analysis/files", tags=["Analysis"])
async def get_analysis_file(
    path: str,
    access_token: str = "",
    db: Connection = Depends(get_db),
    current_user: Optional[User] = None,
):
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing token")
    current_user = await get_current_user_from_query_token(access_token, db)
    file_path = _resolve_output_path(path)
    return FileResponse(file_path)


@app.get("/api/analysis/stream", tags=["Analysis"])
async def stream_analysis_file(
    request: Request,
    path: str,
    access_token: str = "",
    db: Connection = Depends(get_db),
    current_user: Optional[User] = None,
):
    """Range-enabled streaming endpoint for video playback."""
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing token")
    current_user = await get_current_user_from_query_token(access_token, db)
    file_path = _resolve_output_path(path)
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    content_type = "video/mp4" if file_path.suffix.lower() == ".mp4" else "application/octet-stream"

    if not range_header:
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        }
        return StreamingResponse(
            _stream_file_range(file_path, 0, file_size - 1),
            status_code=200,
            media_type=content_type,
            headers=headers,
        )

    try:
        range_value = range_header.strip().lower().replace("bytes=", "")
        start_str, end_str = range_value.split("-", 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
    except Exception:
        return Response(status_code=416)

    if start >= file_size or end >= file_size or start > end:
        return Response(status_code=416)

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
    }
    return StreamingResponse(
        _stream_file_range(file_path, start, end),
        status_code=206,
        media_type=content_type,
        headers=headers,
    )


@app.get("/api/analysis/files/json", tags=["Analysis"])
async def get_analysis_json(
    path: str,
    access_token: str = "",
    db: Connection = Depends(get_db),
    current_user: Optional[User] = None,
):
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing token")
    current_user = await get_current_user_from_query_token(access_token, db)
    file_path = _resolve_output_path(path)
    if file_path.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Only JSON files are supported")
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON content")

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
    frame_limit: int = Form(0),
    skip_json: bool = Form(False),
    current_user: User = Depends(get_current_user),
):
    """
    Analyze uploaded video through tracking_engine pipeline.
    Does NOT auto-create a match in history.
    """
    try:
        effective_match_id = match_id or str(uuid.uuid4())
        analysis_id = str(uuid.uuid4())

        upload_dir = os.path.join(PROJECT_ROOT, "temp_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        safe_name = file.filename or "video.mp4"
        file_path = os.path.abspath(os.path.join(upload_dir, f"{analysis_id}_{safe_name}"))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        _create_analysis_run(
            run_id=analysis_id,
            match_id=effective_match_id,
            input_video_name=safe_name,
            generated_by=current_user.id,
        )

        background_tasks.add_task(
            run_tracking_analysis_job,
            analysis_id,
            file_path,
            effective_match_id,
            frame_limit,
            skip_json,
        )

        return {
            "status": "accepted",
            "message": "Upload complete. Tracking analysis started.",
            "match_id": effective_match_id,
            "analysis_id": analysis_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
