from env_loader import load_backend_env

load_backend_env()

from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, status, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional, Dict
import os
import uuid
import cv2
import asyncio
import numpy as np
import shutil
from datetime import datetime
import json
from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
from fastapi.responses import StreamingResponse, Response
import subprocess

from analysis_status import (
    ANALYSIS_STATUS_PENDING,
    ANALYSIS_STATUS_PROCESSING,
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_QUEUED,
    TERMINAL_STATUSES,
    normalize_status,
)

from database import get_db, Connection
from services.user_service import UserService
from models.user import User
from analysis_engine import FootballAnalyzer
from services.tracking_engine_client import TrackingEngineClient
from services.flow_analysis_service import FlowAnalysisService

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

from security.jwt_keys import get_jwt_public_key

RSA_PUBLIC_KEY = get_jwt_public_key()

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

def _extract_bearer_token_from_header(request: Request) -> str:
    auth = request.headers.get("authorization") or ""
    if not auth:
        return ""
    parts = auth.split(" ", 1)
    if len(parts) != 2:
        return ""
    if parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()


async def get_current_user_flexible(
    request: Request,
    access_token: str,
    db: Connection,
) -> User:
    """
    Prefer Authorization header; fall back to access_token query param.
    This helps avoid leaking tokens in URLs (logs/history/proxies).
    """
    token = _extract_bearer_token_from_header(request) or (access_token or "")
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    return await get_current_user_from_query_token(token, db)

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    return current_user

app = FastAPI(
    title="Football Analysis Management Service",
    description="Dedicated service for managing video analysis tasks.",
    version="1.0.0",
)


@app.get("/health", tags=["health"], summary="Health check")
async def health_check():
    """
    Lightweight health endpoint for Docker/load-balancer checks.
    Future: add Redis liveness when SSE/WebSocket fan-out moves to Redis.
    """
    return {"status": "ok", "service": "analysis"}


cors_origins_raw = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
cors_origins = (
    [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
    if cors_origins_raw
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    # If origins is '*', credentials must be disabled (browsers reject '*' + credentials).
    allow_credentials=False if cors_origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the FootballAnalyzer
single_image_analyzer = FootballAnalyzer()
CHUNK_SIZE = 1024 * 1024

# In-memory tasks for cancellation
_active_analysis_tasks = {}

# Register segment controller for SSE and REST segment endpoints
from controllers.segment_controller import router as segment_router
from controllers.tracking_profile_controller import router as tracking_profile_router
app.include_router(segment_router, prefix="/api", tags=["Analysis Segments"])
app.include_router(tracking_profile_router, prefix="/api", tags=["Tracking Profiles"])


@app.get('/healthz', tags=['Health'])
async def health_check():
    return {
        "status": "ok",
        "service": "analysis-backend",
        "tracking_engine": os.environ.get("TRACKING_ENGINE_HOST", "localhost") + ":" + str(os.environ.get("TRACKING_ENGINE_PORT", 50051)),
    }


@app.on_event("startup")
async def startup_event():
    """Ensure database schema is up-to-date on startup."""
    db_gen = get_db()
    db = next(db_gen)
    try:
        _ensure_analysis_run_columns(db)
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def _ensure_analysis_run_columns(db: Connection):
    """Add optional columns for resilient retry semantics.
    This runs once and ignores errors if columns exist.
    """
    with db.cursor() as cursor:
        try:
            cursor.execute("ALTER TABLE analysis_runs ADD COLUMN input_video_path TEXT NULL")
            db.commit()
        except Exception:
            db.rollback()
        try:
            cursor.execute("ALTER TABLE analysis_runs ADD COLUMN frame_limit INT NULL")
            db.commit()
        except Exception:
            db.rollback()
        try:
            cursor.execute("ALTER TABLE analysis_runs ADD COLUMN skip_json BOOLEAN NULL")
            db.commit()
        except Exception:
            db.rollback()
        try:
            cursor.execute("ALTER TABLE analysis_runs ADD COLUMN is_cancelled BOOLEAN DEFAULT FALSE")
            db.commit()
        except Exception:
            db.rollback()


def _create_analysis_run(
    run_id: str,
    match_id: str,
    input_video_name: str,
    input_video_path: str,
    generated_by: str,
    frame_limit: int = 0,
    skip_json: bool = False,
):
    db_gen = get_db()
    db = next(db_gen)
    try:
        _ensure_analysis_run_columns(db)
        with db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO analysis_runs
                    (id, match_id, input_video_name, input_video_path, status, progress, message, generated_by, frame_limit, skip_json)
                VALUES
                    (%s, %s, %s, %s, %s, 0.0, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    match_id,
                    input_video_name,
                    input_video_path,
                    ANALYSIS_STATUS_PENDING,
                    "Upload complete. Queued for tracking analysis.",
                    generated_by,
                    frame_limit,
                    skip_json,
                ),
            )
            db.commit()
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def _mark_stale_analysis_runs(db: Connection, max_age_seconds: int = 60 * 60 * 2):
    stale_threshold_sql = "DATE_SUB(NOW(), INTERVAL %s SECOND)"
    with db.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE analysis_runs
            SET status = %s,
                message = %s,
                completed_at = NOW(),
                progress = 0.0
            WHERE status = %s
              AND submitted_at < {stale_threshold_sql}
              AND (is_cancelled IS NULL OR is_cancelled = FALSE)
            """,
            (
                ANALYSIS_STATUS_FAILED,
                "Marked failed due to stale processing timeout.",
                ANALYSIS_STATUS_PROCESSING,
                max_age_seconds,
            ),
        )
        db.commit()


def _cleanup_analysis_outputs(run_id: str):
    possible_keys = [
        f"outputs/{run_id}tracking.mp4",
        f"outputs/{run_id}tracking.json",
        f"outputs/analytics/{run_id}backline.mp4",
        f"outputs/heatmaps/{run_id}heatmap.mp4",
        f"outputs/analytics/{run_id}animation.mp4",
        f"outputs/analytics/{run_id}possession.json",
        f"outputs/analytics/{run_id}advisory.json",
    ]
    for rel in possible_keys:
        p = ANALYSIS_OUTPUT_ROOT / rel
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def _resolve_analysis_run(run_id: str, db: Connection):
    with db.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM analysis_runs WHERE id = %s LIMIT 1",
            (run_id,),
        )
        return cursor.fetchone()


def _update_analysis_run(
    run_id: str,
    status: str,
    progress: float,
    message: str,
    completed: bool = False,
):
    db_gen = get_db()
    db = next(db_gen)
    normalized_status = normalize_status(status)
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
                    (normalized_status, progress, message, run_id),
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
                    (normalized_status, progress, message, run_id),
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
    confidence_threshold: float = 0.5,
    ball_confidence: float = 0.3,
    max_lost_frames: int = 15,
    enable_reid: bool = False,
    target_team: str = "Both",
    camera_count: int = 1,
    camera_type: str = "TV",
):
    """Run tracking pipeline through Tracking Engine and persist status in DB."""
    client = TrackingEngineClient()
    try:
        _update_analysis_run(job_id, ANALYSIS_STATUS_PROCESSING, 0.05, "Submitting video to tracking engine...")

        async def progress_callback(response):
            if response.status == "SEGMENT_DONE":
                try:
                    seg_data = json.loads(response.message)
                    from services.segment_service import SegmentService
                    from controllers.segment_controller import push_analysis_segment_event

                    saved = SegmentService.insert_segment(
                        analysis_id=job_id,
                        match_id=match_id,
                        segment_index=seg_data.get("segment_index", 0),
                        start_sec=seg_data.get("start_sec", 0),
                        end_sec=seg_data.get("end_sec", 0),
                        video_start_sec=seg_data.get("video_start_sec", 0),
                        analysis_json=seg_data.get("analysis"),
                        recommendation=seg_data.get("recommendation"),
                        severity_score=seg_data.get("severity_score", 0),
                        severity_label=seg_data.get("severity_label", "LOW"),
                        status=seg_data.get("status", "COMPLETED"),
                    )
                    seg_payload = dict(seg_data)
                    seg_payload.update(saved)
                    seg_payload["type"] = "segment"
                    seg_payload["status"] = "SEGMENT_DONE"
                    await push_analysis_segment_event(job_id, seg_payload)

                    # --- Hybrid Flow Analysis ---
                    flow_event = await FlowAnalysisService.process_segment(job_id, seg_payload)
                    if flow_event:
                        await push_analysis_segment_event(job_id, flow_event)
                        
                except Exception as seg_err:
                    import logging

                    logging.getLogger(__name__).error(f"Failed to persist segment: {seg_err}")
            elif response.status == "ALERT":
                # Only stream tactical alerts; keep run status unchanged.
                pass
            else:
                mapped_status = normalize_status(response.status)
                _update_analysis_run(
                    job_id,
                    mapped_status,
                    float(response.progress),
                    response.message,
                    completed=(mapped_status in TERMINAL_STATUSES),
                )

        # Fetch active tracking profile for this match
        from services.tracking_profile_service import TrackingProfileService
        profiles = TrackingProfileService.get_profiles_for_match(match_id)
        active_profile = None
        for p in profiles:
            if p.get("is_active"):
                active_profile = p
                break
        
        calibration_json = None
        roi_json = None
        if active_profile and active_profile.get("cameras"):
            # For Phase 1: Single camera mapping
            cam = active_profile["cameras"][0]
            calibration_json = json.dumps(cam.get("calibration", []))
            roi_json = json.dumps(cam.get("roi", []))

        result = await client.analyze_video(
            video_path=video_path,
            match_id=job_id,
            frame_limit=frame_limit,
            skip_json=skip_json,
            confidence_threshold=confidence_threshold,
            calibration_json=calibration_json,
            roi_json=roi_json,
            ball_confidence=ball_confidence,
            max_lost_frames=max_lost_frames,
            enable_reid=enable_reid,
            target_team=target_team,
            camera_count=camera_count,
            camera_type=camera_type,
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

        final_status = normalize_status(result.get("status", ANALYSIS_STATUS_COMPLETED))
        _update_analysis_run(
            job_id,
            final_status,
            float(result.get("progress", 1.0)),
            result.get("message", "Analysis complete"),
            completed=(final_status in TERMINAL_STATUSES),
        )
        try:
            from controllers.segment_controller import push_analysis_segment_event

            await push_analysis_segment_event(job_id, {"type": "done", "status": final_status})
        except Exception:
            pass
    except Exception as e:
        _update_analysis_run(
            job_id,
            ANALYSIS_STATUS_FAILED,
            0.0,
            str(e),
            completed=True,
        )
        try:
            from controllers.segment_controller import push_analysis_segment_event

            await push_analysis_segment_event(job_id, {"type": "done", "status": ANALYSIS_STATUS_FAILED})
        except Exception:
            pass
    finally:
        try:
            await client.close()
        except Exception:
            pass
        # Keep source file for reuse/retry; cleanup occurs on cancel or manual maintenance.
        FlowAnalysisService.clear_buffer(job_id)
        pass


@app.post("/api/analysis/stale-detection", tags=["Analysis"])
async def stale_detection(db: Connection = Depends(get_db), current_user: User = Depends(get_current_user)):
    _mark_stale_analysis_runs(db)
    return {"status": "ok", "message": "Stale processing runs marked failed."}


@app.post("/api/analysis/{analysis_id}/cancel", tags=["Analysis"])
async def cancel_analysis_run(analysis_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_user)):
    run = _resolve_analysis_run(analysis_id, db)
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    if run.get("generated_by") != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    status = normalize_status(run.get("status", ""))
    if status in TERMINAL_STATUSES:
        raise HTTPException(status_code=400, detail="Cannot cancel a completed or failed run")

    task = _active_analysis_tasks.get(analysis_id)
    if task and not task.done():
        task.cancel()

    with db.cursor() as cursor:
        cursor.execute(
            "UPDATE analysis_runs SET status = %s, message = %s, completed_at = NOW(), progress = 0.0, is_cancelled = TRUE WHERE id = %s",
            (ANALYSIS_STATUS_FAILED, "Analysis cancelled by user.", analysis_id),
        )
        db.commit()

    _cleanup_analysis_outputs(analysis_id)
    input_video_path = run.get("input_video_path")
    if input_video_path:
        try:
            os.remove(input_video_path)
        except Exception:
            pass
    return {"status": "cancelled", "analysis_id": analysis_id}


@app.post("/api/analysis/{analysis_id}/retry", tags=["Analysis"])
async def retry_analysis_run(
    analysis_id: str,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    run = _resolve_analysis_run(analysis_id, db)
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    if run.get("generated_by") != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    input_video_path = run.get("input_video_path")
    if not input_video_path or not os.path.exists(input_video_path):
        raise HTTPException(status_code=400, detail="Original video not available for retry")

    new_id = str(uuid.uuid4())
    effective_match_id = run.get("match_id") or str(uuid.uuid4())
    frame_limit = run.get("frame_limit") or 0
    skip_json = bool(run.get("skip_json"))

    _create_analysis_run(
        run_id=new_id,
        match_id=effective_match_id,
        input_video_name=run.get("input_video_name") or "video.mp4",
        input_video_path=input_video_path,
        generated_by=current_user.id,
        frame_limit=frame_limit,
        skip_json=skip_json,
    )

    async def _retry_wrapper():
        task = asyncio.current_task()
        if task:
            _active_analysis_tasks[new_id] = task
        try:
            await run_tracking_analysis_job(new_id, input_video_path, effective_match_id, frame_limit, skip_json)
        finally:
            _active_analysis_tasks.pop(new_id, None)

    asyncio.create_task(_retry_wrapper())
    return {"status": "retry_started", "analysis_id": new_id, "original_id": analysis_id}


@app.get("/api/analysis_status/{analysis_id}", tags=["Analysis"])
async def get_analysis_status(analysis_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Check the status of a video analysis task from DB."""
    _mark_stale_analysis_runs(db)
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
        "input_video_path": f"temp_uploads/{os.path.basename(row['input_video_path'])}" if row.get("input_video_path") else None,
        "submitted_at": row["submitted_at"].isoformat() if row.get("submitted_at") else None,
        "completed_at": row["completed_at"].isoformat() if row.get("completed_at") else None,
    }


@app.get("/api/analysis_history", tags=["Analysis"])
async def get_analysis_history(db: Connection = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Return tracking analysis history for Analyze > History page."""
    _mark_stale_analysis_runs(db)
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
    if path.startswith("temp_uploads/"):
        filename = path.replace("temp_uploads/", "")
        candidate = (Path(PROJECT_ROOT) / "temp_uploads" / filename).resolve()
        # Verify it stays within temp_uploads
        if not str(candidate).startswith(os.path.abspath(os.path.join(PROJECT_ROOT, "temp_uploads"))):
            raise HTTPException(status_code=400, detail="Invalid path")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return candidate

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
        "tactical_advisory_path": f"outputs/analytics/{run_id}advisory.json",
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
    request: Request,
    path: str,
    access_token: str = "",
    db: Connection = Depends(get_db),
    current_user: Optional[User] = None,
):
    current_user = await get_current_user_flexible(request, access_token, db)
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
    current_user = await get_current_user_flexible(request, access_token, db)
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
    request: Request,
    path: str,
    access_token: str = "",
    db: Connection = Depends(get_db),
    current_user: Optional[User] = None,
):
    current_user = await get_current_user_flexible(request, access_token, db)
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
    file2: Optional[UploadFile] = File(None),
    match_id: Optional[str] = Form(None),
    frame_limit: int = Form(0),
    skip_json: bool = Form(False),
    confidence_threshold: float = Form(0.5),
    ball_confidence: float = Form(0.3),
    max_lost_frames: int = Form(15),
    enable_reid: bool = Form(False),
    target_team: str = Form("Both"),
    camera_count: int = Form(1),
    camera_type: str = Form("TV"),
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
            
        file_path_2 = None
        if file2:
            safe_name_2 = file2.filename or "video2.mp4"
            file_path_2 = os.path.abspath(os.path.join(upload_dir, f"{analysis_id}_{safe_name_2}"))
            with open(file_path_2, "wb") as buffer:
                shutil.copyfileobj(file2.file, buffer)

        _create_analysis_run(
            run_id=analysis_id,
            match_id=effective_match_id,
            input_video_name=safe_name,
            input_video_path=file_path,
            generated_by=current_user.id,
            frame_limit=frame_limit,
            skip_json=skip_json,
        )

        async def _run_job_wrapper():
            task = asyncio.current_task()
            if task:
                _active_analysis_tasks[analysis_id] = task
            try:
                await run_tracking_analysis_job(
                    analysis_id,
                    file_path,
                    effective_match_id,
                    frame_limit,
                    skip_json,
                    confidence_threshold=confidence_threshold,
                    ball_confidence=ball_confidence,
                    max_lost_frames=max_lost_frames,
                    enable_reid=enable_reid,
                    target_team=target_team,
                    camera_count=camera_count,
                    camera_type=camera_type,
                )
            finally:
                _active_analysis_tasks.pop(analysis_id, None)

        asyncio.create_task(_run_job_wrapper())

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
