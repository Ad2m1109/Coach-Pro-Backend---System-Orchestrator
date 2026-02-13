"""
Controller for video analysis requests.
Handles job submission, status checking, and result retrieval.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from typing import Optional
from pathlib import Path
import uuid
from database import get_db, Connection
from services.analysis_job_service import AnalysisJobService
from dependencies import get_current_active_user
from models.user import User
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/matches/{match_id}/analyze")
async def submit_analysis(
    match_id: str,
    video_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    frame_limit: int = Form(0),
    skip_json: bool = Form(False),
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
):
    """
    Submit a video for analysis.
    
    Returns immediately with a job ID.
    Analysis runs asynchronously in the background.
    """
    try:
        service = AnalysisJobService(db)
        effective_video_path = video_path

        # Support direct file uploads from frontend.
        if file is not None:
            upload_root = Path("uploads") / "analysis_jobs" / match_id
            upload_root.mkdir(parents=True, exist_ok=True)

            safe_name = file.filename or "video.mp4"
            saved_path = upload_root / f"{uuid.uuid4()}_{safe_name}"

            with saved_path.open("wb") as output:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            await file.close()
            effective_video_path = str(saved_path.resolve())

        if not effective_video_path:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'video_path' or upload a 'file'."
            )

        # Create job record
        job = service.create_analysis_job(match_id, effective_video_path, current_user.id)

        # Submit to tracking engine in background
        if background_tasks is None:
            background_tasks = BackgroundTasks()
        background_tasks.add_task(
            service.submit_analysis_job,
            job['job_id'],
            match_id,
            effective_video_path,
            frame_limit,
            skip_json
        )
        
        return {
            'job_id': job['job_id'],
            'match_id': match_id,
            'status': 'PENDING',
            'message': 'Analysis started. You can check progress using the job ID.'
        }
    except Exception as e:
        logger.error(f"Failed to submit analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{job_id}/status")
def get_analysis_status(
    job_id: str,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get the status of an analysis job."""
    try:
        service = AnalysisJobService(db)
        status = service.get_job_status(job_id)
        
        if status['status'] == 'NOT_FOUND':
            raise HTTPException(status_code=404, detail="Job not found")
        
        return status
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/matches/{match_id}/analysis")
def get_match_analysis(
    match_id: str,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get analysis results for a match."""
    try:
        service = AnalysisJobService(db)
        analysis = service.get_match_analysis(match_id)
        
        if not analysis:
            return {
                'status': 'NO_ANALYSIS',
                'message': 'No analysis results available for this match'
            }
        
        return analysis
    except Exception as e:
        logger.error(f"Failed to get match analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
