from typing import Dict, Optional
from fastapi import UploadFile, HTTPException, BackgroundTasks
from tempfile import NamedTemporaryFile
import shutil
import os

from services.match_service import MatchService
from services.player_match_statistics_service import PlayerMatchStatisticsService
from services.analysis_grpc_service import AnalysisGrpcService
from analysis_engine import parse_and_persist_results

# In-memory store for analysis status
# In production, use Redis or similar
_analysis_status: Dict[str, Dict] = {}

# Initialize gRPC service
_grpc_service = AnalysisGrpcService()

async def start_video_analysis(
    match_id: str,
    video_file: UploadFile,
    background_tasks: BackgroundTasks,
    match_service: MatchService,
    player_stats_service: PlayerMatchStatisticsService,
) -> Dict:
    """Start video analysis in background and return status tracking ID."""
    
    # Validate match exists
    match = match_service.get_match(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    # Save uploaded video to temp file
    # In a real system, we might upload to S3 and pass the URL to the analysis engine
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        try:
            shutil.copyfileobj(video_file.file, temp_video)
            temp_video_path = temp_video.name
        finally:
            video_file.file.close()

    # Initialize status tracking
    _analysis_status[match_id] = {
        "status": "pending",
        "progress": 0.0,
        "error": None
    }

    # Start analysis in background
    background_tasks.add_task(
        _run_analysis,
        match_id=match_id,
        video_path=temp_video_path,
        match_service=match_service,
        player_stats_service=player_stats_service
    )

    return {
        "status": "accepted",
        "message": "Video analysis started"
    }

def get_analysis_status(match_id: str) -> Dict:
    """Get current status of video analysis for a match."""
    if match_id not in _analysis_status:
        raise HTTPException(status_code=404, detail="No analysis found for match")
    return _analysis_status[match_id]

async def _run_analysis(
    match_id: str,
    video_path: str,
    match_service: MatchService,
    player_stats_service: PlayerMatchStatisticsService
):
    """Background task to run video analysis via gRPC and persist results."""
    try:
        # Update status to running
        _analysis_status[match_id].update({
            "status": "running",
            "progress": 0.0
        })

        # Run analysis via gRPC stream
        # Note: In a production app, we'd pass actual calibration paths if available
        responses = _grpc_service.analyze_video(
            video_path=video_path,
            match_id=match_id,
            confidence_threshold=0.5
        )

        final_result = None
        for response in responses:
            if response.status == "FAILED":
                raise RuntimeError(response.message)
            
            # Update progress
            _analysis_status[match_id].update({
                "status": response.status.lower(),
                "progress": response.progress,
                "message": response.message
            })
            
            if response.status == "COMPLETED":
                final_result = response.result
                break

        if final_result:
            # Persist results to DB
            # The C++ engine is expected to provide paths to the generated CSVs
            output_dir = os.path.dirname(final_result.player_metrics_csv_path)
            
            results_summary = parse_and_persist_results(
                output_dir=output_dir,
                db_connection=match_service.db_connection,
                match_id=match_id
            )

            # Update status on success
            _analysis_status[match_id].update({
                "status": "completed",
                "progress": 1.0,
                "results": results_summary
            })

    except Exception as e:
        # Update status on failure
        _analysis_status[match_id].update({
            "status": "failed",
            "error": str(e)
        })
        # Log the error properly in a real app
        print(f"Analysis failed for match {match_id}: {e}")
    
    finally:
        # Clean up temp file
        try:
            os.unlink(video_path)
        except:
            pass