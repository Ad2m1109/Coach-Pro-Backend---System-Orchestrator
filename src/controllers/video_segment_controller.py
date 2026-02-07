from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.video_segment_service import VideoSegmentService
from models.video_segment import VideoSegment, VideoSegmentCreate
from dependencies import get_current_active_user # Import the dependency
from services.team_service import TeamService # New import
from models.user import User # Import User model

router = APIRouter()

@router.post("/video_segments", response_model=VideoSegment)
def create_video_segment(segment: VideoSegmentCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = VideoSegmentService(db)
    return service.create_video_segment(segment, user_team_ids)

@router.get("/video_segments", response_model=List[VideoSegment])
def get_all_video_segments(db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = VideoSegmentService(db)
    return service.get_all_video_segments(user_team_ids)

@router.get("/video_segments/{segment_id}", response_model=VideoSegment)
def get_video_segment(segment_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = VideoSegmentService(db)
    segment = service.get_video_segment(segment_id, user_team_ids)
    if not segment:
        raise HTTPException(status_code=404, detail="Video Segment not found")
    return segment

@router.put("/video_segments/{segment_id}", response_model=VideoSegment)
def update_video_segment(segment_id: str, segment_update: VideoSegmentCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = VideoSegmentService(db)
    segment = service.update_video_segment(segment_id, segment_update, user_team_ids)
    if not segment:
        raise HTTPException(status_code=404, detail="Video Segment not found")
    return segment

@router.delete("/video_segments/{segment_id}")
def delete_video_segment(segment_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = VideoSegmentService(db)
    if not service.delete_video_segment(segment_id, user_team_ids):
        raise HTTPException(status_code=404, detail="Video Segment not found")
    return {"message": "Video Segment deleted successfully"}
