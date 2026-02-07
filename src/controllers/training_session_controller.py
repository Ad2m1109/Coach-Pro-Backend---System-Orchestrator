from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.training_session_service import TrainingSessionService
from models.training_session import TrainingSession, TrainingSessionCreate
from dependencies import get_current_active_user # Import the dependency
from services.team_service import TeamService # New import
from models.user import User # Import User model

router = APIRouter()

@router.post("/training_sessions", response_model=TrainingSession)
def create_training_session(session: TrainingSessionCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = TrainingSessionService(db)
    return service.create_training_session(session, user_team_ids)

@router.get("/training_sessions", response_model=List[TrainingSession])
def get_all_training_sessions(db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = TrainingSessionService(db)
    return service.get_all_training_sessions(user_team_ids)

@router.delete("/training_sessions/{session_id}")
def delete_training_session(session_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = TrainingSessionService(db)
    if not service.delete_training_session(session_id, user_team_ids):
        raise HTTPException(status_code=404, detail="Training session not found")
    return {"message": "Training session deleted successfully"}
