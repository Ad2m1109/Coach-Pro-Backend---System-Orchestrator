from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from database import get_db, Connection
from services.match_service import MatchService
from services.team_service import TeamService # New import
from models.match import Match, MatchCreate
from models.match_details import MatchDetails
from dependencies import get_current_active_user # Import the dependency
from models.user import User # Import User model

router = APIRouter()

@router.post("/matches", response_model=Match)
def create_match(match: MatchCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchService(db)
    return service.create_match(match, user_team_ids)

@router.get("/matches", response_model=List[Match])
def get_all_matches(status: Optional[str] = None, event_id: Optional[str] = None, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchService(db)
    return service.get_all_matches(user_team_ids, status=status, event_id=event_id)

@router.get("/matches/{match_id}", response_model=Match)
def get_match(match_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchService(db)
    match = service.get_match(match_id, user_team_ids)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match

@router.put("/matches/{match_id}", response_model=Match)
def update_match(match_id: str, match_update: MatchCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchService(db)
    match = service.update_match(match_id, match_update, user_team_ids)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match

@router.delete("/matches/{match_id}")
def delete_match(match_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchService(db)
    if not service.delete_match(match_id, user_team_ids):
        raise HTTPException(status_code=404, detail="Match not found")
    return {"message": "Match deleted successfully"}

@router.get("/matches/{match_id}/details", response_model=MatchDetails)
def get_match_details(match_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchService(db)
    details = service.get_match_details(match_id, user_team_ids, current_user.id)
    if not details:
        raise HTTPException(status_code=404, detail="Match details not found")
    return details
