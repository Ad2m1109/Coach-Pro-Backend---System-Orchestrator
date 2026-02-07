from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.match_lineup_service import MatchLineupService
from models.match_lineup import MatchLineup, MatchLineupCreate
from dependencies import get_current_active_user # Import the dependency
from services.team_service import TeamService # New import
from models.user import User # Import User model

router = APIRouter()

@router.post("/match_lineups", response_model=MatchLineup)
def create_match_lineup(lineup: MatchLineupCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchLineupService(db)
    return service.create_match_lineup(lineup, user_team_ids)

@router.get("/match_lineups", response_model=List[MatchLineup])
def get_all_match_lineups(match_id: str = None, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchLineupService(db)
    return service.get_all_match_lineups(match_id=match_id, user_team_ids=user_team_ids)

@router.get("/match_lineups/{lineup_id}", response_model=MatchLineup)
def get_match_lineup(lineup_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchLineupService(db)
    lineup = service.get_match_lineup(lineup_id, user_team_ids)
    if not lineup:
        raise HTTPException(status_code=404, detail="Match Lineup not found")
    return lineup

@router.put("/match_lineups/{lineup_id}", response_model=MatchLineup)
def update_match_lineup(lineup_id: str, lineup_update: MatchLineupCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchLineupService(db)
    lineup = service.update_match_lineup(lineup_id, lineup_update, user_team_ids)
    if not lineup:
        raise HTTPException(status_code=404, detail="Match Lineup not found")
    return lineup

@router.delete("/match_lineups/{lineup_id}")
def delete_match_lineup(lineup_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchLineupService(db)
    if not service.delete_match_lineup(lineup_id, user_team_ids):
        raise HTTPException(status_code=404, detail="Match Lineup not found")
    return {"message": "Match Lineup deleted successfully"}
