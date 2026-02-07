from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.player_match_statistics_service import PlayerMatchStatisticsService
from models.player_match_statistics import PlayerMatchStatistics, PlayerMatchStatisticsCreate
from dependencies import get_current_active_user # Import the dependency
from services.team_service import TeamService # New import
from models.user import User # Import User model

router = APIRouter()

@router.post("/player_match_statistics", response_model=PlayerMatchStatistics)
def create_player_match_statistics(stats: PlayerMatchStatisticsCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerMatchStatisticsService(db)
    return service.create_player_match_statistics(stats, user_team_ids)

@router.get("/player_match_statistics", response_model=List[PlayerMatchStatistics])
def get_all_player_match_statistics(match_id: str = None, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerMatchStatisticsService(db)
    return service.get_all_player_match_statistics(match_id=match_id, user_team_ids=user_team_ids)

@router.get("/player_match_statistics/{stat_id}", response_model=PlayerMatchStatistics)
def get_player_match_statistics(stat_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerMatchStatisticsService(db)
    stats = service.get_player_match_statistics(stat_id, user_team_ids)
    if not stats:
        raise HTTPException(status_code=404, detail="Player Match Statistics not found")
    return stats

@router.put("/player_match_statistics/{stat_id}", response_model=PlayerMatchStatistics)
def update_player_match_statistics(stat_id: str, stats_update: PlayerMatchStatisticsCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerMatchStatisticsService(db)
    stats = service.update_player_match_statistics(stat_id, stats_update, user_team_ids)
    if not stats:
        raise HTTPException(status_code=404, detail="Player Match Statistics not found")
    return stats

@router.delete("/player_match_statistics/{stat_id}")
def delete_player_match_statistics(stat_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerMatchStatisticsService(db)
    if not service.delete_player_match_statistics(stat_id, user_team_ids):
        raise HTTPException(status_code=404, detail="Player Match Statistics not found")
    return {"message": "Player Match Statistics deleted successfully"}

@router.get("/player_match_statistics/player/{player_id}", response_model=List[PlayerMatchStatistics])
def get_player_match_statistics_for_player(player_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerMatchStatisticsService(db)
    stats = service.get_player_match_statistics_by_player_id(player_id, user_team_ids)
    if not stats:
        raise HTTPException(status_code=404, detail="Player Match Statistics not found for this player")
    return stats
