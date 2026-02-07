from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.match_team_statistics_service import MatchTeamStatisticsService
from models.match_team_statistics import MatchTeamStatistics
from dependencies import get_current_active_user # Import the dependency
from services.team_service import TeamService # New import
from models.user import User # Import User model

router = APIRouter()

@router.get("/matches/{match_id}/team_statistics", response_model=List[MatchTeamStatistics])
def get_match_team_statistics(match_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = MatchTeamStatisticsService(db)
    stats = service.get_by_match_id(match_id, user_team_ids)
    if not stats:
        raise HTTPException(status_code=404, detail="Team statistics not found for this match")
    return stats
