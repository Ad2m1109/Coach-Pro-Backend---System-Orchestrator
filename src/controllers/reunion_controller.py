from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.reunion_service import ReunionService
from models.reunion import Reunion, ReunionCreate
from dependencies import get_current_active_user # Import the dependency
from services.team_service import TeamService # New import
from models.user import User # Import User model

router = APIRouter()

@router.post("/reunions", response_model=Reunion)
def create_reunion(reunion: ReunionCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = ReunionService(db)
    return service.create_reunion(reunion, user_team_ids)

@router.get("/reunions", response_model=List[Reunion])
def get_all_reunions(db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = ReunionService(db)
    return service.get_all_reunions(user_team_ids)

@router.delete("/reunions/{reunion_id}")
def delete_reunion(reunion_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = ReunionService(db)
    if not service.delete_reunion(reunion_id, user_team_ids):
        raise HTTPException(status_code=404, detail="Reunion not found")
    return {"message": "Reunion deleted successfully"}
