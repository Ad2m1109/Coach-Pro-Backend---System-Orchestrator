from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
from database import get_db, Connection
from services.team_service import TeamService
from models.team import Team, TeamCreate
from dependencies import get_current_active_user # Import the dependency
from models.user import User # Import User model

router = APIRouter()

@router.post("/teams", response_model=Team)
def create_team(team: TeamCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = TeamService(db)
    return service.create_team(team, current_user.id)

@router.get("/teams", response_model=List[Team])
def get_all_teams(db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = TeamService(db)
    return service.get_all_teams(current_user.id)

@router.get("/teams/by_name/{name}", response_model=Team)
def get_team_by_name(name: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = TeamService(db)
    team = service.get_team_by_name(name)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team

@router.get("/teams/{team_id}", response_model=Team)
def get_team(team_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = TeamService(db)
    team = service.get_team(team_id, current_user.id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team

@router.put("/teams/{team_id}", response_model=Team)
def update_team(team_id: str, team_update: TeamCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = TeamService(db)
    team = service.update_team(team_id, team_update, current_user.id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team

@router.delete("/teams/{team_id}")
def delete_team(team_id: str, db: Connection = Depends(get_db), current_user: dict = Depends(get_current_active_user)):
    service = TeamService(db)
    if not service.delete_team(team_id):
        raise HTTPException(status_code=404, detail="Team not found")
    return {"message": "Team deleted successfully"}

@router.post("/teams/{team_id}/upload_logo", response_model=Team)
def upload_team_logo(team_id: str, file: UploadFile = File(...), db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = TeamService(db)
    
    # First, verify the team belongs to the current user
    team = service.get_team(team_id, current_user.id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found or not authorized")

    # Now, handle the file upload
    try:
        updated_team = service.save_team_logo(team_id, file, current_user.id)
        return updated_team
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
