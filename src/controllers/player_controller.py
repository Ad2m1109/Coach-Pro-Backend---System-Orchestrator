from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
from database import get_db, Connection
from services.player_service import PlayerService
from services.team_service import TeamService # New import
from models.player import Player, PlayerCreate
from dependencies import get_current_active_user # Import the dependency
from models.user import User # Import User model

router = APIRouter()

@router.post("/players", response_model=Player)
def create_player(player: PlayerCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerService(db)
    return service.create_player(player, user_team_ids)

@router.get("/players", response_model=List[Player])
def get_all_players(db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerService(db)
    return service.get_all_players(user_team_ids)

@router.get("/players/{player_id}", response_model=Player)
def get_player(player_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerService(db)
    player = service.get_player(player_id, user_team_ids)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player

@router.put("/players/{player_id}", response_model=Player)
def update_player(player_id: str, player_update: PlayerCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerService(db)
    player = service.update_player(player_id, player_update, user_team_ids)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player

@router.delete("/players/{player_id}")
def delete_player(player_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerService(db)
    if not service.delete_player(player_id, user_team_ids):
        raise HTTPException(status_code=404, detail="Player not found")
    return {"message": "Player deleted successfully"}

@router.post("/players/{player_id}/upload_image", response_model=Player)
def upload_player_image(player_id: str, file: UploadFile = File(...), db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = PlayerService(db)
    
    # First, verify the player belongs to one of the user's teams
    player = service.get_player(player_id, user_team_ids)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found or not authorized")

    # Now, handle the file upload
    try:
        updated_player = service.save_player_image(player_id, file, user_team_ids)
        return updated_player
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
