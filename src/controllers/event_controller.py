from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.event_service import EventService
from models.event import Event, EventCreate
from dependencies import get_current_active_user # Import the dependency
from services.team_service import TeamService # New import
from models.user import User # Import User model

router = APIRouter()

@router.post("/events", response_model=Event)
def create_event(event: EventCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = EventService(db)
    return service.create_event(event, user_team_ids)

@router.get("/events", response_model=List[Event])
def get_all_events(db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = EventService(db)
    return service.get_all_events(user_team_ids)

@router.get("/events/{event_id}", response_model=Event)
def get_event(event_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = EventService(db)
    event = service.get_event(event_id, user_team_ids)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event

@router.put("/events/{event_id}", response_model=Event)
def update_event(event_id: str, event_update: EventCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = EventService(db)
    event = service.update_event(event_id, event_update, user_team_ids)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event

@router.delete("/events/{event_id}")
def delete_event(event_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = EventService(db)
    if not service.delete_event(event_id, user_team_ids):
        raise HTTPException(status_code=404, detail="Event not found")
    return {"message": "Event deleted successfully"}