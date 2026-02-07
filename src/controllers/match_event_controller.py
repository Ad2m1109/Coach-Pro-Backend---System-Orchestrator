from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.match_event_service import MatchEventService
from models.match_event import MatchEvent, MatchEventCreate
from dependencies import get_current_active_user # Import the dependency

router = APIRouter()

@router.post("/match_events", response_model=MatchEvent)
def create_match_event(event: MatchEventCreate, db: Connection = Depends(get_db), current_user: dict = Depends(get_current_active_user)):
    service = MatchEventService(db)
    return service.create_match_event(event)

@router.get("/match_events", response_model=List[MatchEvent])
def get_all_match_events(db: Connection = Depends(get_db), current_user: dict = Depends(get_current_active_user)):
    service = MatchEventService(db)
    return service.get_all_match_events()

@router.get("/match_events/{event_id}", response_model=MatchEvent)
def get_match_event(event_id: str, db: Connection = Depends(get_db), current_user: dict = Depends(get_current_active_user)):
    service = MatchEventService(db)
    event = service.get_match_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Match Event not found")
    return event

@router.put("/match_events/{event_id}", response_model=MatchEvent)
def update_match_event(event_id: str, event_update: MatchEventCreate, db: Connection = Depends(get_db), current_user: dict = Depends(get_current_active_user)):
    service = MatchEventService(db)
    event = service.update_match_event(event_id, event_update)
    if not event:
        raise HTTPException(status_code=404, detail="Match Event not found")
    return event

@router.delete("/match_events/{event_id}")
def delete_match_event(event_id: str, db: Connection = Depends(get_db), current_user: dict = Depends(get_current_active_user)):
    service = MatchEventService(db)
    if not service.delete_match_event(event_id):
        raise HTTPException(status_code=404, detail="Match Event not found")
    return {"message": "Match Event deleted successfully"}
