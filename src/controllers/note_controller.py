from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.note_service import NoteService
from models.note import MatchNote, MatchNoteCreate
from dependencies import get_current_active_user
from models.user import User
from services.staff_service import StaffService

router = APIRouter()

@router.post("/matches/{match_id}/notes", response_model=MatchNote)
def create_match_note(
    match_id: str,
    note: MatchNoteCreate,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    # Owners always have permission, staff need 'notes' permission
    if current_user.user_type != 'owner':
        staff_service = StaffService(db)
        staff = staff_service.get_staff_by_user_id(current_user.id)
        if not staff or not staff_service.check_permission(staff, 'notes'):
            raise HTTPException(status_code=403, detail="Permission denied to add notes")
    
    note.match_id = match_id
    note.user_id = current_user.id
    
    service = NoteService(db)
    try:
        return service.create_note(note)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/matches/{match_id}/notes", response_model=List[MatchNote])
def get_match_notes(
    match_id: str,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    service = NoteService(db)
    return service.get_match_notes(match_id)

@router.delete("/notes/{note_id}")
def delete_note(
    note_id: str,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    service = NoteService(db)
    if not service.delete_note(note_id, current_user.id):
        raise HTTPException(status_code=404, detail="Note not found or you are not the author")
    return {"message": "Note deleted successfully"}
