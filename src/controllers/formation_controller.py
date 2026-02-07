from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.formation_service import FormationService
from models.formation import Formation, FormationCreate
from dependencies import get_current_active_user
from models.user import User

router = APIRouter()

@router.post("/formations", response_model=Formation)
def create_formation(formation: FormationCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = FormationService(db)
    return service.create_formation(formation, current_user.id)

@router.get("/formations", response_model=List[Formation])
def get_all_formations(db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = FormationService(db)
    return service.get_all_formations(current_user.id)

@router.get("/formations/{formation_id}", response_model=Formation)
def get_formation(formation_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = FormationService(db)
    formation = service.get_formation(formation_id, current_user.id)
    if not formation:
        raise HTTPException(status_code=404, detail="Formation not found")
    return formation

@router.put("/formations/{formation_id}", response_model=Formation)
def update_formation(formation_id: str, formation_update: FormationCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = FormationService(db)
    formation = service.update_formation(formation_id, formation_update, current_user.id)
    if not formation:
        raise HTTPException(status_code=404, detail="Formation not found or not owned by user")
    return formation

@router.delete("/formations/{formation_id}")
def delete_formation(formation_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    service = FormationService(db)
    if not service.delete_formation(formation_id, current_user.id):
        raise HTTPException(status_code=404, detail="Formation not found or not owned by user")
    return {"message": "Formation deleted successfully"}
