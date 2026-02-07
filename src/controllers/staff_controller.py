from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.staff_service import StaffService
from models.staff import Staff, StaffCreate, StaffCreateWithAccount
from app import get_current_active_user # Import the dependency
from models.user import User # Import User model
from services.team_service import TeamService

router = APIRouter()

@router.post("/staff/create_with_account", response_model=Staff)
def create_staff_with_account(
    staff_data: StaffCreateWithAccount, 
    db: Connection = Depends(get_db), 
    current_user: User = Depends(get_current_active_user)
):
    """Create a staff member with a new user account (owner only)"""
    # Only owners can create staff accounts
    if current_user.user_type != "owner":
        raise HTTPException(status_code=403, detail="Only team owners can create staff accounts")
    
    # Get user's teams
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    
    service = StaffService(db)
    try:
        return service.create_staff_with_account(staff_data, user_team_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/staff", response_model=Staff)
def create_staff(staff: StaffCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    # Get user's teams
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    
    service = StaffService(db)
    return service.create_staff(staff, user_team_ids)

@router.get("/staff", response_model=List[Staff])
def get_all_staff(db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    # Get user's teams
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    
    service = StaffService(db)
    return service.get_all_staff(user_team_ids)

@router.get("/staff/{staff_id}", response_model=Staff)
def get_staff(staff_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    # Get user's teams
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    
    service = StaffService(db)
    staff = service.get_staff(staff_id, user_team_ids)
    if not staff:
        raise HTTPException(status_code=404, detail="Staff not found")
    return staff

@router.put("/staff/{staff_id}", response_model=Staff)
def update_staff(staff_id: str, staff_update: StaffCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    # Get user's teams
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    
    service = StaffService(db)
    staff = service.update_staff(staff_id, staff_update, user_team_ids)
    if not staff:
        raise HTTPException(status_code=404, detail="Staff not found")
    return staff

@router.delete("/staff/{staff_id}")
def delete_staff(staff_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    # Get user's teams
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    
    service = StaffService(db)
    if not service.delete_staff(staff_id, user_team_ids):
        raise HTTPException(status_code=404, detail="Staff not found")
    return {"message": "Staff deleted successfully"}
