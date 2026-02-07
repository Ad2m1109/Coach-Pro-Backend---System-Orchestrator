from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.user_service import UserService
from models.user import User, UserCreate
from dependencies import get_current_active_user # Import the dependency

router = APIRouter()

@router.post("/users", response_model=User)
def create_user(user: UserCreate, db: Connection = Depends(get_db)):
    service = UserService(db)
    return service.create_user(user)

@router.get("/users/me", response_model=User) # New endpoint
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@router.get("/users", response_model=List[User])
def get_all_users(db: Connection = Depends(get_db), current_user: dict = Depends(get_current_active_user)):
    service = UserService(db)
    return service.get_all_users()

@router.get("/users/{user_id}", response_model=User)
def get_user(user_id: str, db: Connection = Depends(get_db), current_user: dict = Depends(get_current_active_user)):
    service = UserService(db)
    user = service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/users/{user_id}", response_model=User)
def update_user(user_id: str, user_update: UserCreate, db: Connection = Depends(get_db), current_user: dict = Depends(get_current_active_user)):
    service = UserService(db)
    user = service.update_user(user_id, user_update)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.delete("/users/{user_id}")
def delete_user(user_id: str, db: Connection = Depends(get_db), current_user: dict = Depends(get_current_active_user)):
    service = UserService(db)
    if not service.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}
