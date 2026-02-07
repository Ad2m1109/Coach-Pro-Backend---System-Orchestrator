from pydantic import BaseModel, EmailStr
from typing import Optional
from enum import Enum

class StaffRoleEnum(str, Enum):
    head_coach = "head_coach"
    assistant_coach = "assistant_coach"
    physio = "physio"
    analyst = "analyst"

class PermissionLevelEnum(str, Enum):
    full_access = "full_access"
    view_only = "view_only"
    notes_only = "notes_only"

class StaffBase(BaseModel):
    team_id: Optional[str] = None
    name: str
    role: Optional[StaffRoleEnum] = None
    permission_level: PermissionLevelEnum = PermissionLevelEnum.view_only
    email: Optional[str] = None

class StaffCreate(StaffBase):
    password: Optional[str] = None  # For creating user account
    user_id: Optional[str] = None   # If linking to existing user

class StaffCreateWithAccount(BaseModel):
    """For creating staff with new user account"""
    team_id: str
    name: str
    role: StaffRoleEnum
    permission_level: PermissionLevelEnum
    email: EmailStr
    password: str

class Staff(StaffBase):
    id: str
    user_id: Optional[str] = None

    class Config:
        from_attributes = True
