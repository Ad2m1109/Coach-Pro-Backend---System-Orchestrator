from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from enum import Enum

class UserTypeEnum(str, Enum):
    owner = "owner"
    staff = "staff"

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    user_type: UserTypeEnum = UserTypeEnum.owner
    is_active: bool = True

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: str
    password_hash: str # Add this line
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True
