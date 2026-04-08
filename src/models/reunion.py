from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ReunionBase(BaseModel):
    title: str
    date: datetime
    location: str
    icon_name: str
    team_id: Optional[str] = None

class ReunionCreate(ReunionBase):
    pass

class Reunion(ReunionBase):
    id: str

    class Config:
        from_attributes = True
