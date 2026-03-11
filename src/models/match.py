from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum

class MatchStatusEnum(str, Enum):
    upcoming = "upcoming"
    live = "live"
    completed = "completed"

class MatchBase(BaseModel):
    home_team_id: str
    away_team_id: str
    date_time: datetime
    venue: Optional[str] = None
    event_id: Optional[str] = None # Changed from competition
    status: Optional[MatchStatusEnum] = None
    home_score: Optional[int] = 0
    away_score: Optional[int] = 0
    video_anchor_seconds: Optional[int] = None

class MatchCreate(MatchBase):
    pass

class Match(MatchBase):
    id: str
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None
    event_name: Optional[str] = None # Added event_name

    class Config:
        from_attributes = True
