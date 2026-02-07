from pydantic import BaseModel
from typing import Optional
from enum import Enum
from datetime import datetime

class NoteTypeEnum(str, Enum):
    pre_match = "pre_match"
    live_reaction = "live_reaction"
    tactical = "tactical"

class MatchNoteBase(BaseModel):
    match_id: str
    content: str
    note_type: NoteTypeEnum = NoteTypeEnum.tactical
    video_timestamp: float = 0.0

class MatchNoteCreate(MatchNoteBase):
    user_id: Optional[str] = None # Will be set from current_user

class MatchNote(MatchNoteBase):
    id: str
    user_id: str
    created_at: datetime
    author_name: Optional[str] = None
    author_role: Optional[str] = None

    class Config:
        from_attributes = True
