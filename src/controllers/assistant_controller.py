"""
Assistant Router — /assistant endpoints
========================================

Exposes:
  POST /assistant/query     — Ask the AI assistant a question
  POST /assistant/mode      — Switch system mode (ASSISTANT / ANALYSIS)
  GET  /assistant/mode      — Get current system mode
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from database import get_db, Connection
from dependencies import get_current_active_user
from models.user import User

from services.assistant_service import (
    query_assistant,
    get_system_mode,
    set_system_mode,
    SystemMode,
)


router = APIRouter()


# ------------------------------------------------------------------
# Request / Response Models
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    status: str
    answer: str | None = None
    message: str | None = None


class ModeRequest(BaseModel):
    mode: SystemMode


class ModeResponse(BaseModel):
    mode: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
async def assistant_query(
    body: QueryRequest,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Send a question to the AI assistant."""
    result = await query_assistant(body.question, db, current_user.id)
    return result


@router.get("/mode", response_model=ModeResponse)
async def get_mode():
    """Get the current system mode."""
    return {"mode": get_system_mode().value}


@router.post("/mode", response_model=ModeResponse)
async def set_mode(body: ModeRequest):
    """Switch between ASSISTANT and ANALYSIS modes."""
    set_system_mode(body.mode)
    return {"mode": body.mode.value}
