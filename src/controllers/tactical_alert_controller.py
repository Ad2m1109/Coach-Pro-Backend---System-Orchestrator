from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dependencies import get_current_active_user
from models.user import User
from services.tactical_alert_service import TacticalAlertService

router = APIRouter()
alert_service = TacticalAlertService()


class DecisionFeedbackPayload(BaseModel):
    decision_id: str = Field(..., min_length=1)
    action: str = Field(..., pattern="^(ACCEPT|DISMISS)$")
    match_time: Optional[float] = None
    match_id: Optional[str] = None


@router.get("/api/matches/{match_id}/alerts", tags=["Tactical Alerts"])
async def get_alert_history(
    match_id: str,
    current_user: User = Depends(get_current_active_user),
):
    return alert_service.get_alert_history(match_id, user_id=str(current_user.id))


@router.post("/api/decision/feedback", tags=["Tactical Alerts"])
@router.post("/decision/feedback", tags=["Tactical Alerts"])
async def submit_decision_feedback(
    payload: DecisionFeedbackPayload,
    current_user: User = Depends(get_current_active_user),
):
    try:
        result = await alert_service.submit_feedback(
            user_id=str(current_user.id),
            decision_id=payload.decision_id,
            action=payload.action,
            match_time=payload.match_time,
            match_id=payload.match_id,
        )
        return result
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/matches/{match_id}/alerts/{alert_id}/feedback", tags=["Tactical Alerts"])
async def submit_alert_feedback_legacy(
    match_id: str,
    alert_id: str,
    feedback: str,
    match_time: Optional[float] = None,
    current_user: User = Depends(get_current_active_user),
):
    normalized = feedback.strip().lower()
    if normalized not in {"accepted", "dismissed", "none"}:
        raise HTTPException(status_code=400, detail="Invalid feedback type")
    if normalized == "none":
        raise HTTPException(status_code=400, detail="Feedback 'none' is not allowed for decision submission")

    action = "ACCEPT" if normalized == "accepted" else "DISMISS"
    result = await alert_service.submit_feedback(
        user_id=str(current_user.id),
        decision_id=alert_id,
        action=action,
        match_time=match_time,
        match_id=match_id,
    )
    return {
        "status": "success",
        "alert_id": alert_id,
        "feedback": normalized,
        "idempotent": result.get("idempotent", False),
    }


@router.get("/api/decision/metrics", tags=["Tactical Alerts"])
@router.get("/decision/metrics", tags=["Tactical Alerts"])
async def get_decision_metrics(
    match_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
):
    return alert_service.get_decision_metrics(user_id=str(current_user.id), match_id=match_id)
