"""
Controller for analysis segments — per-time-window tactical analytics.

Endpoints:
    GET  /api/matches/{match_id}/segments       → list all segments
    GET  /api/matches/{match_id}/segments/stream → SSE stream (live during analysis)
    DELETE /api/matches/{match_id}/segments      → wipe segments before re-analysis

    GET  /api/analysis/{analysis_id}/segments        → list segments for a specific analysis run
    GET  /api/analysis/{analysis_id}/segments/stream → SSE stream (live during analysis run)
    DELETE /api/analysis/{analysis_id}/segments      → wipe segments for an analysis run
"""

import asyncio
import json
import logging
from typing import Dict, Set

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from database import get_db, Connection
from dependencies import get_current_active_user
from models.user import User
from services.segment_service import SegmentService

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory SSE fan-out (one asyncio.Queue per listener)
# ---------------------------------------------------------------------------
_segment_listeners_by_match: Dict[str, Set[asyncio.Queue]] = {}
_segment_listeners_by_analysis: Dict[str, Set[asyncio.Queue]] = {}


def _add_match_listener(match_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _segment_listeners_by_match.setdefault(match_id, set()).add(q)
    return q


def _remove_match_listener(match_id: str, q: asyncio.Queue):
    if match_id in _segment_listeners_by_match:
        _segment_listeners_by_match[match_id].discard(q)
        if not _segment_listeners_by_match[match_id]:
            del _segment_listeners_by_match[match_id]


async def push_segment_event(match_id: str, payload: dict):
    """Back-compat: push by match_id (used by classic backend flow)."""
    await push_match_segment_event(match_id, payload)


async def push_match_segment_event(match_id: str, payload: dict):
    """Called when a SEGMENT_DONE arrives for a match-scoped run."""
    if match_id not in _segment_listeners_by_match:
        return
    data = json.dumps(payload)
    dead: list = []
    for q in _segment_listeners_by_match[match_id]:
        try:
            q.put_nowait(data)
        except Exception:
            dead.append(q)
    for q in dead:
        _remove_match_listener(match_id, q)


def _add_analysis_listener(analysis_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _segment_listeners_by_analysis.setdefault(analysis_id, set()).add(q)
    return q


def _remove_analysis_listener(analysis_id: str, q: asyncio.Queue):
    if analysis_id in _segment_listeners_by_analysis:
        _segment_listeners_by_analysis[analysis_id].discard(q)
        if not _segment_listeners_by_analysis[analysis_id]:
            del _segment_listeners_by_analysis[analysis_id]


async def push_analysis_segment_event(analysis_id: str, payload: dict):
    """Called when a SEGMENT_DONE arrives for an analysis-scoped run."""
    if analysis_id not in _segment_listeners_by_analysis:
        return
    data = json.dumps(payload)
    dead: list = []
    for q in _segment_listeners_by_analysis[analysis_id]:
        try:
            q.put_nowait(data)
        except Exception:
            dead.append(q)
    for q in dead:
        _remove_analysis_listener(analysis_id, q)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.get("/matches/{match_id}/segments")
def list_segments(
    match_id: str,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return all analysis segments for a match, ordered by index."""
    try:
        segments = SegmentService.get_segments(match_id)
        return {"match_id": match_id, "segments": segments, "count": len(segments)}
    except Exception as e:
        logger.error(f"Failed to list segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/{analysis_id}/segments")
def list_segments_for_analysis(
    analysis_id: str,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return all analysis segments for a single analysis run, ordered by index."""
    try:
        segments = SegmentService.get_segments_for_analysis(analysis_id)
        return {
            "analysis_id": analysis_id,
            "segments": segments,
            "count": len(segments),
        }
    except Exception as e:
        logger.error(f"Failed to list analysis segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/matches/{match_id}/segments")
def delete_segments(
    match_id: str,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete all segments for a match (called before re-analysis)."""
    try:
        deleted = SegmentService.delete_match_segments(match_id)
        return {"match_id": match_id, "deleted": deleted}
    except Exception as e:
        logger.error(f"Failed to delete segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/analysis/{analysis_id}/segments")
def delete_segments_for_analysis(
    analysis_id: str,
    db: Connection = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete all segments for an analysis run (used before retry/re-analysis)."""
    try:
        deleted = SegmentService.delete_analysis_segments(analysis_id)
        return {"analysis_id": analysis_id, "deleted": deleted}
    except Exception as e:
        logger.error(f"Failed to delete analysis segments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

@router.get("/matches/{match_id}/segments/stream")
async def stream_segments(
    match_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Server-Sent Events stream for live segment delivery.

    The Flutter client connects here after submitting an analysis job.
    Each SEGMENT_DONE event is pushed as an SSE `data:` line.
    The stream ends with a `event: done` message.
    """
    q = _add_match_listener(match_id)

    async def event_generator():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=600)
                except asyncio.TimeoutError:
                    # Keep-alive ping
                    yield ": keepalive\n\n"
                    continue

                payload = json.loads(data) if isinstance(data, str) else data

                # Check for terminal signal (only end when explicitly signaled).
                if payload.get("type") == "done":
                    yield f"event: done\ndata: {data}\n\n"
                    break

                yield f"data: {data}\n\n"
        finally:
            _remove_match_listener(match_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/analysis/{analysis_id}/segments/stream")
async def stream_segments_for_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    SSE stream for live segment delivery scoped to a specific analysis run.

    This is the preferred stream for the Analyze page and for run previews.
    """
    q = _add_analysis_listener(analysis_id)

    async def event_generator():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=600)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                payload = json.loads(data) if isinstance(data, str) else data
                if payload.get("type") == "done":
                    yield f"event: done\ndata: {data}\n\n"
                    break
                yield f"data: {data}\n\n"
        finally:
            _remove_analysis_listener(analysis_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
