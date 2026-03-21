from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from services.tracking_profile_service import TrackingProfileService
from models.tracking_profile import TrackingProfile, TrackingProfileCreate
from dependencies import get_current_active_user
from models.user import User

router = APIRouter()

@router.post("/tracking-profiles", response_model=TrackingProfile)
def create_profile(profile: TrackingProfileCreate, current_user: User = Depends(get_current_active_user)):
    """Create a new tracking profile with its camera configurations."""
    try:
        return TrackingProfileService.create_profile(profile.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create profile: {str(e)}")

@router.get("/tracking-profiles/{profile_id}", response_model=TrackingProfile)
def get_profile(profile_id: str, current_user: User = Depends(get_current_active_user)):
    """Retrieve a specific tracking profile by its ID."""
    profile = TrackingProfileService.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Tracking profile not found")
    return profile

@router.get("/matches/{match_id}/tracking-profiles", response_model=List[TrackingProfile])
def get_match_profiles(match_id: str, current_user: User = Depends(get_current_active_user)):
    """Retrieve all tracking profiles associated with a match."""
    return TrackingProfileService.get_profiles_for_match(match_id)
