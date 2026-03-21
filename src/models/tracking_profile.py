from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class CalibrationPoint(BaseModel):
    u: float  # pixel x
    v: float  # pixel y
    x: float  # pitch x (meters)
    y: float  # pitch y (meters)

class ROICoordinate(BaseModel):
    x: float
    y: float

class ROIZone(BaseModel):
    points: List[ROICoordinate]
    label: str

class CameraConfigBase(BaseModel):
    label: str
    video_source: str
    sync_offset_ms: int = 0
    calibration: List[CalibrationPoint] = []
    roi: List[ROIZone] = []

class CameraConfigCreate(CameraConfigBase):
    pass

class CameraConfig(CameraConfigBase):
    id: str
    profile_id: str

class TrackingProfileBase(BaseModel):
    match_id: Optional[str] = None
    name: str
    engine_settings: Optional[Dict] = {}
    is_active: bool = True

class TrackingProfileCreate(TrackingProfileBase):
    cameras: List[CameraConfigCreate] = []

class TrackingProfile(TrackingProfileBase):
    id: str
    cameras: List[CameraConfig] = []
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
