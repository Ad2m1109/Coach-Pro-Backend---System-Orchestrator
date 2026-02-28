from pydantic import BaseModel
from typing import Optional
from decimal import Decimal

class PlayerMatchStatisticsBase(BaseModel):
    match_id: str
    player_id: str
    minutes_played: Optional[int] = None
    shots: Optional[int] = 0
    shots_on_target: Optional[int] = 0
    passes: Optional[int] = 0
    accurate_passes: Optional[int] = 0
    tackles: Optional[int] = 0
    interceptions: Optional[int] = 0
    clearances: Optional[int] = 0
    saves: Optional[int] = 0
    fouls_committed: Optional[int] = 0
    fouls_suffered: Optional[int] = 0
    offsides: Optional[int] = 0
    distance_covered_km: Optional[float] = None
    sprint_count: Optional[int] = 0
    sprint_distance_m: Optional[float] = 0.0
    avg_speed_kmh: Optional[float] = 0.0
    max_speed_kmh: Optional[float] = 0.0
    notes: Optional[str] = None
    rating: Optional[float] = None

class PlayerMatchStatisticsCreate(PlayerMatchStatisticsBase):
    pass

class PlayerMatchStatistics(PlayerMatchStatisticsBase):
    id: str

    class Config:
        from_attributes = True
