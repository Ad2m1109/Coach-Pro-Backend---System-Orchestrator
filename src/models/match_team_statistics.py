from pydantic import BaseModel
from typing import Optional, Dict, Any
from decimal import Decimal

class MatchTeamStatisticsBase(BaseModel):
    match_id: str
    team_id: str
    possession_percentage: Optional[float] = None
    total_shots: Optional[int] = None
    shots_on_target: Optional[int] = None
    expected_goals: Optional[float] = None
    pressures: Optional[int] = None
    final_third_passes: Optional[int] = None
    pass_network_data: Optional[Dict[str, Any]] = None
    zone_analysis_data: Optional[Dict[str, Any]] = None
    tactical_weakness_data: Optional[Dict[str, Any]] = None
    high_turnover_zones_data: Optional[Dict[str, Any]] = None
    set_piece_xg_breakdown_data: Optional[Dict[str, Any]] = None
    transition_speed_data: Optional[Dict[str, Any]] = None
    build_up_patterns: Optional[Dict[str, Any]] = None
    defensive_block_patterns: Optional[Dict[str, Any]] = None

class MatchTeamStatisticsCreate(MatchTeamStatisticsBase):
    pass

class MatchTeamStatistics(MatchTeamStatisticsBase):
    id: str

    class Config:
        from_attributes = True
