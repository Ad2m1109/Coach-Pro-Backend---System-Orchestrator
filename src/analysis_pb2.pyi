from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AnalysisRequest(_message.Message):
    __slots__ = ("video_path", "match_id", "frame_limit", "skip_json", "confidence_threshold", "calibration_json", "roi_json", "ball_confidence", "max_lost_frames", "enable_reid", "target_team", "camera_count", "camera_type")
    VIDEO_PATH_FIELD_NUMBER: _ClassVar[int]
    MATCH_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_LIMIT_FIELD_NUMBER: _ClassVar[int]
    SKIP_JSON_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    CALIBRATION_JSON_FIELD_NUMBER: _ClassVar[int]
    ROI_JSON_FIELD_NUMBER: _ClassVar[int]
    BALL_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    MAX_LOST_FRAMES_FIELD_NUMBER: _ClassVar[int]
    ENABLE_REID_FIELD_NUMBER: _ClassVar[int]
    TARGET_TEAM_FIELD_NUMBER: _ClassVar[int]
    CAMERA_COUNT_FIELD_NUMBER: _ClassVar[int]
    CAMERA_TYPE_FIELD_NUMBER: _ClassVar[int]
    video_path: str
    match_id: str
    frame_limit: int
    skip_json: bool
    confidence_threshold: float
    calibration_json: str
    roi_json: str
    ball_confidence: float
    max_lost_frames: int
    enable_reid: bool
    target_team: str
    camera_count: int
    camera_type: str
    def __init__(self, video_path: _Optional[str] = ..., match_id: _Optional[str] = ..., frame_limit: _Optional[int] = ..., skip_json: bool = ..., confidence_threshold: _Optional[float] = ..., calibration_json: _Optional[str] = ..., roi_json: _Optional[str] = ..., ball_confidence: _Optional[float] = ..., max_lost_frames: _Optional[int] = ..., enable_reid: bool = ..., target_team: _Optional[str] = ..., camera_count: _Optional[int] = ..., camera_type: _Optional[str] = ...) -> None: ...

class AnalysisResponse(_message.Message):
    __slots__ = ("match_id", "status", "progress", "message", "alert", "result")
    MATCH_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ALERT_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    match_id: str
    status: str
    progress: float
    message: str
    alert: TacticalAlert
    result: AnalysisResult
    def __init__(self, match_id: _Optional[str] = ..., status: _Optional[str] = ..., progress: _Optional[float] = ..., message: _Optional[str] = ..., alert: _Optional[_Union[TacticalAlert, _Mapping]] = ..., result: _Optional[_Union[AnalysisResult, _Mapping]] = ...) -> None: ...

class TacticalAlert(_message.Message):
    __slots__ = ("alert_id", "timestamp", "severity_score", "severity_label", "category", "decision_type", "status", "action", "review_countdown", "category_trigger_count")
    ALERT_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_LABEL_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    DECISION_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    REVIEW_COUNTDOWN_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_TRIGGER_COUNT_FIELD_NUMBER: _ClassVar[int]
    alert_id: str
    timestamp: str
    severity_score: float
    severity_label: str
    category: str
    decision_type: str
    status: str
    action: str
    review_countdown: int
    category_trigger_count: int
    def __init__(self, alert_id: _Optional[str] = ..., timestamp: _Optional[str] = ..., severity_score: _Optional[float] = ..., severity_label: _Optional[str] = ..., category: _Optional[str] = ..., decision_type: _Optional[str] = ..., status: _Optional[str] = ..., action: _Optional[str] = ..., review_countdown: _Optional[int] = ..., category_trigger_count: _Optional[int] = ...) -> None: ...

class AnalysisResult(_message.Message):
    __slots__ = ("match_id", "total_frames", "players_tracked", "tracking_video_path", "tracking_json_path", "backline_video_path", "heatmap_video_path", "possession_analysis_path", "animation_video_path", "processing_time_seconds", "fps", "error_message", "tactical_advisory_path")
    MATCH_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FRAMES_FIELD_NUMBER: _ClassVar[int]
    PLAYERS_TRACKED_FIELD_NUMBER: _ClassVar[int]
    TRACKING_VIDEO_PATH_FIELD_NUMBER: _ClassVar[int]
    TRACKING_JSON_PATH_FIELD_NUMBER: _ClassVar[int]
    BACKLINE_VIDEO_PATH_FIELD_NUMBER: _ClassVar[int]
    HEATMAP_VIDEO_PATH_FIELD_NUMBER: _ClassVar[int]
    POSSESSION_ANALYSIS_PATH_FIELD_NUMBER: _ClassVar[int]
    ANIMATION_VIDEO_PATH_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    FPS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TACTICAL_ADVISORY_PATH_FIELD_NUMBER: _ClassVar[int]
    match_id: str
    total_frames: int
    players_tracked: int
    tracking_video_path: str
    tracking_json_path: str
    backline_video_path: str
    heatmap_video_path: str
    possession_analysis_path: str
    animation_video_path: str
    processing_time_seconds: float
    fps: float
    error_message: str
    tactical_advisory_path: str
    def __init__(self, match_id: _Optional[str] = ..., total_frames: _Optional[int] = ..., players_tracked: _Optional[int] = ..., tracking_video_path: _Optional[str] = ..., tracking_json_path: _Optional[str] = ..., backline_video_path: _Optional[str] = ..., heatmap_video_path: _Optional[str] = ..., possession_analysis_path: _Optional[str] = ..., animation_video_path: _Optional[str] = ..., processing_time_seconds: _Optional[float] = ..., fps: _Optional[float] = ..., error_message: _Optional[str] = ..., tactical_advisory_path: _Optional[str] = ...) -> None: ...
