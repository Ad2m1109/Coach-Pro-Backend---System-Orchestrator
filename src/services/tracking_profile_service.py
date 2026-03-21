import json
import uuid
import logging
from typing import List, Optional
from datetime import datetime

import pymysql
from database import DB_CONFIG

logger = logging.getLogger(__name__)

class TrackingProfileService:
    """CRUD operations for tracking_profiles and camera_configs tables."""

    @staticmethod
    def ensure_tables():
        """Create/migrate tracking_profiles and camera_configs tables."""
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                # 1. Tracking Profiles Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tracking_profiles (
                        id              CHAR(36) PRIMARY KEY,
                        match_id        CHAR(36) NULL,
                        name            VARCHAR(255) NOT NULL,
                        engine_settings JSON,
                        is_active       BOOLEAN DEFAULT TRUE,
                        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_match_profile (match_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

                # 2. Camera Configurations Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS camera_configs (
                        id              CHAR(36) PRIMARY KEY,
                        profile_id      CHAR(36) NOT NULL,
                        label           VARCHAR(64) NOT NULL,
                        video_source    VARCHAR(255) NOT NULL,
                        sync_offset_ms  INT DEFAULT 0,
                        calibration_json JSON,
                        roi_json        JSON,
                        order_index     INT DEFAULT 0,
                        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (profile_id) REFERENCES tracking_profiles(id) ON DELETE CASCADE,
                        INDEX idx_profile_camera (profile_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                conn.commit()
        except Exception as exc:
            logger.error(f"ensure_tables error: {exc}")
        finally:
            conn.close()

    @staticmethod
    def create_profile(profile_data: dict) -> dict:
        """
        Create a new tracking profile and its associated camera configs.
        profile_data should follow TrackingProfileCreate schema.
        """
        TrackingProfileService.ensure_tables()
        profile_id = str(uuid.uuid4())
        cameras = profile_data.get("cameras", [])
        
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                # Insert Profile
                cursor.execute(
                    """
                    INSERT INTO tracking_profiles (id, match_id, name, engine_settings, is_active)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        profile_id,
                        profile_data.get("match_id"),
                        profile_data.get("name"),
                        json.dumps(profile_data.get("engine_settings", {})),
                        profile_data.get("is_active", True)
                    )
                )

                # Insert Cameras
                for i, cam in enumerate(cameras):
                    cam_id = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO camera_configs 
                        (id, profile_id, label, video_source, sync_offset_ms, calibration_json, roi_json, order_index)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            cam_id,
                            profile_id,
                            cam.get("label"),
                            cam.get("video_source"),
                            cam.get("sync_offset_ms", 0),
                            json.dumps(cam.get("calibration", [])),
                            json.dumps(cam.get("roi", [])),
                            i
                        )
                    )
                conn.commit()
        finally:
            conn.close()

        return TrackingProfileService.get_profile(profile_id)

    @staticmethod
    def get_profile(profile_id: str) -> Optional[dict]:
        """Fetch a full profile with its camera configurations."""
        TrackingProfileService.ensure_tables()
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                # Get Profile
                cursor.execute("SELECT * FROM tracking_profiles WHERE id = %s", (profile_id,))
                profile = cursor.fetchone()
                if not profile:
                    return None
                
                # Get Cameras
                cursor.execute("SELECT * FROM camera_configs WHERE profile_id = %s ORDER BY order_index", (profile_id,))
                cameras = cursor.fetchall()
                
                return TrackingProfileService._map_profile(profile, cameras)
        finally:
            conn.close()

    @staticmethod
    def get_profiles_for_match(match_id: str) -> List[dict]:
        """Fetch all profiles associated with a match."""
        TrackingProfileService.ensure_tables()
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM tracking_profiles WHERE match_id = %s ORDER BY created_at DESC", (match_id,))
                ids = [row['id'] for row in cursor.fetchall()]
                
                return [TrackingProfileService.get_profile(pid) for pid in ids]
        finally:
            conn.close()

    @staticmethod
    def _map_profile(profile: dict, cameras: List[dict]) -> dict:
        """Map raw DB rows to the TrackingProfile dictionary structure."""
        # Parse JSON fields
        engine_settings = profile.get("engine_settings")
        if isinstance(engine_settings, str):
            engine_settings = json.loads(engine_settings)
            
        mapped_cameras = []
        for cam in cameras:
            cal = cam.get("calibration_json")
            if isinstance(cal, str): cal = json.loads(cal)
            roi = cam.get("roi_json")
            if isinstance(roi, str): roi = json.loads(roi)
            
            mapped_cameras.append({
                "id": cam.get("id"),
                "label": cam.get("label"),
                "video_source": cam.get("video_source"),
                "sync_offset_ms": cam.get("sync_offset_ms"),
                "calibration": cal or [],
                "roi": roi or [],
                "order_index": cam.get("order_index")
            })

        return {
            "id": profile.get("id"),
            "match_id": profile.get("match_id"),
            "name": profile.get("name"),
            "engine_settings": engine_settings or {},
            "is_active": bool(profile.get("is_active")),
            "cameras": mapped_cameras,
            "created_at": profile.get("created_at").isoformat() if profile.get("created_at") else None
        }
