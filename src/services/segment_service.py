"""
Service for managing analysis segments (per-time-window analytics).

Each segment represents a fixed-duration window of the match video.
Metrics are derived from the global tracking JSON (no re-processing).
When severity is high enough, the LLM recommendation is included.
"""

import json
import uuid
import logging
from typing import List, Optional

import pymysql
from database import DB_CONFIG

logger = logging.getLogger(__name__)


class SegmentService:
    """CRUD operations for the analysis_segments table."""

    # ------------------------------------------------------------------
    # Schema bootstrap (idempotent)
    # ------------------------------------------------------------------
    @staticmethod
    def ensure_table():
        """
        Create/migrate the analysis_segments table if it does not exist.

        Segments must be queryable per analysis run (analysis_id). match_id may
        be absent for ad-hoc Analyze runs (no Match row), so we avoid hard FK
        constraints here and use best-effort schema migration.
        """
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_segments (
                        id            CHAR(36) PRIMARY KEY,
                        analysis_id   CHAR(36) NULL,
                        match_id      VARCHAR(64) NULL,
                        segment_index INT NOT NULL,
                        start_sec     FLOAT NOT NULL,
                        end_sec       FLOAT NOT NULL,
                        video_start_sec FLOAT NOT NULL,
                        analysis_json JSON,
                        recommendation TEXT,
                        severity_score DECIMAL(5,4) DEFAULT 0.0,
                        severity_label VARCHAR(32) DEFAULT 'LOW',
                        status        ENUM('PENDING','PROCESSING','COMPLETED','FAILED','EMPTY') DEFAULT 'PENDING',
                        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_analysis_segment (analysis_id, segment_index),
                        INDEX idx_match_segment (match_id, segment_index)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                conn.commit()

                # --- Best-effort migrations for existing schemas ---
                if not SegmentService._has_column(cursor, "analysis_id"):
                    cursor.execute(
                        "ALTER TABLE analysis_segments ADD COLUMN analysis_id CHAR(36) NULL AFTER id"
                    )

                try:
                    cursor.execute("ALTER TABLE analysis_segments MODIFY match_id VARCHAR(64) NULL")
                except Exception:
                    # Already compatible.
                    pass

                SegmentService._drop_match_fk_if_present(cursor)

                # MySQL doesn't consistently support IF NOT EXISTS for indexes; ignore duplicates.
                try:
                    cursor.execute(
                        "CREATE INDEX idx_analysis_segment ON analysis_segments (analysis_id, segment_index)"
                    )
                except Exception:
                    pass
                try:
                    cursor.execute(
                        "CREATE INDEX idx_match_segment ON analysis_segments (match_id, segment_index)"
                    )
                except Exception:
                    pass

                conn.commit()
        except Exception as exc:
            logger.warning(f"ensure_table warning (table may already exist): {exc}")
        finally:
            conn.close()

    @staticmethod
    def _has_column(cursor, column_name: str) -> bool:
        cursor.execute(
            """
            SELECT COUNT(*) AS c
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'analysis_segments'
              AND COLUMN_NAME = %s
            """,
            (column_name,),
        )
        row = cursor.fetchone() or {}
        return int(row.get("c", 0) or 0) > 0

    @staticmethod
    def _drop_match_fk_if_present(cursor) -> None:
        """
        Drop any foreign key constraint that references match_id.

        Analyze page runs may not have a corresponding matches row, so FK
        constraints can block segment persistence.
        """
        try:
            cursor.execute(
                """
                SELECT CONSTRAINT_NAME AS name
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'analysis_segments'
                  AND COLUMN_NAME = 'match_id'
                  AND REFERENCED_TABLE_NAME IS NOT NULL
                """
            )
            rows = cursor.fetchall() or []
            for r in rows:
                name = r.get("name")
                if not name:
                    continue
                try:
                    cursor.execute(f"ALTER TABLE analysis_segments DROP FOREIGN KEY `{name}`")
                except Exception:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    @staticmethod
    def insert_segment(
        *,
        analysis_id: Optional[str] = None,
        match_id: str,
        segment_index: int,
        start_sec: float,
        end_sec: float,
        video_start_sec: float,
        analysis_json: Optional[dict] = None,
        recommendation: Optional[str] = None,
        severity_score: float = 0.0,
        severity_label: str = "LOW",
        status: str = "COMPLETED",
    ) -> dict:
        """Insert a single segment row and return its data dict."""
        SegmentService.ensure_table()
        seg_id = str(uuid.uuid4())
        effective_analysis_id = analysis_id or match_id
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO analysis_segments
                        (id, analysis_id, match_id, segment_index, start_sec, end_sec,
                         video_start_sec, analysis_json, recommendation,
                         severity_score, severity_label, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        seg_id,
                        effective_analysis_id,
                        match_id,
                        segment_index,
                        start_sec,
                        end_sec,
                        video_start_sec,
                        json.dumps(analysis_json) if analysis_json else None,
                        recommendation,
                        severity_score,
                        severity_label,
                        status,
                    ),
                )
                conn.commit()
        finally:
            conn.close()

        return {
            "id": seg_id,
            "analysis_id": effective_analysis_id,
            "match_id": match_id,
            "segment_index": segment_index,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "video_start_sec": video_start_sec,
            "analysis_json": analysis_json,
            "recommendation": recommendation,
            "severity_score": severity_score,
            "severity_label": severity_label,
            "status": status,
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @staticmethod
    def get_segments_for_analysis(analysis_id: str) -> List[dict]:
        """Return all segments for an analysis run, ordered by index."""
        SegmentService.ensure_table()
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, analysis_id, match_id, segment_index, start_sec, end_sec,
                           video_start_sec, analysis_json, recommendation,
                           severity_score, severity_label, status, created_at
                    FROM analysis_segments
                    WHERE analysis_id = %s
                    ORDER BY segment_index ASC
                    """,
                    (analysis_id,),
                )
                rows = cursor.fetchall()
        finally:
            conn.close()
        return SegmentService._rows_to_segments(rows)

    @staticmethod
    def get_segments(match_id: str) -> List[dict]:
        """Return all segments for a match, ordered by index."""
        SegmentService.ensure_table()
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, analysis_id, match_id, segment_index, start_sec, end_sec,
                           video_start_sec, analysis_json, recommendation,
                           severity_score, severity_label, status, created_at
                    FROM analysis_segments
                    WHERE match_id = %s
                    ORDER BY segment_index ASC
                    """,
                    (match_id,),
                )
                rows = cursor.fetchall()
        finally:
            conn.close()
        return SegmentService._rows_to_segments(rows)

    @staticmethod
    def _rows_to_segments(rows: List[dict]) -> List[dict]:
        results: List[dict] = []
        for row in rows or []:
            analysis = row.get("analysis_json")
            if isinstance(analysis, str):
                try:
                    analysis = json.loads(analysis)
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(
                {
                    "id": row.get("id"),
                    "analysis_id": row.get("analysis_id"),
                    "match_id": row.get("match_id"),
                    "segment_index": row.get("segment_index"),
                    "start_sec": float(row.get("start_sec", 0) or 0),
                    "end_sec": float(row.get("end_sec", 0) or 0),
                    "video_start_sec": float(row.get("video_start_sec", 0) or 0),
                    "analysis_json": analysis,
                    "recommendation": row.get("recommendation"),
                    "severity_score": float(row.get("severity_score", 0) or 0),
                    "severity_label": row.get("severity_label", "LOW"),
                    "status": row.get("status", "PENDING"),
                    "created_at": str(row.get("created_at")) if row.get("created_at") else None,
                }
            )
        return results

    @staticmethod
    def delete_analysis_segments(analysis_id: str) -> int:
        """Delete all segments for an analysis run. Returns number of rows deleted."""
        SegmentService.ensure_table()
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM analysis_segments WHERE analysis_id = %s",
                    (analysis_id,),
                )
                conn.commit()
                return cursor.rowcount
        finally:
            conn.close()

    @staticmethod
    def delete_match_segments(match_id: str) -> int:
        """Delete all segments for a match. Returns number of rows deleted."""
        SegmentService.ensure_table()
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM analysis_segments WHERE match_id = %s",
                    (match_id,),
                )
                conn.commit()
                return cursor.rowcount
        finally:
            conn.close()
