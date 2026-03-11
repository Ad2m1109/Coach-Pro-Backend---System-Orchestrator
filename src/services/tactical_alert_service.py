import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Set, Tuple

import pymysql
from fastapi import WebSocket

from database import DB_CONFIG
from services.decision_effectiveness_service import (
    DecisionEffectivenessService,
    parse_match_time,
)

logger = logging.getLogger(__name__)


class TacticalAlertService:
    """Manages real-time tactical alerts, persistence, feedback, and suppression."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TacticalAlertService, cls).__new__(cls)
            cls._instance.active_connections: Dict[str, Set[WebSocket]] = {}
            cls._instance.alert_history: Dict[str, List[dict]] = {}
            cls._instance.suppressed_decision_types: Dict[str, Dict[str, int]] = {}
            cls._instance.effectiveness_service = DecisionEffectivenessService()
            cls._instance._ensure_schema()
        return cls._instance

    def _ensure_schema(self):
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tactical_alerts (
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        match_id VARCHAR(64) NOT NULL,
                        alert_id VARCHAR(64) NOT NULL,
                        decision_id VARCHAR(64) NULL,
                        timestamp VARCHAR(64) NOT NULL,
                        match_time DOUBLE NULL,
                        severity_score DOUBLE NOT NULL,
                        severity_label VARCHAR(32) NOT NULL,
                        category VARCHAR(64) DEFAULT '',
                        decision_type VARCHAR(128) NOT NULL,
                        trigger_metric TEXT NULL,
                        recommended_action TEXT NULL,
                        status VARCHAR(32) NOT NULL,
                        action TEXT NULL,
                        review_countdown INT DEFAULT 0,
                        category_trigger_count INT DEFAULT 0,
                        feedback VARCHAR(32) DEFAULT 'none',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uniq_match_alert (match_id, alert_id),
                        INDEX idx_match_time (match_id, match_time),
                        INDEX idx_match_decision_type (match_id, decision_type)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS decision_feedback (
                        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        user_id VARCHAR(64) NOT NULL,
                        match_id VARCHAR(64) NOT NULL,
                        decision_id VARCHAR(64) NOT NULL,
                        decision_type VARCHAR(128) NOT NULL,
                        action VARCHAR(16) NOT NULL,
                        severity_at_time DOUBLE NOT NULL,
                        match_time DOUBLE NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        decision_effective TINYINT(1) DEFAULT 0,
                        decision_failed TINYINT(1) DEFAULT 0,
                        escalation_after_dismiss TINYINT(1) DEFAULT 0,
                        post_severity_avg DOUBLE NULL,
                        evaluation_windows INT NULL,
                        evaluated_at TIMESTAMP NULL,
                        UNIQUE KEY uniq_user_decision (user_id, decision_id),
                        INDEX idx_user_decision_type (user_id, decision_type),
                        INDEX idx_user_match (user_id, match_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )

                # Backfill/upgrade existing installations where tactical_alerts
                # already existed before new decision feedback columns.
                self._ensure_column(cursor, "tactical_alerts", "decision_id", "VARCHAR(64) NULL")
                self._ensure_column(cursor, "tactical_alerts", "match_time", "DOUBLE NULL")
                self._ensure_column(cursor, "tactical_alerts", "trigger_metric", "TEXT NULL")
                self._ensure_column(cursor, "tactical_alerts", "recommended_action", "TEXT NULL")
                self._ensure_column(cursor, "tactical_alerts", "feedback", "VARCHAR(32) DEFAULT 'none'")

                # Ensure expected indexes exist for query paths.
                self._ensure_index(cursor, "tactical_alerts", "idx_match_time", "CREATE INDEX idx_match_time ON tactical_alerts(match_id, match_time)")
                self._ensure_index(cursor, "tactical_alerts", "idx_match_decision_type", "CREATE INDEX idx_match_decision_type ON tactical_alerts(match_id, decision_type)")
                self._ensure_index(cursor, "tactical_alerts", "uniq_match_alert", "CREATE UNIQUE INDEX uniq_match_alert ON tactical_alerts(match_id, alert_id)")

                # Backfill/upgrade decision_feedback table in case it pre-exists.
                self._ensure_column(cursor, "decision_feedback", "match_id", "VARCHAR(64) NOT NULL DEFAULT ''")
                self._ensure_column(cursor, "decision_feedback", "decision_effective", "TINYINT(1) DEFAULT 0")
                self._ensure_column(cursor, "decision_feedback", "decision_failed", "TINYINT(1) DEFAULT 0")
                self._ensure_column(cursor, "decision_feedback", "escalation_after_dismiss", "TINYINT(1) DEFAULT 0")
                self._ensure_column(cursor, "decision_feedback", "post_severity_avg", "DOUBLE NULL")
                self._ensure_column(cursor, "decision_feedback", "evaluation_windows", "INT NULL")
                self._ensure_column(cursor, "decision_feedback", "evaluated_at", "TIMESTAMP NULL")
                self._ensure_index(cursor, "decision_feedback", "uniq_user_decision", "CREATE UNIQUE INDEX uniq_user_decision ON decision_feedback(user_id, decision_id)")
                self._ensure_index(cursor, "decision_feedback", "idx_user_decision_type", "CREATE INDEX idx_user_decision_type ON decision_feedback(user_id, decision_type)")
                self._ensure_index(cursor, "decision_feedback", "idx_user_match", "CREATE INDEX idx_user_match ON decision_feedback(user_id, match_id)")
                conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ensure_column(cursor, table_name: str, column_name: str, ddl: str):
        cursor.execute(
            """
            SELECT COUNT(*) AS count
            FROM information_schema.columns
            WHERE table_schema = DATABASE()
              AND table_name = %s
              AND column_name = %s
            """,
            (table_name, column_name),
        )
        row = cursor.fetchone()
        if not row or int(row["count"]) == 0:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")

    @staticmethod
    def _ensure_index(cursor, table_name: str, index_name: str, create_sql: str):
        cursor.execute(
            """
            SELECT COUNT(*) AS count
            FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = %s
              AND index_name = %s
            """,
            (table_name, index_name),
        )
        row = cursor.fetchone()
        if not row or int(row["count"]) == 0:
            cursor.execute(create_sql)

    async def connect(self, match_id: str, websocket: WebSocket):
        await websocket.accept()
        if match_id not in self.active_connections:
            self.active_connections[match_id] = set()
        self.active_connections[match_id].add(websocket)
        logger.info(
            "WebSocket connected for match %s. Active: %s",
            match_id,
            len(self.active_connections[match_id]),
        )

    def disconnect(self, match_id: str, websocket: WebSocket):
        if match_id in self.active_connections:
            self.active_connections[match_id].discard(websocket)
            if not self.active_connections[match_id]:
                del self.active_connections[match_id]
        logger.info("WebSocket disconnected for match %s", match_id)

    async def broadcast_alert(self, match_id: str, alert: dict):
        normalized = self._normalize_alert_payload(match_id, alert)

        if self._is_suppressed(
            match_id=match_id,
            decision_type=normalized["decision_type"],
        ):
            logger.info(
                "Suppressed alert for match=%s decision_type=%s",
                match_id,
                normalized["decision_type"],
            )
            return

        if match_id not in self.alert_history:
            self.alert_history[match_id] = []

        already_known = any(
            a.get("alert_id") == normalized["alert_id"] for a in self.alert_history[match_id]
        )
        if not already_known:
            self.alert_history[match_id].append(normalized)
            self._persist_alert(normalized)
        else:
            # Keep local memory consistent when updates arrive for same alert.
            self._merge_existing_alert(match_id, normalized)

        if match_id in self.active_connections:
            message = json.dumps(normalized)
            disconnected_websockets = set()
            for websocket in self.active_connections[match_id]:
                try:
                    await websocket.send_text(message)
                except Exception as exc:
                    logger.warning("Failed to send alert to websocket: %s", exc)
                    disconnected_websockets.add(websocket)
            for ws in disconnected_websockets:
                self.disconnect(match_id, ws)

    def get_alert_history(self, match_id: str, user_id: Optional[str] = None) -> List[dict]:
        if user_id is not None:
            conn = pymysql.connect(**DB_CONFIG)
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT
                            ta.match_id, ta.alert_id, ta.decision_id, ta.timestamp, ta.match_time,
                            ta.severity_score, ta.severity_label, ta.category, ta.decision_type,
                            ta.trigger_metric, ta.recommended_action, ta.status, ta.action,
                            ta.review_countdown, ta.category_trigger_count, ta.feedback, ta.created_at,
                            df.action AS feedback_action,
                            df.decision_effective,
                            df.decision_failed,
                            df.escalation_after_dismiss
                        FROM tactical_alerts ta
                        LEFT JOIN decision_feedback df
                          ON df.user_id = %s
                         AND (df.decision_id = ta.decision_id OR df.decision_id = ta.alert_id)
                        WHERE ta.match_id = %s
                        ORDER BY ta.created_at ASC
                        """,
                        (user_id, match_id),
                    )
                    rows = cursor.fetchall()
            finally:
                conn.close()

            for row in rows:
                action = row.get("feedback_action")
                if action == "ACCEPT":
                    row["feedback"] = "accepted"
                elif action == "DISMISS":
                    row["feedback"] = "dismissed"
            return rows

        if match_id not in self.alert_history or not self.alert_history[match_id]:
            conn = pymysql.connect(**DB_CONFIG)
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT
                            match_id, alert_id, decision_id, timestamp, match_time,
                            severity_score, severity_label, category, decision_type,
                            trigger_metric, recommended_action, status, action,
                            review_countdown, category_trigger_count, feedback,
                            created_at
                        FROM tactical_alerts
                        WHERE match_id = %s
                        ORDER BY created_at ASC
                        """,
                        (match_id,),
                    )
                    self.alert_history[match_id] = cursor.fetchall()
            finally:
                conn.close()
        return self.alert_history.get(match_id, [])

    def clear_history(self, match_id: str):
        if match_id in self.alert_history:
            del self.alert_history[match_id]
        if match_id in self.suppressed_decision_types:
            del self.suppressed_decision_types[match_id]

        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM tactical_alerts WHERE match_id = %s", (match_id,))
                conn.commit()
        finally:
            conn.close()

    async def submit_feedback(
        self,
        *,
        user_id: str,
        decision_id: str,
        action: str,
        match_time: Optional[float],
        match_id: Optional[str] = None,
    ) -> dict:
        """
        Idempotent decision feedback write scoped by user_id + decision_id.
        Also updates tactical_alerts feedback state and triggers async evaluation.
        """
        normalized_action = action.upper().strip()
        if normalized_action not in {"ACCEPT", "DISMISS"}:
            raise ValueError("action must be ACCEPT or DISMISS")

        alert_record = self._get_decision_alert(decision_id=decision_id, match_id=match_id)
        if not alert_record:
            raise LookupError("decision_id not found")

        resolved_match_id = alert_record["match_id"]
        decision_type = alert_record["decision_type"]
        severity_at_time = float(alert_record["severity_score"])
        resolved_match_time = (
            match_time if match_time is not None else parse_match_time(alert_record.get("match_time"))
        )

        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, action FROM decision_feedback
                    WHERE user_id = %s AND decision_id = %s
                    LIMIT 1
                    """,
                    (user_id, decision_id),
                )
                existing = cursor.fetchone()
                if existing:
                    return {
                        "status": "success",
                        "idempotent": True,
                        "decision_id": decision_id,
                        "action": existing["action"],
                        "decision_type": decision_type,
                        "match_id": resolved_match_id,
                    }

                cursor.execute(
                    """
                    INSERT INTO decision_feedback
                        (user_id, match_id, decision_id, decision_type, action, severity_at_time, match_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        resolved_match_id,
                        decision_id,
                        decision_type,
                        normalized_action,
                        severity_at_time,
                        resolved_match_time,
                    ),
                )

                feedback_value = "accepted" if normalized_action == "ACCEPT" else "dismissed"
                cursor.execute(
                    """
                    UPDATE tactical_alerts
                    SET feedback = %s
                    WHERE match_id = %s AND (decision_id = %s OR alert_id = %s)
                    """,
                    (feedback_value, resolved_match_id, decision_id, decision_id),
                )
                conn.commit()
        finally:
            conn.close()

        self._update_memory_feedback(resolved_match_id, decision_id, normalized_action)

        if normalized_action == "DISMISS":
            self._register_suppression(
                match_id=resolved_match_id,
                decision_type=decision_type,
                windows=2,
            )

        asyncio.create_task(
            self.effectiveness_service.evaluate_feedback_async(
                user_id=str(user_id),
                match_id=resolved_match_id,
                decision_id=decision_id,
                decision_type=decision_type,
                action=normalized_action,
                baseline_severity=severity_at_time,
                baseline_match_time=resolved_match_time,
            )
        )

        return {
            "status": "success",
            "idempotent": False,
            "decision_id": decision_id,
            "action": normalized_action,
            "decision_type": decision_type,
            "match_id": resolved_match_id,
        }

    def get_decision_metrics(self, user_id: str, match_id: Optional[str] = None) -> dict:
        rows = self.effectiveness_service.get_metrics(user_id=str(user_id), match_id=match_id)
        totals = {
            "total_feedbacks": sum(int(r["total_feedbacks"]) for r in rows),
            "acceptance_rate": 0.0,
            "dismissal_rate": 0.0,
            "effectiveness_rate": 0.0,
            "escalation_after_dismiss_rate": 0.0,
        }
        if totals["total_feedbacks"] > 0:
            totals["acceptance_rate"] = sum(r["acceptance_rate"] * r["total_feedbacks"] for r in rows) / totals["total_feedbacks"]
            totals["dismissal_rate"] = sum(r["dismissal_rate"] * r["total_feedbacks"] for r in rows) / totals["total_feedbacks"]
            totals["effectiveness_rate"] = sum(r["effectiveness_rate"] * r["total_feedbacks"] for r in rows) / totals["total_feedbacks"]

            total_dismissed_weight = sum(r["dismissal_rate"] * r["total_feedbacks"] for r in rows)
            if total_dismissed_weight > 0:
                weighted_escalation = 0.0
                for item in rows:
                    dismissed_count = item["dismissal_rate"] * item["total_feedbacks"]
                    weighted_escalation += item["escalation_after_dismiss_rate"] * dismissed_count
                totals["escalation_after_dismiss_rate"] = weighted_escalation / total_dismissed_weight

        return {"summary": totals, "by_decision_type": rows}

    def _normalize_alert_payload(self, match_id: str, alert: dict) -> dict:
        decision_id = alert.get("decision_id") or alert.get("alert_id")
        timestamp = alert.get("timestamp") or ""
        match_time = alert.get("match_time")
        if match_time is None:
            match_time = parse_match_time(timestamp)

        return {
            "match_id": match_id,
            "alert_id": alert.get("alert_id"),
            "decision_id": decision_id,
            "timestamp": timestamp,
            "match_time": match_time,
            "severity_score": float(alert.get("severity_score", 0.0)),
            "severity_label": alert.get("severity_label", "LOW"),
            "category": alert.get("category", ""),
            "decision_type": alert.get("decision_type", "NONE"),
            "trigger_metric": alert.get("trigger_metric"),
            "recommended_action": alert.get("recommended_action") or alert.get("action"),
            "status": alert.get("status", "ACTIVE"),
            "action": alert.get("action", ""),
            "review_countdown": int(alert.get("review_countdown", 0)),
            "category_trigger_count": int(alert.get("category_trigger_count", 0)),
            "feedback": alert.get("feedback", "none"),
            # Optional UI-only tactical layout payload (not persisted to DB schema).
            "players": alert.get("players"),
            "ball": alert.get("ball"),
            "zone": alert.get("zone"),
        }

    def _persist_alert(self, alert: dict):
        conn = pymysql.connect(**DB_CONFIG)
        params = (
            alert["match_id"],
            alert["alert_id"],
            alert["decision_id"],
            alert["timestamp"],
            alert["match_time"],
            alert["severity_score"],
            alert["severity_label"],
            alert["category"],
            alert["decision_type"],
            alert["trigger_metric"],
            alert["recommended_action"],
            alert["status"],
            alert["action"],
            alert["review_countdown"],
            alert["category_trigger_count"],
            alert["feedback"],
        )
        try:
            with conn.cursor() as cursor:
                insert_sql = """
                    INSERT INTO tactical_alerts
                        (match_id, alert_id, decision_id, timestamp, match_time,
                         severity_score, severity_label, category, decision_type,
                         trigger_metric, recommended_action, status, action,
                         review_countdown, category_trigger_count, feedback)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        decision_id = VALUES(decision_id),
                        match_time = VALUES(match_time),
                        severity_score = VALUES(severity_score),
                        severity_label = VALUES(severity_label),
                        category = VALUES(category),
                        decision_type = VALUES(decision_type),
                        trigger_metric = VALUES(trigger_metric),
                        recommended_action = VALUES(recommended_action),
                        status = VALUES(status),
                        action = VALUES(action),
                        review_countdown = VALUES(review_countdown),
                        category_trigger_count = VALUES(category_trigger_count),
                        feedback = VALUES(feedback)
                """
                try:
                    cursor.execute(insert_sql, params)
                except pymysql.err.OperationalError as exc:
                    # Compatibility path for legacy schema from full_creation.sql:
                    # tactical_alerts.id is CHAR(36) NOT NULL without default.
                    if int(exc.args[0]) == 1364 and "Field 'id' doesn't have a default value" in str(exc):
                        insert_with_id_sql = """
                            INSERT INTO tactical_alerts
                                (id, match_id, alert_id, decision_id, timestamp, match_time,
                                 severity_score, severity_label, category, decision_type,
                                 trigger_metric, recommended_action, status, action,
                                 review_countdown, category_trigger_count, feedback)
                            VALUES
                                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                decision_id = VALUES(decision_id),
                                match_time = VALUES(match_time),
                                severity_score = VALUES(severity_score),
                                severity_label = VALUES(severity_label),
                                category = VALUES(category),
                                decision_type = VALUES(decision_type),
                                trigger_metric = VALUES(trigger_metric),
                                recommended_action = VALUES(recommended_action),
                                status = VALUES(status),
                                action = VALUES(action),
                                review_countdown = VALUES(review_countdown),
                                category_trigger_count = VALUES(category_trigger_count),
                                feedback = VALUES(feedback)
                        """
                        cursor.execute(
                            insert_with_id_sql,
                            (str(uuid.uuid4()), *params),
                        )
                    else:
                        raise
                conn.commit()
        except Exception as exc:
            logger.error("Failed to persist alert to DB: %s", exc)
        finally:
            conn.close()

    def _merge_existing_alert(self, match_id: str, updated: dict):
        for idx, existing in enumerate(self.alert_history.get(match_id, [])):
            if existing.get("alert_id") == updated["alert_id"]:
                merged = dict(existing)
                merged.update(updated)
                self.alert_history[match_id][idx] = merged
                self._persist_alert(merged)
                return

    def _get_decision_alert(self, decision_id: str, match_id: Optional[str]) -> Optional[dict]:
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                if match_id:
                    cursor.execute(
                        """
                        SELECT * FROM tactical_alerts
                        WHERE match_id = %s AND (decision_id = %s OR alert_id = %s)
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (match_id, decision_id, decision_id),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM tactical_alerts
                        WHERE decision_id = %s OR alert_id = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (decision_id, decision_id),
                    )
                return cursor.fetchone()
        finally:
            conn.close()

    def _update_memory_feedback(self, match_id: str, decision_id: str, action: str):
        feedback_value = "accepted" if action == "ACCEPT" else "dismissed"
        for alert in self.alert_history.get(match_id, []):
            if alert.get("decision_id") == decision_id or alert.get("alert_id") == decision_id:
                alert["feedback"] = feedback_value

    def _register_suppression(
        self,
        *,
        match_id: str,
        decision_type: str,
        windows: int,
    ):
        if match_id not in self.suppressed_decision_types:
            self.suppressed_decision_types[match_id] = {}
        self.suppressed_decision_types[match_id][decision_type] = int(windows)

    def _is_suppressed(self, *, match_id: str, decision_type: str) -> bool:
        if match_id not in self.suppressed_decision_types:
            return False
        if decision_type not in self.suppressed_decision_types[match_id]:
            return False
        remaining = self.suppressed_decision_types[match_id][decision_type]
        if remaining > 0:
            self.suppressed_decision_types[match_id][decision_type] = remaining - 1
            return True
        del self.suppressed_decision_types[match_id][decision_type]
        if not self.suppressed_decision_types[match_id]:
            del self.suppressed_decision_types[match_id]
        return False
