import asyncio
import logging
from typing import Dict, List, Optional

import pymysql

from database import DB_CONFIG

logger = logging.getLogger(__name__)


def parse_match_time(value: Optional[object]) -> Optional[float]:
    """Convert tactical match_time representation to seconds."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    # Accept "MM:SS-MM:SS" and use start of window.
    if "-" in text:
        text = text.split("-", 1)[0]

    if ":" in text:
        try:
            minute, second = text.split(":", 1)
            return float(int(minute) * 60 + int(second))
        except Exception:
            return None

    try:
        return float(text)
    except Exception:
        return None


class DecisionEffectivenessService:
    """Asynchronous post-feedback evaluator and metrics provider."""

    async def evaluate_feedback_async(
        self,
        user_id: str,
        match_id: str,
        decision_id: str,
        decision_type: str,
        action: str,
        baseline_severity: float,
        baseline_match_time: Optional[float],
    ) -> None:
        """
        Evaluate the decision after 2-3 subsequent windows.
        Runs in background to avoid blocking real-time alert pipeline.
        """
        required_windows = 2
        max_windows = 3
        retries = 15
        retry_delay_seconds = 2.0

        observed: List[float] = []
        for _ in range(retries):
            observed = self._fetch_future_window_severities(
                match_id=match_id,
                baseline_match_time=baseline_match_time,
                limit=max_windows,
            )
            if len(observed) >= required_windows:
                break
            await asyncio.sleep(retry_delay_seconds)

        if not observed:
            return

        post_avg = sum(observed) / len(observed)
        decision_effective = 0
        decision_failed = 0
        escalation_after_dismiss = 0

        if action == "ACCEPT":
            if baseline_severity > 0:
                drop_ratio = (baseline_severity - post_avg) / baseline_severity
                if drop_ratio >= 0.20:
                    decision_effective = 1
            if post_avg > baseline_severity:
                decision_failed = 1

        if action == "DISMISS":
            if post_avg > baseline_severity:
                escalation_after_dismiss = 1
                decision_failed = 1
            else:
                decision_effective = 1

        self._persist_evaluation(
            user_id=user_id,
            decision_id=decision_id,
            decision_effective=decision_effective,
            decision_failed=decision_failed,
            escalation_after_dismiss=escalation_after_dismiss,
            post_severity_avg=post_avg,
            evaluation_windows=len(observed),
        )

    def get_metrics(self, user_id: str, match_id: Optional[str] = None) -> List[Dict]:
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                where = "WHERE user_id = %s"
                params: List[object] = [user_id]
                if match_id:
                    where += " AND match_id = %s"
                    params.append(match_id)

                cursor.execute(
                    f"""
                    SELECT
                        decision_type,
                        COUNT(*) AS total_feedbacks,
                        SUM(CASE WHEN action = 'ACCEPT' THEN 1 ELSE 0 END) AS accepted_count,
                        SUM(CASE WHEN action = 'DISMISS' THEN 1 ELSE 0 END) AS dismissed_count,
                        SUM(CASE WHEN decision_effective = 1 THEN 1 ELSE 0 END) AS effective_count,
                        SUM(CASE WHEN action = 'DISMISS' AND escalation_after_dismiss = 1 THEN 1 ELSE 0 END) AS escalation_after_dismiss_count
                    FROM decision_feedback
                    {where}
                    GROUP BY decision_type
                    ORDER BY total_feedbacks DESC
                    """,
                    tuple(params),
                )
                rows = cursor.fetchall()
        finally:
            conn.close()

        metrics: List[Dict] = []
        for row in rows:
            total = int(row["total_feedbacks"] or 0)
            accepted = int(row["accepted_count"] or 0)
            dismissed = int(row["dismissed_count"] or 0)
            effective = int(row["effective_count"] or 0)
            escalation = int(row["escalation_after_dismiss_count"] or 0)
            metrics.append(
                {
                    "decision_type": row["decision_type"],
                    "acceptance_rate": (accepted / total) if total else 0.0,
                    "dismissal_rate": (dismissed / total) if total else 0.0,
                    "effectiveness_rate": (effective / total) if total else 0.0,
                    "escalation_after_dismiss_rate": (escalation / dismissed) if dismissed else 0.0,
                    "total_feedbacks": total,
                }
            )
        return metrics

    def _fetch_future_window_severities(
        self,
        match_id: str,
        baseline_match_time: Optional[float],
        limit: int,
    ) -> List[float]:
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                if baseline_match_time is None:
                    cursor.execute(
                        """
                        SELECT severity_score
                        FROM tactical_alerts
                        WHERE match_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (match_id, limit),
                    )
                    rows = list(reversed(cursor.fetchall()))
                else:
                    cursor.execute(
                        """
                        SELECT severity_score
                        FROM tactical_alerts
                        WHERE match_id = %s AND match_time > %s
                        ORDER BY match_time ASC
                        LIMIT %s
                        """,
                        (match_id, baseline_match_time, limit),
                    )
                    rows = cursor.fetchall()
        finally:
            conn.close()

        return [float(r["severity_score"]) for r in rows]

    def _persist_evaluation(
        self,
        user_id: str,
        decision_id: str,
        decision_effective: int,
        decision_failed: int,
        escalation_after_dismiss: int,
        post_severity_avg: float,
        evaluation_windows: int,
    ) -> None:
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE decision_feedback
                    SET decision_effective = %s,
                        decision_failed = %s,
                        escalation_after_dismiss = %s,
                        post_severity_avg = %s,
                        evaluation_windows = %s,
                        evaluated_at = NOW()
                    WHERE user_id = %s AND decision_id = %s
                    """,
                    (
                        decision_effective,
                        decision_failed,
                        escalation_after_dismiss,
                        post_severity_avg,
                        evaluation_windows,
                        user_id,
                        decision_id,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error("Failed to persist decision effectiveness: %s", exc)
        finally:
            conn.close()

