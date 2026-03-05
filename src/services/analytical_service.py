"""
Analytical Service — Server-Side Statistics
===========================================

Performs deterministic SQL calculations for complex queries.
Ensures calculations happen in Python/SQL, not the LLM.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from database import Connection

logger = logging.getLogger("AnalyticalService")

class AnalyticalService:
    def __init__(self, db: Connection, user_id: str):
        self.db = db
        self.user_id = user_id

    def resolve_query_type(self, question: str) -> Optional[str]:
        """Maps question keywords to analytical functions."""
        q = question.lower()
        if any(w in q for w in ["improve", "trend", "increase"]):
            return "top_improvers"
        if any(w in q for w in ["highest", "average"]):
            return "highest_average_rating"
        if "formation" in q and ("press" in q or "pressure" in q):
            return "best_pressing_formation"
        return None

    def run_analytics(self, question: str) -> Optional[str]:
        """Orchestrates query resolution and execution."""
        query_type = self.resolve_query_type(question)
        if not query_type:
            return None

        start_time = time.time()
        result_text = ""
        
        try:
            if query_type == "top_improvers":
                result_text = self._get_top_improving_players()
            elif query_type == "highest_average_rating":
                result_text = self._get_highest_average_rating()
            elif query_type == "best_pressing_formation":
                result_text = self._get_best_pressing_formation()

            latency = round(time.time() - start_time, 2)
            logger.info(f"[AnalyticalService] Query type: {query_type} | Latency: {latency}s")
            return result_text
        except Exception as e:
            logger.error(f"[AnalyticalService] Error running {query_type}: {str(e)}")
            return "Error calculating analytical results."

    def _get_top_improving_players(self) -> str:
        """
        Calculates sprint intensity trend: 
        trend = AVG(sprint_count_last_5) - AVG(sprint_count_previous_5)
        Requires min 10 matches per player.
        """
        sql = """
            WITH indexed_stats AS (
                SELECT 
                    pms.player_id,
                    p.name AS player_name,
                    pms.sprint_count,
                    m.date_time,
                    ROW_NUMBER() OVER(PARTITION BY pms.player_id ORDER BY m.date_time DESC) as rn
                FROM player_match_statistics pms
                JOIN matches m ON pms.match_id = m.id
                JOIN players p ON pms.player_id = p.id
                JOIN teams t ON p.team_id = t.id
                WHERE t.user_id = %s
            ),
            last_window AS (
                SELECT player_id, player_name, AVG(sprint_count) as avg_last
                FROM indexed_stats WHERE rn <= 5 GROUP BY player_id, player_name
                HAVING COUNT(*) = 5
            ),
            prev_window AS (
                SELECT player_id, AVG(sprint_count) as avg_prev
                FROM indexed_stats WHERE rn > 5 AND rn <= 10 GROUP BY player_id
                HAVING COUNT(*) = 5
            )
            SELECT lw.player_name, (lw.avg_last - pw.avg_prev) as trend
            FROM last_window lw
            JOIN prev_window pw ON lw.player_id = pw.player_id
            WHERE (lw.avg_last - pw.avg_prev) > 0
            ORDER BY trend DESC
            LIMIT 5
        """
        with self.db.cursor() as cursor:
            cursor.execute(sql, (self.user_id,))
            rows = cursor.fetchall()

        if not rows:
            return "Insufficient data for trend analysis (requires 10+ matches per player)."

        header = "## ANALYTICAL_RESULT: TOP IMPROVING PLAYERS (Sprint Intensity Trend)\n"
        body = "\n".join([f"- {r['player_name']}: trend index +{r['trend']:.2f}" for r in rows])
        return header + body

    def _get_highest_average_rating(self) -> str:
        """Computes top 5 players by average rating over their last 5 matches."""
        sql = """
            WITH indexed_stats AS (
                SELECT 
                    pms.player_id,
                    p.name AS player_name,
                    pms.rating,
                    m.date_time,
                    ROW_NUMBER() OVER(PARTITION BY pms.player_id ORDER BY m.date_time DESC) as rn
                FROM player_match_statistics pms
                JOIN matches m ON pms.match_id = m.id
                JOIN players p ON pms.player_id = p.id
                JOIN teams t ON p.team_id = t.id
                WHERE t.user_id = %s
            )
            SELECT player_name, AVG(rating) as avg_rating
            FROM indexed_stats
            WHERE rn <= 5
            GROUP BY player_id, player_name
            HAVING COUNT(*) >= 1
            ORDER BY avg_rating DESC
            LIMIT 5
        """
        with self.db.cursor() as cursor:
            cursor.execute(sql, (self.user_id,))
            rows = cursor.fetchall()

        if not rows:
            return "No rating data available for analysis."

        header = "## ANALYTICAL_RESULT: HIGHEST AVERAGE RATINGS (Last 5 Matches)\n"
        body = "\n".join([f"- {r['player_name']}: {r['avg_rating']:.2f} avg rating" for r in rows])
        return header + body

    def _get_best_pressing_formation(self) -> str:
        """Finds formation with highest average pressures."""
        sql = """
            SELECT f.name as formation_name, AVG(mts.pressures) as avg_pressures
            FROM match_team_statistics mts
            JOIN match_lineup ml ON mts.match_id = ml.match_id AND mts.team_id = ml.team_id
            JOIN formations f ON ml.formation_id = f.id
            JOIN teams t ON mts.team_id = t.id
            WHERE t.user_id = %s AND ml.is_starting = 1
            GROUP BY f.name
            ORDER BY avg_pressures DESC
            LIMIT 3
        """
        with self.db.cursor() as cursor:
            cursor.execute(sql, (self.user_id,))
            rows = cursor.fetchall()

        if not rows:
            return "Insufficient data to analyze formation pressing success."

        header = "## ANALYTICAL_RESULT: BEST PRESSING FORMATIONS (Avg Pressures)\n"
        body = "\n".join([f"- {r['formation_name']}: {r['avg_pressures']:.1f} avg pressures" for r in rows])
        return header + body
