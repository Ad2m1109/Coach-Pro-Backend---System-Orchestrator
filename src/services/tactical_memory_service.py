import logging
from typing import List, Dict, Any, Optional
from database import Connection

logger = logging.getLogger("TacticalMemoryService")

class TacticalMemoryService:
    def __init__(self, db: Connection):
        self.db = db

    def get_player_history(self, user_id: str, player_id: str, metric_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves historical insights for a specific player and metric within the last 90 days.
        """
        sql = """
            SELECT metric_value, window_definition, created_at
            FROM tactical_insights
            WHERE user_id = %s 
              AND subject_type = 'player' 
              AND subject_id = %s 
              AND metric_name = %s
              AND created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
            ORDER BY created_at DESC
            LIMIT %s
        """
        try:
            with self.db.cursor() as cursor:
                cursor.execute(sql, (user_id, player_id, metric_name, limit))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"[TacticalMemory] Error retrieving player history: {str(e)}")
            return []

    def get_formation_history(self, user_id: str, formation_id: str, metric_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves historical insights for a specific formation and metric within the last 90 days.
        """
        sql = """
            SELECT metric_value, window_definition, created_at
            FROM tactical_insights
            WHERE user_id = %s 
              AND subject_type = 'formation' 
              AND subject_id = %s 
              AND metric_name = %s
              AND created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
            ORDER BY created_at DESC
            LIMIT %s
        """
        try:
            with self.db.cursor() as cursor:
                cursor.execute(sql, (user_id, formation_id, metric_name, limit))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"[TacticalMemory] Error retrieving formation history: {str(e)}")
            return []

    def compute_trend_consistency(self, insights: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Computes a consistency score for trend-type metrics.
        Requires at least 3 historical windows.
        """
        if not insights or len(insights) < 3:
            return None

        # Only count windows where trend is positive (> 0)
        total_windows = len(insights)
        positive_windows = sum(1 for row in insights if row['metric_value'] > 0)
        
        consistency_percentage = round((positive_windows / total_windows) * 100)

        return {
            "positive_windows": positive_windows,
            "total_windows": total_windows,
            "consistency_percentage": consistency_percentage
        }

    def format_memory_block(self, insights: List[Dict[str, Any]]) -> Optional[str]:
        """
        Formats a list of insights into a structured TACTICAL_MEMORY block for the LLM.
        """
        if not insights:
            return None

        # Take the window definition from the most recent one for the header
        window = insights[0].get('window_definition', 'unknown')
        
        lines = [f"## TACTICAL_MEMORY (window: {window})", "---"]
        for row in insights:
            date_str = row['created_at'].strftime('%Y-%m-%d')
            val = row['metric_value']
            # Format as float if it has decimals, else int
            val_str = f"{val:+.2f}" if isinstance(val, (float, int)) else str(val)
            lines.append(f"{date_str}: {val_str}")

        return "\n".join(lines)
