"""
Retrieval Service — Targeted Database Access
============================================

Fetches specific metrics based on intent with strict limits and timeouts.
Ensures user-level data isolation.
"""

import time
from typing import List, Dict, Any, Optional
from database import Connection
from services.team_service import TeamService
from services.player_service import PlayerService
from services.match_service import MatchService
from services.player_match_statistics_service import PlayerMatchStatisticsService

# --- Config ---
QUERY_TIMEOUT = 2.0  # seconds
MAX_PLAYERS = 5
MAX_MATCHES = 3


class RetrievalService:
    def __init__(self, db: Connection, user_id: str):
        self.db = db
        self.user_id = user_id
        self.team_service = TeamService(db)
        self.player_service = PlayerService(db)
        self.match_service = MatchService(db)
        self.start_time = time.time()

    def _check_timeout(self):
        """Raise error if the 2s timeout is exceeded."""
        if time.time() - self.start_time > QUERY_TIMEOUT:
            raise TimeoutError("Retrieval deadline exceeded")

    def get_team_context_ids(self) -> List[str]:
        self._check_timeout()
        teams = self.team_service.get_all_teams(self.user_id)
        return [t.id for t in teams]

    def retrieve_player_stats(self, query: str) -> List[Dict[str, Any]]:
        """Find relevant players and their recent stats with timeout protection."""
        try:
            team_ids = self.get_team_context_ids()
            if not team_ids: return []

            # 1. Keyword search for players
            all_players = self.player_service.get_all_players(team_ids)
            self._check_timeout()
            
            target_players = []
            q_lower = query.lower()
            for p in all_players:
                if p.name.lower() in q_lower or (p.jersey_number and str(p.jersey_number) in query):
                    target_players.append(p)
            
            target_players = target_players[:MAX_PLAYERS]
            if not target_players and ("player" in q_lower or "doing" in q_lower):
                target_players = all_players[:2] # Top 2 fallback

            results = []
            for p in target_players:
                self._check_timeout()
                with self.db.cursor() as cursor:
                    sql = """
                        SELECT s.*, m.date_time, ht.name as home_team, at.name as away_team
                        FROM player_match_statistics s
                        JOIN matches m ON s.match_id = m.id
                        JOIN teams ht ON m.home_team_id = ht.id
                        JOIN teams at ON m.away_team_id = at.id
                        WHERE s.player_id = %s
                        ORDER BY m.date_time DESC
                        LIMIT %s
                    """
                    cursor.execute(sql, (p.id, MAX_MATCHES))
                    stats = cursor.fetchall()
                    results.append({
                        "player_name": p.name,
                        "jersey": p.jersey_number,
                        "recent_stats": stats
                    })
            return results
        except TimeoutError:
            return [] # Fail gracefully

    def retrieve_match_stats(self) -> List[Dict[str, Any]]:
        try:
            team_ids = self.get_team_context_ids()
            if not team_ids: return []

            matches = self.match_service.get_all_matches(team_ids)
            self._check_timeout()
            
            completed = [m for m in matches if m.status and m.status.value == "completed"]
            completed.sort(key=lambda x: x.date_time or "", reverse=True)
            
            return [m.model_dump() for m in completed[:MAX_MATCHES]]
        except TimeoutError:
            return []

    def retrieve_team_summary(self) -> List[Dict[str, Any]]:
        try:
            self._check_timeout()
            teams = self.team_service.get_all_teams(self.user_id)
            return [{"name": t.name, "colors": f"{t.primary_color}/{t.secondary_color}"} for t in teams]
        except TimeoutError:
            return []
