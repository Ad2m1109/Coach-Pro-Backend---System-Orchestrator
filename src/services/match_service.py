from typing import List, Optional
from models.match import Match, MatchCreate
from models.team import Team
from models.event import Event
from models.formation import Formation
from models.player import Player
from models.match_details import MatchDetails, TeamLineup, PlayerWithPosition
from .match_lineup_service import MatchLineupService
from .player_match_statistics_service import PlayerMatchStatisticsService
from .match_team_statistics_service import MatchTeamStatisticsService
from .match_event_service import MatchEventService
from .player_service import PlayerService
from .formation_service import FormationService
import uuid

class MatchService:
    _video_anchor_schema_checked = False

    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.match_lineup_service = MatchLineupService(db_connection)
        self.player_stats_service = PlayerMatchStatisticsService(db_connection)
        self.team_stats_service = MatchTeamStatisticsService(db_connection)
        self.match_event_service = MatchEventService(db_connection)
        self.player_service = PlayerService(db_connection)
        self.formation_service = FormationService(db_connection)
        self._ensure_video_anchor_schema()

    def _ensure_video_anchor_schema(self):
        if MatchService._video_anchor_schema_checked:
            return
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) AS count
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name = 'matches'
                  AND column_name = 'video_anchor_seconds'
                """
            )
            exists = cursor.fetchone()
            if not exists or int(exists["count"]) == 0:
                cursor.execute("ALTER TABLE matches ADD COLUMN video_anchor_seconds INT NULL")
                self.db_connection.commit()
        MatchService._video_anchor_schema_checked = True

    def create_match(self, match: MatchCreate, user_team_ids: List[str]) -> Match:
        with self.db_connection.cursor() as cursor:
            # Validate home_team_id and away_team_id against user_team_ids
            if match.home_team_id not in user_team_ids:
                raise ValueError(f"Home team with ID {match.home_team_id} not owned by current user.")
            # Note: away_team_id might be an opponent not owned by the user, so only validate if it's a user's team
            # For simplicity, we assume if away_team_id is a user's team, it must be in user_team_ids

            new_match_id = str(uuid.uuid4())
            sql = "INSERT INTO matches (id, home_team_id, away_team_id, date_time, venue, event_id, status, home_score, away_score, video_anchor_seconds) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (new_match_id, match.home_team_id, match.away_team_id, match.date_time, match.venue, match.event_id, match.status.value if match.status else None, match.home_score, match.away_score, match.video_anchor_seconds))
            self.db_connection.commit()

            return self.get_match(new_match_id, user_team_ids)

    def get_match(self, match_id: str, user_team_ids: List[str]) -> Optional[Match]:
        with self.db_connection.cursor() as cursor:
            sql = """
                SELECT
                    m.*,
                    ht.name as home_team_name,
                    at.name as away_team_name,
                    e.name as event_name
                FROM matches m
                JOIN teams ht ON m.home_team_id = ht.id
                JOIN teams at ON m.away_team_id = at.id
                LEFT JOIN events e ON m.event_id = e.id
                WHERE m.id = %s AND (m.home_team_id IN %s OR m.away_team_id IN %s)
            """
            cursor.execute(sql, (match_id, user_team_ids, user_team_ids))
            match_data = cursor.fetchone()
            if match_data:
                return Match(**match_data)
            return None

    def get_all_matches(self, user_team_ids: List[str], status: Optional[str] = None, event_id: Optional[str] = None) -> List[Match]:
        if not user_team_ids:
            return [] # No teams, no matches
        with self.db_connection.cursor() as cursor:
            sql = """
                SELECT
                    m.id,
                    m.home_team_id,
                    m.away_team_id,
                    m.date_time,
                    m.venue,
                    m.event_id,
                    m.status,
                    m.home_score,
                    m.away_score,
                    m.video_anchor_seconds,
                    ht.name as home_team_name,
                    at.name as away_team_name,
                    e.name as event_name
                FROM matches m
                JOIN teams ht ON m.home_team_id = ht.id
                JOIN teams at ON m.away_team_id = at.id
                LEFT JOIN events e ON m.event_id = e.id
                WHERE m.home_team_id IN %s OR m.away_team_id IN %s
            """

            params = [user_team_ids, user_team_ids]
            conditions = []

            if status:
                conditions.append("m.status = %s")
                params.append(status)
            if event_id:
                conditions.append("m.event_id = %s")
                params.append(event_id)

            if conditions:
                sql += " AND " + " AND ".join(conditions)

            cursor.execute(sql, tuple(params))
            matches_data = cursor.fetchall()
            return [Match(**match) for match in matches_data]

    def update_match(self, match_id: str, match_update: MatchCreate, user_team_ids: List[str]) -> Optional[Match]:
        if match_update.home_team_id not in user_team_ids:
            raise ValueError(f"Home team with ID {match_update.home_team_id} not owned by current user.")
        with self.db_connection.cursor() as cursor:
            sql = "UPDATE matches SET home_team_id = %s, away_team_id = %s, date_time = %s, venue = %s, event_id = %s, status = %s, home_score = %s, away_score = %s WHERE id = %s AND (home_team_id IN %s OR away_team_id IN %s)"
            cursor.execute(sql, (match_update.home_team_id, match_update.away_team_id, match_update.date_time, match_update.venue, match_update.event_id, match_update.status.value if match_update.status else None, match_update.home_score, match_update.away_score, match_id, user_team_ids, user_team_ids))
            self.db_connection.commit()
            return self.get_match(match_id, user_team_ids)

    def delete_match(self, match_id: str, user_team_ids: List[str]) -> bool:
        with self.db_connection.cursor() as cursor:
            sql = "DELETE FROM matches WHERE id = %s AND (home_team_id IN %s OR away_team_id IN %s)"
            cursor.execute(sql, (match_id, user_team_ids, user_team_ids))
            self.db_connection.commit()
            return cursor.rowcount > 0

    def get_match_details(self, match_id: str, user_team_ids: List[str], user_id: str) -> Optional[MatchDetails]:
        match_info = self.get_match(match_id, user_team_ids)
        if not match_info:
            return None

        all_lineups = self.match_lineup_service.get_all_match_lineups(user_team_ids, match_id)
        player_stats = self.player_stats_service.get_all_player_match_statistics(match_id, user_team_ids)
        team_stats = self.team_stats_service.get_by_match_id(match_id=match_id, user_team_ids=user_team_ids)
        events = self.match_event_service.get_all_match_events(match_id=match_id)

        def get_team_lineup(team_id: str, team_name: str) -> TeamLineup:
            team_lineups = [lu for lu in all_lineups if lu.team_id == team_id]
            formation = None
            if team_lineups and team_lineups[0].formation_id:
                formation = self.formation_service.get_formation(team_lineups[0].formation_id, user_id)

            player_ids = [lu.player_id for lu in team_lineups]
            players = [self.player_service.get_player(pid, user_team_ids) for pid in player_ids]
            
            players_with_position = []
            for lu in team_lineups:
                player = next((p for p in players if p and p.id == lu.player_id), None)
                if player:
                    player_data = player.model_dump()
                    player_data['position_in_formation'] = lu.position_in_formation
                    players_with_position.append(PlayerWithPosition(**player_data))

            return TeamLineup(
                team_id=team_id,
                team_name=team_name,
                formation=formation,
                players=players_with_position
            )

        home_lineup = get_team_lineup(match_info.home_team_id, match_info.home_team_name)
        away_lineup = get_team_lineup(match_info.away_team_id, match_info.away_team_name)

        return MatchDetails(
            match_info=match_info,
            home_lineup=home_lineup,
            away_lineup=away_lineup,
            events=events,
            player_stats=player_stats,
            team_stats=team_stats,
        )

    def set_video_anchor(
        self,
        match_id: str,
        video_anchor_seconds: int,
        user_team_ids: List[str],
        overwrite: bool = False,
    ) -> Optional[int]:
        match = self.get_match(match_id, user_team_ids)
        if not match:
            return None

        with self.db_connection.cursor() as cursor:
            cursor.execute(
                "SELECT video_anchor_seconds FROM matches WHERE id = %s LIMIT 1",
                (match_id,),
            )
            row = cursor.fetchone()
            existing_anchor = row["video_anchor_seconds"] if row else None
            if existing_anchor is not None and not overwrite:
                raise ValueError("Anchor already set. Use force=true to overwrite.")

            cursor.execute(
                "UPDATE matches SET video_anchor_seconds = %s WHERE id = %s",
                (video_anchor_seconds, match_id),
            )
            self.db_connection.commit()
            return video_anchor_seconds

    def reset_video_anchor(self, match_id: str, user_team_ids: List[str]) -> bool:
        match = self.get_match(match_id, user_team_ids)
        if not match:
            return False
        with self.db_connection.cursor() as cursor:
            cursor.execute("UPDATE matches SET video_anchor_seconds = NULL WHERE id = %s", (match_id,))
            self.db_connection.commit()
            return True
